---
name: video-txt
description: 把本地视频/音频变成中文字幕视频、外挂字幕、配音或文字稿。用于"把 xx.mp4 翻译成中文硬字幕/硬编码字幕"、"给这个视频加中文字幕"、"这段视频转成文字"、"做中文配音"、"只要一份 .zh.srt"、"字幕有几句听错了要重识别"、"检查一下这份字幕质量"。Also triggers on burn/hardcode Chinese subtitles, transcribe a video, dub, translate an .srt.
---

# 用自然语言驱动 video-txt

用户说的是意图，这里把意图落到一条 `video-txt` 命令上。**先选命令，再补参数，最后跑。**

## 每条命令都长这样

```bash
cd /Users/sun/Documents/docx/video-txt
uv run --python 3.12 video-txt <子命令> '/绝对路径/视频.mp4' --provider lmstudio ...
```

- 路径一律用**绝对路径 + 单引号**，中文名和空格才不会散架。
- 翻译默认 `--provider lmstudio`（本机、不花钱、不用 key）。用户说"快一点/用云端/DeepSeek"才换 `--provider deepseek`。
- 产物默认落在**视频所在目录**，不是本仓库。

## 从意图到命令

| 用户想要 | 跑什么 | 出什么 |
| --- | --- | --- |
| 中文硬字幕成片（"硬编码字幕""烧进去"） | `run "$V" --provider lmstudio --mux-mode hard` | `视频.zh-burned.mp4` |
| 可开关的中文字幕单文件 | `run "$V" --provider lmstudio --mux-mode soft` | `视频.zh-subbed.mp4` |
| 片源自带官方中文字幕轨（**先查这个**） | `ffprobe` 探轨 → `ffmpeg -map 0:<轨号> -c:s srt` 抽出 | `视频.zh.srt`，两秒，无损 |
| 只要中文字幕文件（**`.mkv` 电影默认走这条**） | `transcribe "$V" -f srt --language <src>` 然后 `translate '视频.srt' --provider lmstudio` | `视频.srt` + `视频.zh.srt` |
| 中文配音 | `dub "$V" --provider lmstudio` | `视频.zh-dubbed.mp4` |
| 只要文字稿 | `transcribe "$V"`（要字幕加 `-f srt`） | `视频.txt` / `.srt` |
| 已有原文 `.srt`，别重转写 | 上面任意一条加 `--subtitle '/绝对路径/字幕.srt'` | — |
| 已有 `.srt`，只想翻译 | `translate '/绝对路径/字幕.srt' --provider lmstudio` | `字幕.zh.srt` |
| 字幕跨行过长、一次铺满屏 | 加 `--refine-subtitles`（不能和 `--subtitle` 同用） | — |
| 只有几句听错 | `retranscribe-range "$V" --subtitle "$S" --from 00:19:30 --to 00:20:10` | `字幕.repaired.srt` |
| 先检查字幕质量 | `audit "$S" --media "$V" --language en` | `字幕.audit.json` |
| 安全清理坏字幕块 | `clean "$S" -o '/绝对路径/字幕.clean.srt'` | 新 `.srt` |
| 同一部长片要反复精修 | `project init` → `project status` → `project run` | `project.video-txt.json` |

同名 `.srt` 就在视频旁边时会**自动复用**，不必加 `--subtitle`。

## 跑之前必须判断的六件事

**第 0 条最省事，永远先做。**

0. **片子自带官方中文字幕吗**——`.mkv` / `.mp4` 只要是 WEB-DL、WEBRip、蓝光重编码这类片源，多半封了几十条官方字幕轨。先探一眼，别上来就转写：
   ```bash
   ffprobe -v error -select_streams s -show_entries stream=index:stream_tags=language,title -of csv=p=0 "$V"
   ```
   看到 `chi` / `zho` / `Chinese Simplified` 就**直接抽出来**，流拷贝，两秒钟，原片不动：
   ```bash
   ffmpeg -v error -i "$V" -map 0:<轨号> -c:s srt "${V%.*}.zh.srt"
   ```
   抽完照样跑 `video-txt audit "${V%.*}.zh.srt" --media "$V" --language zh` 验一遍。
   官方字幕是人工翻译，剧名、专有名词、歌词都对，Whisper 转写 + 本地模型翻译在这些地方必错，还要多等十几分钟——**同样的产物，选便宜且更好的那条**。
   输出名跟视频同前缀，播放器才会自动挂上。有繁体（`Chinese Traditional`）或原文轨时一并告诉用户，抽哪条都是秒出。
   **没有中文轨也别急着开 Whisper——先看有没有官方原文轨。** 实测 Barry S03（HMAX WEBRip）8 集都没有
   中文轨，但每集都封了官方英文轨，抽出来直接当翻译的原文即可：省掉每集十几分钟转写，而且人名、剧名、
   专有名词不会听错——Whisper 恰恰在这些地方必错，也正是术语表最难补救的部分。
   ```bash
   ffmpeg -v error -i "$V" -map 0:<原文轨号> -c:s srt "${V%.*}.srt"   # 注意无 .zh，这是 translate 的输入
   ```
   同语言有 `SDH` 和非 SDH 两条时选**非 SDH**——SDH 夹着 `[door creaks]` 这类音效描述，翻出来是噪音。
   抽完直接 `translate`，`transcribe` 整步跳过。
   只有**中文轨和原文轨都没有**时才真的走转写。用户明说"我要机器转写稿"或"重做一版字幕"时也照办。

1. **本机模型开着吗**——只要这条命令真要翻译（`run` / `dub` / `translate` 且 `--provider lmstudio`），先确认：
   ```bash
   uv run python -c "from video_txt.translate import loaded_local_model; print(loaded_local_model('http://localhost:1234/v1'))"
   ```
   报错就告诉用户去 LM Studio 加载模型、打开本地服务器，别硬跑。
2. **源语言要不要钉死**——片头是音乐/静音/夹英文时 Whisper 会认错语种，整片跑偏。电影一律显式 `--language ja|en|ko|fr...`，并加 `--whisper-arg=--condition_on_previous_text --whisper-arg=False` 防重复循环。看不出来就问用户源语言是什么。
3. **源文件是 `.mkv` 电影吗**——是就**默认只出外挂 `.zh.srt`，既不烧硬字幕也不做软封装**，原片一个字节都不碰，画质无损。走上表"只要中文字幕文件"那一行，跑完告诉用户把 `.zh.srt` 拖进 IINA / VLC 即可。
   硬字幕要把整部片重编码（4K 一个多小时、画质有损），软封装虽是流拷贝无损，但要多占一份视频的空间——两者都只在**用户明确说要成片、要单文件、要发给别人**时才做，做之前把代价说清楚。
4. **原片底部已有字幕吗**——有就加 `--hard-subtitle-layout top`，新字幕走顶部，别互相压。

5. **是剧集/系列片、要翻不止一集吗**——是就**先写术语表，再开翻**。逐集独立翻译时模型没有跨集记忆，
   人名各集不一，同一集里也会混用（实测 Barry S01 无术语表：E04 出现「巴里」18 次、`Barry` 10 次；
   E03「莎莉」2 次、`Sally` 11 次）。
   开翻前先扫一遍原文字幕，把反复出现的人名、地名、机构名、称呼列成一份 JSON 术语表，全季共用一份，
   每集都带同一个 `--term-file`（格式见 README「项目术语表与翻译审计」）：
   ```bash
   T='/绝对路径/剧名.terms.json'   # {"schema":"video-txt.terminology","version":1,"terms":[
                                   #   {"source":"Barry","target":"巴里","match":"word"},
                                   #   {"source":"NoHo Hank","target":"诺霍·汉克","match":"phrase"}]}
   for f in "$D"/*.mkv; do
     uv run --python 3.12 video-txt translate "${f%.*}.srt" --provider lmstudio --term-file "$T"
   done
   ```
   单词人名用 `match: "word"`（`Barry` 不会误中 `Barrymore`），多词名用 `"phrase"`；已经流传的错译放
   `aliases`，审计会按 error 拦下来。
   术语表在**每个批次**的 prompt 里都要重复一遍，而 `--provider lmstudio` 的 `--batch-chars` 默认只有
   **800**，批次数远比想象多——实测 8 集切成 186 批，一份 12 条的术语表就让输入 token 从 12.9 万涨到
   16.1 万（**+25%**）。本机模型不花钱，这点开销无所谓；但**先写表远比事后重跑整季便宜**。
   已经翻完才想统一：修好术语表加 `--retranslate` 重跑，代价是全季重译（实测 8 集约 30 分钟）。

   **换季/换片复用上一季术语表时，先扫新季的新名字，别直接套。** 旧词条会把形近的新角色吸附过去：
   实测把 S01 的 `Ryan Madison`→瑞恩·麦迪逊 原样带进 Barry S02，模型把第二季新角色 `Aaron Ryan`
   全译成了「瑞恩·麦迪逊」——张冠李戴，4 处，而且两个名字在 S02 原文里**都真的出现**，删掉旧词条也不对，
   必须给新名字**单独立目**（`Aaron Ryan`→亚伦·瑞恩）。所以复用的正确做法是：抽完新季原文后，
   拿旧术语表的词过滤一遍高频专有名词，把**未覆盖的新名字补成新词条**，旧映射一字不改（跨季才一致）。
   这类错误审计能抓（`glossary_target_missing` 是 error，命令返回非零），**别看到非零就当误报放过**——
   实测 S02 三集报错，两条是真错（`Aaron Ryan` 张冠李戴、`Sasha` 被译成「莎莉」串成了另一个角色），
   一条是异写（Esther 写成「埃斯特」）。
   修法统一走 `aliases`：模型实际吐出的错误形态填进对应词条的 `aliases`，`enforce_terminology` 会自动
   规范化，不用手改字幕。它的替换**有原文门控**（`terminology.py` 的 `_source_matches`）——别名只在原文
   确实含该词条的字幕块内生效，所以给 `Sasha` 加别名「莎莉」不会误伤真正的 Sally。加这种跨角色别名前，
   先确认两个名字在原文里没有同块共现，确认完再补跑受影响的那几集（`--overwrite`），其余集不动。

   **补跑前先给每条 error 定性，`aliases` 不是万能修法。** 实测 Barry S03 的四条 error 是三种病：
   - **异写**（`Krauss`→「克拉乌斯」、`Ken Goulet`→「肯尼思·古莱」）和**张冠李戴**（`Berkman` 被译成
     「布洛克」，串到巴里的艺名 Barry Block 上）——这两类走 `aliases` + `--overwrite` 补跑，有效。
   - **漏译**（`Lindsay` 整个名字不见，译文只剩前半句）——`aliases` **救不了**：规范化是把错误形态替换成
     批准形态，而译文里根本没有可替换的目标。重跑也会**原样复现**同一处漏译（实测 E04 跑两遍都漏在同一块），
     只能定点把那一块补回来。
   **`--overwrite` 是重新掷骰子，不是纯修复。** 实测 E04 补跑后 error 从 1 条变成 2 条：旧的漏译没修好，
   又新掷出一个异写（`Elena`→「埃莱娜」）。所以问题只剩零星几处时，**定点修比重跑整集更稳也更快**；
   重跑只留给「补了别名、且该集确实有多处异写」的场合。

## 执行方式

- 转写很慢（24 分钟视频 CPU 约 7.5 分钟，109 分钟 4K 约 35 分钟）。**超过几分钟的一律 `run_in_background` 跑**，跑完再汇报，不要卡住会话。Apple Silicon 上先 `uv sync --extra mlx` 再加 `--backend mlx-whisper` 能快数倍。
- 要重编码、要覆盖、或用户对参数还没拍板时，**先 `--dry-run` 把命令打给用户看**，确认再真跑。
- 已存在的阶段会自动跳过，改字幕样式重跑很便宜。强制重做才用 `--retranscribe` / `--retranslate` / `--overwrite-video`，且只在用户明说时加。
- 转写完会自动体检（重复空转、语种不符、只覆盖前半段）。体检不过就停下来报给用户，别接着白翻一遍。
- **整季/多集翻完，收尾跑一遍自己的全季自检，别只信每集的 `translation-audit.json`。** 两个理由：一是
  **手改过字幕后那份报告就过期了**（它是 `translate` 当时写的，不会跟着你的编辑更新）；二是**逐块审计
  查不出跨块错位**——实测 S03E07 模型把 cue 5、6 合并成一句，导致 cue 6–9 的中文整体比台词**提前一格**，
  到 cue 11 才自己对回来：术语一条没少、每块单看都通顺、审计 0 error，但播放时字幕和口型对不上。
  自检查两样：
  1. **结构**——译文与原文逐块比对**块数、序号、时间轴**必须完全一致，且无空块（手改后尤其要查）；
  2. **术语落位**——原文块含词条、而同块译文没有批准写法的，挑出来逐条看上下文。
  第 2 项**误报很多，必须看完上下文再定性**：中英语序不同，名字合法地落在相邻块是常态（实测 S03 剩下
  八处提示全是这种，"…提供一些道歉的选项，巴里。" 接 "信手拈来。" 连起来读是对的，不用改）；
  但**连续几块都对不上**就是真错位，按 S03E07 那样重排译文、**时间轴一格不动**。
- 报告结果时，把生成的文件按绝对路径列出来。

## 需要更细的参数

字幕外观、术语表 `--term-file`、多音轨 `--audio-stream`、多说话人 `--diarize`、原声克隆
`--tts-engine f5-tts`、并发和凭据，全在 [README.md](../../../README.md)；技术取舍的实测数据在
[NOTES.md](../../../NOTES.md)。拿不准就 `uv run video-txt <子命令> --help`。

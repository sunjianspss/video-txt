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
| 只有几句**译**得不对（听对了、译错了） | 写 `revisions.json`，再 `revise '原文.srt' '译文.zh.srt' --revisions r.json` | `译文.zh.revised.srt` |
| 开翻整季前先建术语表 | `draft-terms "$D"/*.srt -o 剧名.terms.json` | `target` 待填的草稿 |
| 原文译文对照（**仅用户明说要双语时**） | 加 `--bilingual`，或 `bilingual '原文.srt' '译文.zh.srt'` | `译文.zh.bilingual.srt` |
| 要片子的配乐/背景音乐 | `music "$V"`，只要干净段加 `--clean` | `视频.music/` 下若干 .flac |
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
   开翻前要有一份全季共用的术语表，每集都带同一个 `--term-file`。**这一步不用手读，`draft-terms` 会扫**——
   它按「句中大写、几乎不写小写」认专有名词，把全季候选按出现次数排出来，`target` 一律留空等你填：
   ```bash
   T='/绝对路径/剧名.terms.json'
   uv run --python 3.12 video-txt draft-terms "$D"/*.srt -o "$T" --source-language en
   # 填完每条 target，删掉不值得立目的。target 为空时 load_terminology 会直接报错，草稿不会被误用
   for f in "$D"/*.mkv; do
     uv run --python 3.12 video-txt translate "${f%.*}.srt" --provider lmstudio --term-file "$T"
   done
   ```
   每条候选带 `count` 和两条 `examples`，够直接判断该译成什么、值不值得立目。
   **同名异写它会自己发现**并互相填进 `aliases`——实测 Toy Story 5 的字幕把同一角色拼成 `Jesse`(31 次)
   和 `Jessie`(26 次) 两种，人工表只有后者。
   **草稿是起点不是终点。** 实测 Barry S02 全季（4717 条字幕、8 集，`--min-count 8`）扫出 30 个候选，
   命中人工表 26/44，噪声 3 个；Toy Story 5（1683 条，`--min-count 3`）扫出 28 个，命中 7/11。
   **漏的全是低频名字**——Toy Story 漏的 4 条全片只出现 1–2 次（Monty 只说过一次），频率扫描
   找不到也不该编。反过来它扫出了人工表漏掉的真名字（Forky、Dolly、Zerg、Cleveland）。
   所以扫完仍要过一遍片里台词少但重要的配角。整季用 `--min-count 8` 左右，单片用默认 3。
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
   这一步交给 `--against`：旧表已覆盖的名字自动略去，只列新名字，并对**和旧词条共用一个词**的新名字
   直接打警告——`Aaron Ryan` 撞 `Ryan Madison` 正是这条规则要抓的：
   ```bash
   uv run --python 3.12 video-txt draft-terms "$D2"/*.srt -o /tmp/s02.new.json \
     --against '/绝对路径/剧名.terms.json'
   ```
   把填好的新词条并进旧表，**旧映射一个字都不要动**。
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
     只能定点补——**写进校订文件**（见下面「改译文」），别直接改 `.srt`，也别写一次性脚本。
   **`--overwrite` 是重新掷骰子，不是纯修复。** 实测 E04 补跑后 error 从 1 条变成 2 条：旧的漏译没修好，
   又新掷出一个异写（`Elena`→「埃莱娜」）。所以问题只剩零星几处时，**定点修比重跑整集更稳也更快**；
   重跑只留给「补了别名、且该集确实有多处异写」的场合。

6. **要做多说话人配音吗**——`--diarize` 的声学分割在配乐大的片子上**不可信，别拿它的结果直接开跑**。
   实测（2 分 43 秒的 ChatGPT 演示合集，人机对话，背景音乐从头铺到尾）：`--speakers 2` 把几个不同的
   真人和 ChatGPT 的应答混进同一个标签；放开让它自己数得到 4 个说话人，仍然把**一整句应答从中间
   劈成两个标签**。台词内容才是可靠依据——谁在提要求、谁在应答，读一遍原文一目了然。做法：
   - 先正常跑一次拿到 `<视频>.speakers.json`，再**按台词内容重写 `turns`**。最省事的写法是
     **一块字幕一条 turn**，直接用字幕自己的时间范围，标签写成 `HUMAN` / `ASSISTANT` 这种有意义的
     名字（`assign_speakers` 按重叠时长取多数，跨角色的块归给占比大的那一方）。
   - **`version` / `model` / `cache_key` 一个字都别动**，否则下次运行会重新跑 pyannote 把手改覆盖掉；
     重跑时 `--speakers` 等参数也要和 `cache_key` 里记的一致，不一致同样会重算。
   - **字幕块数变了（做过 `retranscribe-range`）必须按新时间轴重建 `speakers.json`**，
     旧 turn 的时间对不上新块，角色会整段错位。
   - **自动切的参考片段会混进对方的声音。** `pick_reference_window` 按"同一说话人的连续块"找 3–12 秒
     窗口，分割一错参考里就同时有两个人——实测两段参考各自都夹着对方一句，等于拿混合体克隆了两次，
     两个音色听着差不多而且都偏慢。**开跑前先 `cat` 一遍 `<视频>.dub-cache/reference/*.txt`**，
     确认那段话只有一个角色在说；不是就用 `--clone-reference 角色=片段.wav` 手工指定
     （旁边放同名 `.txt` 写清这段说了什么）。配 `--separate-bgm` 时从
     `<视频>.dub-cache/bgm/vocals.flac` 这条干声里切，参考里不会带配乐。
   - 换纯净参考的收益，同一条命令前后对比实测：语速 7.3→8.0 字/秒，需压缩改写的句子 20→9，
     最紧一句 2.63×→1.65×，超出槽位 1 句→0，时间轴漂移 0.4→0.0 秒。

## 提配乐 / 背景音乐

用户说"把这片子的配乐提出来""要背景音乐""这段音乐单独存一下"——走 `video-txt music`，
**不要自己拼 demucs 或 ffmpeg**：

```bash
uv run --python 3.12 video-txt music '/绝对路径/电影.mkv'            # 找出所有音乐段
uv run --python 3.12 video-txt music "$V" --clean --min-duration 15  # 只要没人说话的干净段
uv run --python 3.12 video-txt music "$V" --stems                    # 顺带导出全部分轨
```

产物是若干归一化的 `.flac` 加一份 `.music.json`，每段带 `voice_share`（多少比例被人声盖着）。
**要拿去用在别的视频里就加 `--clean`**：分离器在对白最响的地方瑕疵最多，只有没人说话的段落
听起来才不露馅。

**慢，第一次要跑分离**（几分钟）。但缓存和配音共用——片子如果跑过 `--separate-bgm` 就是秒出。
`run_in_background` 跑，别卡住会话。

### 它不判断有没有人在唱

分离器听到的只是"人声"，**唱和说一律进同一条轨**。用"人声占比高"猜歌曲实测会把一段 71 秒的对话
判成歌，所以工具不会仅凭音频给任何一段贴"歌曲"标签。用户要切某首插曲时只有两条路：

1. **自己给范围**：`--from 00:31:07 --to 00:34:20 --source mix`（歌要从原始混音切，
   没有唱的歌不是歌）
2. **指一份人工字幕**：`--subtitle` 会读里面 `♪` 标记的唱词来定位。**只有官方字幕轨有这个标记**
   ——实测 Barry S02E01 有 17 个 ♪ 块连成一首完整插曲，而 Whisper 转写稿 1683 块里一个都没有。

所以用户要提插曲时，**先按第 0 条探一下有没有官方字幕轨**；没有就问他歌大概在第几分钟，
用 `--from/--to`。别假装能自动找出来。

## 双语字幕：只在用户明确要求时才做

**默认单语，不要顺手加。** 用户说"加中文字幕""翻译成中文"，要的就是中文字幕——双语每块两行、
信息密度翻倍，正常观看时是干扰而不是加分。只有用户明说**"原文也要显示""中英对照""双语字幕"
"学英语用""校对要看原文"**这类要求时才加 `--bilingual`。拿不准就按单语做完，再问一句要不要原文。

要做的时候**不要自己写脚本把两个 `.srt` 拼起来**：

```bash
video-txt run "$V" --bilingual                    # 软字幕成片
video-txt run "$V" --bilingual --mux-mode hard    # 烧进画面
video-txt bilingual '原文.srt' '译文.zh.srt'      # 只要文件
```

默认译文在上、原文在下（`--order source-first` 反过来）。**合并后每块恒定两行**：一行译文、
一行原文，各自语言原有的换行会被压平（原文两行加译文两行会摞成四行，吃掉画面下半部分）。
个别很长的行渲染时仍会被播放器自动折到第三行，这跟单语字幕一样，交给渲染器。

**烧硬字幕时原文自动小一号 + 半透明**，不用另外调参数。合并要求两份字幕块数/序号/时间轴逐条
一致，对不上会直接报错——正常流程出来的文件天然满足，手改过就先跑 `audit-translation`。

校对译文时这个特别好用：原文译文并排，一眼能看出哪句译错了，挑出来写进校订文件。

**别拿 `audit` 去查双语字幕**：每块两行、行也更宽，必然报 `too_many_lines` 和 `line_too_wide`。
双语本来就比单语密，要审计就审合并之前那一份。

## 改译文：只改在校订文件里，绝不直接改 `.srt`

用户说"第 3 句意思反了""这里应该是 XX 不是 YY"——**不要打开 `.zh.srt` 改，也不要写一次性脚本**。
译文是生成物，下一次 `--retranslate` 或换个模型重跑就全没了；按字幕编号写死的脚本更危险：
`clean` 删一条幻觉、`retranscribe-range` 拼一段，下面所有编号都平移，修订会**静默地**写到别人的
台词上。实测一份 1683 条的电影字幕，上游只删 3 条，34 条按编号写死的修订全部错行，没有任何提示：

```
原 #110      -> 现 #108   'You can take today to figure out how to'   ← 锚定原文，跟着台词走
按编号写 #110 -> 'having...'                                           ← 错行
```

正确做法是一份**锚定原文台词**的 JSON：

```json
{
  "schema": "video-txt.revisions",
  "version": 1,
  "source_language": "en",
  "target_language": "zh-CN",
  "revisions": [
    { "source": "Where is Jessie?",
      "at": "00:31:07,120",
      "text": "翠丝在哪里？",
      "note": "模型把 Jessie 串成了另一个角色" }
  ]
}
```

写条目的规矩：

- `source` **从原文 `.srt` 里原样复制**那一行，不要自己回译、不要把两块合成一条。大小写和空白不敏感，
  但内容必须是原文里真实存在的一行。
- `at` 是那一行在原文 `.srt` 里的**开始时间戳**，同样原样复制。只有同一句台词全片说过多次时才**必须**填，
  但**建议每条都填**：既能当书签跳转，也能挡住以后新增重复台词造成的歧义。
- `text` 是定稿译文，`\n` 表示字幕内换行。
- `note` 写清为什么改，下次重跑时你自己会需要它。

跑：

```bash
uv run --python 3.12 video-txt revise '/绝对路径/原文.srt' '/绝对路径/译文.zh.srt' \
  --revisions '/绝对路径/revisions.json'
# → 译文.zh.revised.srt + 译文.zh.revised.revision-report.json
```

原文和译文都**只读**，输出必须是新路径。流水线里直接加 `--revisions`，翻译之后自动套回去，
**重翻多少次都不丢**：

```bash
video-txt run "$V" --term-file "$T" --revisions "$R"
video-txt dub "$V" --term-file "$T" --revisions "$R"
video-txt mux "$V" '原文.srt'  --revisions "$R"
```

### 报错怎么读

- **`anchors to a line that is not in the source subtitle`**——原文变了（重转写过，或 `clean` 删了那块）。
  去新原文里找到对应那行，把 `source`（和 `at`）更新成现在的写法。**不要**为了让命令跑过去就删掉这条修订。
- **`is said N times ... Add "at"`**——这句台词全片说过多遍。报错会列出候选时间戳，挑对应的填进 `at`。
- **`both correct cue N`**——两条修订锚到同一句了，留对的那条。
- **`already_current`（不是报错）**——这行译文已经和定稿一致，说明模型自己译对了，这条可以从文件里删掉。
- **`Anchor moved: cue N is now Xs from ...`（警告，不阻断）**——台词位置挪了。挪一两秒是重识别的正常现象；
  挪到几分钟就要怀疑匹配错了行，去核对一下。

**锚点匹配不上一律是硬错误，不会静默跳过。** 这是这个格式存在的理由：一份悄悄失效的校订文件比没有更糟，
命令照样返回 0，而每一条人工定稿都退回成了模型的说法。

### 和术语表怎么分工

- **同一个词永远这么译** → 术语表 `--term-file`。人名、地名、机构名、设备名。
- **这一句的意思错了** → 校订文件 `--revisions`。剧情语义、反话、语气、跨块错位重排、`aliases` 救不了的漏译。

口诀：能写成「X 一律译成 Y」的进术语表；只对这一句成立的进校订文件。
术语表能自动套到全片全季，校订文件不能也不该。

### 审计知道哪些行是人工定稿的

`revise` 跑完会**重新审计**，写出 `<译文>.revised.translation-audit.json`——这份描述的是真正出片的文件。
人工校订过的行，若命中 `glossary_target_missing` / `glossary_alias` / `source_text_residue` /
`source_text_unchanged` / `translation_unusually_long` 这五类**启发式**判断，会标成 `accepted`：
照样列在报告里，但不计入 error/warning，也不再拦下流水线——那一行已经有人读过并做了决定。

**结构性错误（块数、序号、时间轴、空译文）永远不会被标 accepted**：校订只换一句的文字，造不成这些错，
出现了就是别处出了问题。

只审计、不改文件时也能带上：

```bash
video-txt audit-translation '原文.srt' '译文.zh.srt' --term-file "$T" --revisions "$R"
```

### 长片/整季：写进项目文件

`project init` 加 `--revisions`，`project run` 就会自动带上。**改校订文件只让 `revise` 和成片过期，
不会让 `translate` 过期**——修一行不会触发整片重译（这正是 `revise` 单独成一个阶段的原因）。
`project status` 直接看得到：

```
translate: current
revise: stale — revisions changed
mux: stale — upstream revise is stale
```

## 执行方式

- 转写很慢（24 分钟视频 CPU 约 7.5 分钟，109 分钟 4K 约 35 分钟）。**超过几分钟的一律 `run_in_background` 跑**，跑完再汇报，不要卡住会话。Apple Silicon 上先 `uv sync --extra mlx` 再加 `--backend mlx-whisper` 能快数倍。
- 要重编码、要覆盖、或用户对参数还没拍板时，**先 `--dry-run` 把命令打给用户看**，确认再真跑。
- 已存在的阶段会自动跳过，改字幕样式重跑很便宜。强制重做才用 `--retranscribe` / `--retranslate` / `--overwrite-video`，且只在用户明说时加。
- 转写完会自动体检（重复空转、语种不符、只覆盖前半段）。体检不过就停下来报给用户，别接着白翻一遍。
- **整季/多集翻完，收尾跑一遍自己的全季自检，别只信每集的 `translation-audit.json`。** 两个理由：一是
  **手改过字幕后那份报告就过期了**（它是 `translate` 当时写的，不会跟着你的编辑更新）——
  走 `--revisions` 就没有这个问题，`revise` 阶段会对真正出片的那份重新审计；二是**逐块审计
  查不出跨块错位**——实测 S03E07 模型把 cue 5、6 合并成一句，导致 cue 6–9 的中文整体比台词**提前一格**，
  到 cue 11 才自己对回来：术语一条没少、每块单看都通顺、审计 0 error，但播放时字幕和口型对不上。
  自检查两样：
  1. **结构**——译文与原文逐块比对**块数、序号、时间轴**必须完全一致，且无空块（手改后尤其要查）；
  2. **术语落位**——原文块含词条、而同块译文没有批准写法的，挑出来逐条看上下文。
  第 2 项**误报很多，必须看完上下文再定性**：中英语序不同，名字合法地落在相邻块是常态（实测 S03 剩下
  八处提示全是这种，"…提供一些道歉的选项，巴里。" 接 "信手拈来。" 连起来读是对的，不用改）；
  但**连续几块都对不上**就是真错位，按 S03E07 那样重排译文、**时间轴一格不动**——重排结果写进校订文件。
- **装可选依赖必须一次装全。** `uv sync --extra X` 会把**不在本次列表里的可选依赖卸掉**——实测单独
  `uv sync --extra diarize` 把已经装好的 demucs 和 edge-tts 剪没了，紧接着那条带 `--separate-bgm`
  的命令跑完分离说话人才报缺依赖退出，白等一趟。要哪几个就写在同一行：
  `uv sync --extra dub --extra separate --extra diarize`。
- **pyannote 的 Hugging Face 门禁是三个仓库，不是两个。** 除了 `segmentation-3.0` 和
  `speaker-diarization-3.1`，pyannote 4.x 会把 3.1 内部重定向到
  `pyannote/speaker-diarization-community-1`，那个要**单独再点一次同意**，三个都过了才 load 得动。
  读 token 但没接受条款报的是 **403**（"not in the authorized list"），没 token 才是 401——
  看到 403 别以为 token 坏了。全程免费，gated 不等于收费。
- **`retranscribe-range` 不加 `--refine-subtitles`，常常原样复现同一处分块错位。** 实测重识别一段
  被劈在半句上的字幕（"…that it's just / slightly damaged…"），词一个没听错，**分块位置和上次
  一模一样**——问题出在 Whisper 的解码窗口，不在识别。加 `--refine-subtitles` 用词级时间戳重建块
  边界才真正解决（实测 130 个词重排成 13 块，最长 5.7 秒）。修完还要**手工看一眼首块**：
  `--padding` 会把上一块的尾巴连带重识别一遍，第一块常常重复上一块结尾的几个词，不删就会出现
  连着两块说同一句话。
- 报告结果时，把生成的文件按绝对路径列出来。

## 需要更细的参数

字幕外观、术语表 `--term-file`、多音轨 `--audio-stream`、多说话人 `--diarize`、原声克隆
`--tts-engine f5-tts`、并发和凭据，全在 [README.md](../../../README.md)；技术取舍的实测数据在
[NOTES.md](../../../NOTES.md)。拿不准就 `uv run video-txt <子命令> --help`。

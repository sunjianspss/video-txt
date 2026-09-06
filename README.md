# video-txt

把本地视频变成**中文字幕视频**或**中文配音视频**,一条命令跑完转写、翻译、封装。

做技术决定时的实测数据和取舍记在 [NOTES.md](NOTES.md),这里只讲怎么用。

版本演进与各阶段用法见 [CHANGELOG.md](CHANGELOG.md)；快速浏览见
[版本时间线](docs/version-timeline.md)。

## 快速开始

第一次用先装依赖,只做一次:

```bash
brew install ffmpeg
uv sync
```

翻译交给**本机 LM Studio**,不花钱也不用 key:装好 LM Studio,加载一个模型,
在 Developer 面板里把本地服务器打开(默认 `http://localhost:1234`)。这条命令只检查、不翻译,
打印出来的就是接下来会用到的模型:

```bash
uv run python -c "from video_txt.translate import loaded_local_model; print(loaded_local_model('http://localhost:1234/v1'))"
```

想用云端 DeepSeek(快很多,按量收费)就把下面所有的 `--provider lmstudio` 换成
`--provider deepseek`,并按文末「凭据约定」准备好 key。两者的取舍见
[「本机还是云端」](#本机还是云端)。

然后跑:

```bash
cd /Users/sun/Documents/docx/video-txt

uv run video-txt run '/绝对路径/你的视频.mp4' --provider lmstudio --mux-mode hard
```

跑完在**视频所在目录**得到四个文件:

| 文件 | 是什么 |
| --- | --- |
| `你的视频.srt` | Whisper 转写的原文字幕 |
| `你的视频.zh.srt` | 翻译后的中文字幕 |
| `你的视频.zh.translation-audit.json` | 原文/译文结构与翻译质量审计报告 |
| `你的视频.zh-burned.mp4` | 中文字幕烧进画面的成片 |

加 `--refine-subtitles` 时还会生成 `你的视频.words.json`,保存可复用、与具体 Whisper
实现无关的词级时间戳。

去掉 `--mux-mode hard` 就是软字幕版,输出 `你的视频.zh-subbed.mp4`,字幕可以在播放器里开关。
想先看看会执行什么、不真跑,加 `--dry-run`。

## 长视频项目配置（v0.6）

同一部长片需要反复精修时，用项目文件固定音轨、Whisper、翻译、术语表、字幕样式、配音和输出路径：

```bash
uv run --python 3.12 video-txt project init "$V" \
  -o project.video-txt.json \
  --audio-stream 2 --language en \
  --provider lmstudio --model '当前加载的模型名' \
  --term-file project.terms.json --mux-mode hard

uv run --python 3.12 video-txt project status project.video-txt.json
uv run --python 3.12 video-txt project run project.video-txt.json --dry-run
uv run --python 3.12 video-txt project run project.video-txt.json
```

- `project.video-txt.json` 只保存稳定声明；相对路径始终以该文件所在目录为基准。
- `.video-txt/state.json` 保存上次成功运行的内容/配置指纹、实际 Whisper 后端与模型、实际音轨及选择依据、阶段状态和产物。
- 视频、原文字幕、术语表和译文按内容 SHA-256 判断，单纯 touch 文件不会失效。
- 音轨或转写设置变化会使转写及下游过期；原文、术语表或翻译设置只从翻译向下失效；字幕样式只重跑 mux；音色只重跑配音并复用未变化的语音片段缓存。
- 项目 JSON 和 state 只记录凭据环境变量名，不保存 API key、token 或其他秘密；真正执行云端阶段时仍从环境变量或凭据文件读取。
- 没有 state 时，已有目标译文或成片视为外部文件，真实运行会拒绝覆盖；先 dry-run，或把文件移开/改输出路径。
- `project run --dry-run` 不写字幕、成片或 state。原有 `run`、`dub`、`translate` 等命令行为不变。

## 指定时间段局部重新识别（v0.7）

长片只有少量字幕听错时，不必重新识别整部电影。按绝对时间指定核心区间：

```bash
V='/绝对路径/电影.mkv'
S='/绝对路径/电影.srt'

uv run --python 3.12 video-txt retranscribe-range "$V" \
  --subtitle "$S" \
  --from 00:19:30 --to 00:20:10 \
  --language en --audio-stream 2 \
  -o '/绝对路径/电影.repaired.srt'
```

知道字幕位置时也可按一基序号选择（这里指 SRT 中第 320～335 条，而不是依赖文件里可能损坏的
显示编号）：

```bash
uv run --python 3.12 video-txt retranscribe-range "$V" \
  --subtitle "$S" --from-cue 320 --to-cue 335 \
  --language en --audio-stream 2 \
  -o '/绝对路径/电影.repaired.srt'
```

命令默认在核心区间前后各提取 1.5 秒上下文，可用 `--padding` 调整。它复用现有多音轨选择、
OpenAI Whisper / MLX Whisper、`--model`、`--initial-prompt`、`--whisper-arg` 和
`--refine-subtitles`；临时 16 kHz 单声道 WAV 及 Whisper 中间文件会自动清理。局部时间戳会加回
全片偏移，padding 中重复识别的边界字幕会被丢弃，结果重新排序并连续编号。

源字幕始终只读。省略 `-o` 时默认生成 `<字幕名>.repaired.srt`，同时写出
`<输出名>.retranscription-report.json`，记录核心/提取区间、音轨选择依据、删除和新增的字幕。
已有生成物默认不覆盖；确认替换时加 `--overwrite`。先加 `--dry-run` 可只查看音频提取、Whisper 和
替换计划，不写任何文件。

修好原文之后翻译不必从头再来。`--reuse` 指向修复前的原文字幕，读得一样的行直接沿用旧译文，
只有真正改过的那几行才发给模型：

```bash
uv run --python 3.12 video-txt translate '/绝对路径/电影.repaired.srt' \
  --reuse "$S" --provider lmstudio
```

旧译文默认按命名规则在 `$S` 旁边找（`电影.zh.srt`），不在那儿就用 `--reuse-translation` 指出来。
匹配只看文字（忽略空白与大小写），所以重新编号、时间轴微调都不影响沿用；同一句话在片中说过多次时，
取时间上最近的那条译文。终端会打印沿用了多少条、还剩多少条要翻译。`dub`、`run`、`mux` 也认这两个参数。

参考耗时:24 分钟的 480p 视频,CPU 转写 7.5 分钟 + 烧字幕 42 秒,翻译那一步走云端约 30 秒,
走本机模型看机器和模型大小,慢不少。

可选功能按需装,首次运行还要下模型权重:

| `--extra` | 装的是什么 |
| --- | --- |
| `dub` | 中文配音(edge-tts) |
| `separate` | 配音时保留背景音乐和环境声(Demucs 人声分离) |
| `mlx` | Apple Silicon GPU 转写,比 CPU 快数倍,另下 1.5G 权重 |
| `diarize` | 多说话人分离(pyannote) |
| `clone` | 原声克隆(F5-TTS) |

**要哪几个就写在同一行,别分几次装。** `uv sync` 每次都把环境对齐到本次给的 `--extra` 列表,
不在列表里的会被**卸掉**——单独跑一条 `uv sync --extra diarize` 会把之前装好的 Demucs 和 edge-tts
剪掉,下一条带 `--separate-bgm` 的命令才发现缺依赖:

```bash
uv sync --extra dub --extra separate --extra diarize
```

## 按场景选命令

下面都以 `V='/绝对路径/你的视频.mp4'` 为例,先 `V=...` 设一下会少打很多字。

| 想做什么 | 命令 |
| --- | --- |
| 中文硬字幕成片(最常用) | `uv run video-txt run "$V" --provider lmstudio --mux-mode hard` |
| 自动修掉过长、满屏的 Whisper 字幕 | 上面那条加 `--refine-subtitles` |
| 已有 `.srt`,先检查质量 | `uv run --python 3.12 video-txt audit '/绝对路径/字幕.srt'` |
| 已有 `.srt`,安全清理明显坏块 | `uv run --python 3.12 video-txt clean '/绝对路径/字幕.srt' -o '/绝对路径/字幕.clean.srt'` |
| 只重识别一小段错误字幕 | `uv run --python 3.12 video-txt retranscribe-range "$V" --subtitle source.srt --from 00:19:30 --to 00:20:10` |
| 修好原文后只翻译变过的行 | `uv run --python 3.12 video-txt translate source.repaired.srt --reuse source.srt` |
| 固定人名/术语译法并审计译文 | 翻译命令加 `--term-file project.terms.json`,详见 v0.5 |
| 双语或多音轨影片 | 正常传 `--language en`;选不准时再加 `--audio-stream 2` |
| 原片底部已有烧死字幕 | 上面那条加 `--hard-subtitle-layout top`,新字幕放顶部,两边各占一头 |
| 已有原文 `.srt`,不想重新转写 | 加 `--subtitle '/绝对路径/字幕.srt'` |
| 电影这类大文件,只要外挂中文字幕 | 见下面「.mkv 电影:只做外挂字幕」 |
| 已有 `.srt`,只要中文字幕文件 | `uv run video-txt translate '/绝对路径/字幕.srt' --provider lmstudio` |
| 只要文字稿,不翻译 | `uv run video-txt transcribe "$V"`,加 `--format srt` 出字幕 |
| 中文配音 | `uv run video-txt dub "$V" --provider lmstudio` |
| 对谈类,几个人配几个音色 | 上面那条加 `--diarize` |
| 保留原讲者的音色 | 再加 `--tts-engine f5-tts` |

同名 `.srt` 就在视频旁边时会自动复用,`--subtitle` 只在字幕不同名或不同目录时才需要。
底部已有字幕时优先用 `top` 而不是黑罩,原因见 [NOTES.md](NOTES.md#硬字幕顶部布局-vs-黑罩)。

## 字幕自动精修(v0.3)

Whisper 默认按解码窗口输出字幕,开头几句有时会跨二三十秒、一次铺满整屏。精修模式改用
Whisper 的词级时间戳重建字幕块:

```bash
V='/绝对路径/你的视频.mp4'

# 完整流水线
uv run --python 3.12 video-txt run "$V" --provider lmstudio --mux-mode hard \
  --refine-subtitles

# 只生成精修后的原文字幕
uv run --python 3.12 video-txt transcribe "$V" -f srt --refine-subtitles
```

它会优先在完整句末和超过 0.8 秒的停顿处断句,同时把每条字幕限制在约 6 秒、最多两行;
东亚全角字符按双倍显示宽度计算。OpenAI Whisper 和 MLX Whisper 两个后端都支持,默认不开启,
所以旧命令的输出完全不变。

一次成功转写会分别原子写入 `<视频名>.srt` 和 `<视频名>.words.json`,并以最后写入的词级文档作为
完成标记。流水线只有在两者都存在、且词级文档格式有效时才复用;缺失或损坏会自动重转,显式重做
仍用 `--retranscribe`。如果 `.srt` 在生成后被手工改过,校验不匹配时会停下而不会覆盖修改;
确认要丢弃手工修改再加 `--retranscribe`。厂商原始 JSON 只在临时目录中存在,成功或失败后都会清理。

精修必须从音视频重新取得词级时间戳,因此不能和 `--subtitle` 一起用;单独运行 `transcribe` 时必须
同时指定 `-f srt`。

## 已有字幕的审计与安全清理(v0.3.1)

`--refine-subtitles` 是“转写时用词级时间戳重新断句”;如果手里已经有 `.srt`,不想重新跑 Whisper,
用 `audit` / `clean`。先只检查:

```bash
S='/绝对路径/原文字幕.srt'
V='/绝对路径/对应视频.mkv'

uv run --python 3.12 video-txt audit "$S" --media "$V" --language en
```

终端会显示问题数,并在字幕旁写出 `<字幕名>.audit.json`。检查范围包括:接近 30 秒解码窗的短句幻觉、
零时长/倒序/重叠时间轴、空字幕、过长字幕、超过两行或 42 显示列、序号异常、长段重复、明显的语种
文字不符,以及字幕是否远早于视频结束或反过来超出视频结尾。发现任何项目时 `audit` 返回非零状态,方便放进脚本或流水线;
JSON 报告已经存在时要显式加 `--overwrite`,避免误盖人工留存的报告。

确认要生成一份安全清理版时:

```bash
uv run --python 3.12 video-txt clean "$S" -o '/绝对路径/原文字幕.clean.srt' \
  --media "$V" --language en
```

`clean` 只自动做可以确定的操作:删除整窗短句幻觉和空块、把零时长碎片并入紧邻的前一条、重新连续
编号。重叠、倒序、普通长句和版式告警不会猜着改,都留在 `<输出名>.audit.json` 供人工复核。源字幕永远
不会被覆盖,`-o` 必须是新路径;目标已存在时也会停下,只有明确传 `--overwrite` 才替换生成物。
先预览而不写任何文件可加 `--dry-run`。

## 多音轨智能选择(v0.4)

双语电影经常把默认音轨设成配音版,甚至把语言标签写错。现在 `transcribe`、`run` 和 `dub` 在真正
转写前会先读取所有音轨:优先匹配 `--language` 与音轨标题/语言标签,其次参考默认轨,并主动避开
评论音轨和无障碍解说。最推荐的用法仍然只需指定源语言:

```bash
V='/绝对路径/双语电影.mkv'

uv run --python 3.12 video-txt transcribe "$V" -f srt --language en
```

选择成功时会先显示依据,例如:

```text
Audio stream: 2 (Eng, spa, stereo) — title 'Eng' matches en
```

上例故意保留了真实的矛盾元数据:语言标签误写成 `spa`,但标题 `Eng` 与用户要求的英语一致,所以选择
stream 2。选中的轨道会先被提取成临时 16 kHz 单声道 WAV 再交给 Whisper;任务结束后临时音频自动
清理,源视频不变。

两条轨道同样可信时程序会停下并列出索引、标题、语言和声道布局。确认后用 FFmpeg 的全局 stream
index 明确覆盖:

```bash
uv run --python 3.12 video-txt transcribe "$V" -f srt --language en --audio-stream 2

# 完整字幕流水线和配音流水线同样支持
uv run --python 3.12 video-txt run "$V" --provider lmstudio --language en --audio-stream 2
uv run --python 3.12 video-txt dub "$V" --provider lmstudio --language en --audio-stream 2
```

如果流水线已经有同名原文字幕,它仍会保护并复用旧文件;要按新音轨重转,同时加 `--retranscribe`。
不确定选择结果时先加 `--dry-run`,它只显示音轨、提取命令和 Whisper 命令。

## 术语表起草(v0.8)

术语表要在开翻前写好，但读一整季字幕找专有名词是没人愿意开始的活。`draft-terms` 把这步机械化：

```bash
uv run --python 3.12 video-txt draft-terms '/绝对路径/剧集目录'/*.srt \
  -o '/绝对路径/剧名.terms.json' --source-language en
```

判断专有名词不靠停用词表，靠字幕自己的写法：**一个名字会出现在句子中间且仍然大写，而且几乎不会
被写成小写**；`Well` / `But` 只在句首大写，`hope` 满篇都是小写。英语无条件大写的那几个词
（`I`、`OK`）单列一份名单排除。

产物就是一份 terminology 文件，只是每条 `target` **留空**：

```json
{
  "source": "Jesse", "target": "", "match": "word", "count": 31,
  "examples": ["This way, you'll never lose me, Jesse.", "Hello, Jesse, hello."],
  "aliases": ["Jessie"],
  "draft_warning": "The subtitles also spell this Jessie. ..."
}
```

`target` 为空时 `load_terminology` 直接报错，所以**没填完的草稿不会被误当成术语表用**。填完
`target`、删掉不值得立目的条目，就是可用的 `--term-file`；`count` 和 `examples` 是多余字段，
加载时忽略，留着当依据也无妨。

相差一个字母的两个拼写会被认成同一个名字的异写，互相写进 `aliases`——实测某部片的字幕把同一角色
拼成 `Jesse`(31 次) 和 `Jessie`(26 次)。

**换季复用旧表**用 `--against`：旧表已覆盖的名字自动略去，只列新名字，并对和旧词条**共用一个词**的
新名字打警告（这正是 `Aaron Ryan` 被旧词条 `Ryan Madison` 吸附那类错误的来源）：

```bash
uv run --python 3.12 video-txt draft-terms '/新一季目录'/*.srt -o /tmp/new.json \
  --against '/绝对路径/剧名.terms.json'
```

`--min-count`（默认 3）控制一个名字要出现几次才提案，`--limit`（默认 60）封顶。
**它找不到只说过一两次的名字**，那不是漏，是频率扫描的边界——扫完仍要过一遍片里台词少但重要的角色。

## 项目术语表与翻译审计(v0.5)

电影人名、产品名和行业术语不要靠每一批模型临场决定。为项目建立一份 JSON 术语表:

```json
{
  "schema": "video-txt.terminology",
  "version": 1,
  "source_language": "en",
  "target_language": "zh-CN",
  "terms": [
    {
      "source": "Woody",
      "target": "胡迪",
      "match": "word",
      "aliases": ["伍迪", "乌迪"]
    },
    {
      "source": "Buzz Lightyear",
      "target": "巴斯光年",
      "match": "phrase"
    }
  ]
}
```

`word` 只匹配完整单词,例如 `Woody` 不会误中 `Woodyard`;`phrase` 匹配完整短语。
`aliases` 只放你明确批准替换的错误或旧译法。程序不会猜测近义词,也不会把普通词擅自改成人名。

翻译、完整成片和配音都使用同一个参数:

```bash
T='/绝对路径/project.terms.json'

uv run --python 3.12 video-txt translate source.srt --provider lmstudio --term-file "$T"
uv run --python 3.12 video-txt run "$V" --provider lmstudio --term-file "$T"
uv run --python 3.12 video-txt dub "$V" --provider lmstudio --term-file "$T"
```

术语表会同时用于三层保障:作为模型提示、精确规范化源词/`aliases`、翻译后审计。翻译完成会在译文
旁写出 `<译文名>.translation-audit.json`;字幕数量、序号、时间轴或批准译法有确定性错误时命令返回
非零,`run` / `dub` 会在封装或合成前停下。整句疑似未翻译和译文异常膨胀只记为 warning,不会因
启发式判断阻断正常结果;连续至少 3 个英文词残留也会给出 warning。已经存在的译文也会按当前术语表
复审;若不合格,修正术语表后加 `--retranslate` 重新生成。已有审计报告默认保留:`translate` 用
`--overwrite`,流水线用 `--retranslate`,独立审计用 `--overwrite` 才会明确替换它。

只审计已有译文、不调用模型也不改字幕:

```bash
uv run --python 3.12 video-txt audit-translation source.srt source.zh.srt \
  --term-file "$T"
```

术语表可省略,此时仍检查原文/译文数量、序号、时间轴、空译文、整句残留原文和异常长度。
报告已经存在时显式加 `--overwrite`;两份字幕始终只读。

## 人工校订层(v0.8)

术语表管的是「同一个词永远这么译」。管不了的是逐句的剧情语义:模型把 explore 译成了「探险」,
把一句反话译成了正话——这些只有人读一遍才能发现,而它们是整个项目里最贵的文本。

改进译文文件里是留不住的:译文是生成物,下一次 `--retranslate` 就没了。所以校订单独存一份 JSON,
**锚定在原文台词上,而不是字幕编号上**:

```json
{
  "schema": "video-txt.revisions",
  "version": 1,
  "source_language": "en",
  "target_language": "zh-CN",
  "revisions": [
    {
      "source": "You can take today to figure out how to",
      "at": "00:10:33,280",
      "text": "今天你可以先摸索",
      "note": "模型把 figure out 译成了「弄明白」"
    }
  ]
}
```

`source` 是原文那一行,大小写和空白不敏感。`at` 是它在原文字幕里的开始时间,只在同一句台词
全片说过多次时才必须填(不填而有歧义会报错并列出候选时间);填上也无妨,可以当书签用。
`text` 是定稿译文,`\n` 表示字幕内换行。`note` 只给人看。

为什么不用字幕编号:`clean` 删掉一条幻觉、`retranscribe-range` 拼回一段,下面所有编号都会平移。
按编号写死的修订会**静默地**写到别人的台词上。锚定原文则跟着台词走:

```bash
uv run --python 3.12 video-txt revise source.srt source.zh.srt \
  --revisions revisions.json
# → source.zh.revised.srt 和 source.zh.revised.revision-report.json
```

原文和译文都只读,输出必须是新路径。报告里逐条记录落在哪个 cue、改前改后、以及锚点时间漂移了多少。

流水线里加 `--revisions` 就会在翻译之后自动套回去,重翻多少次都不丢:

```bash
uv run --python 3.12 video-txt run "$V" --term-file "$T" --revisions revisions.json
uv run --python 3.12 video-txt dub "$V" --term-file "$T" --revisions revisions.json
uv run --python 3.12 video-txt mux "$V" source.srt --revisions revisions.json
```

三条硬规则:

- **锚点匹配不上就报错**,绝不静默跳过。一份悄悄失效的校订文件比没有更糟——命令照样报成功,
  而每一条校订都退回成了模型的说法。
- **两条修订抢同一句**会报错,不会后写的赢。
- **重复套用是安全的**。已经是定稿写法的行记为 `already_current`,报告里单独计数;某一行长期
  `already_current`,说明模型已经能自己译对,可以从文件里删掉了。

套完修订会**重新审计**并写出 `<译文>.revised.translation-audit.json`。这份描述的是真正出片的文件，
不像 `translate` 当时写的那份会随后续改动过期。人工校订过的行若命中 `glossary_target_missing` /
`glossary_alias` / `source_text_residue` / `source_text_unchanged` / `translation_unusually_long`
这五类启发式判断，会标成 `accepted`：照样列在报告里，但不计入 error/warning，也不再拦下流水线——
那一行已经有人读过并做了决定。结构性错误（块数、序号、时间轴、空译文）**永远不会**被标 accepted：
校订只替换一句的文字，造不成这些错。

只审计、不改文件时同样可以带上校订文件：

```bash
uv run --python 3.12 video-txt audit-translation source.srt source.zh.srt \
  --term-file "$T" --revisions revisions.json
```

长片和整季写进项目文件，`project run` 自动带上：

```bash
uv run --python 3.12 video-txt project init "$V" --revisions revisions.json --term-file "$T"
```

`revise` 在项目里是**独立阶段**，夹在 `translate` 和 `mux` / `tts` 之间。改校订文件只让它和成片过期，
**不会让 `translate` 过期**——修一行不会触发整片重译：

```
translate: current
revise: stale — revisions changed
mux: stale — upstream revise is stale
```

锚点时间漂移超过 5 秒会打印提醒但不阻断:重新识别本来就会让台词挪动一两秒,漂移到几分钟才值得
怀疑是匹配错了行。

## 配乐提取(v0.9)

片子的配乐是混在对白底下的，没有一条现成的轨可以拷。分离能把对白拿掉——`separate` 早就在做——
但剩下的是一条两小时、中间大段静音的伴奏，不是一组能播的曲子。找出曲子才是这一步的工作。

```bash
uv run --python 3.12 video-txt music '/绝对路径/电影.mkv'
# → 电影.music/ 里若干 .flac + 一份 .music.json
```

判据只用音频，因为**片源可能是网上看到的、旁边没有字幕**。ffmpeg 的 `ebur128` 每 100 毫秒报一次
瞬时响度（109 分钟的片子不到 2 秒算完），伴奏连续高于底噪的地方就是有音乐。

每段会报 `voice_share`——这段有多少比例被人声盖着。加 `--clean` 只保留**没人说话**的段落：
分离器在对白最响的地方留下的瑕疵最多，所以这些才是拿去用在别的视频里不露馅的。

```bash
video-txt music "$V" --clean --min-duration 15    # 只要干净的
video-txt music "$V" --stems                      # 顺带导出全部分轨
video-txt music "$V" --from 00:31:07 --to 00:34:20 --source mix   # 只切你指定的一段
```

**分轨**：`--stems` 用 `htdemucs`（drums/bass/other/vocals）或 `--model htdemucs_6s`
（再拆出 guitar/piano），各存一份 FLAC 并缓存。

**它不判断有没有人在唱。** 分离器听到的只是"人声"，唱和说一律进同一条轨；用"人声占比高"去猜歌曲
实测会把一段 71 秒的对话判成歌。所以没有任何一段会仅凭音频被标成歌曲。要切歌只有两条路：
`--from/--to` 自己给范围，或者 `--subtitle` 指一份**人工字幕**——里面用 `♪` 标了唱词
（Whisper 转写稿没有这个标记，官方字幕轨才有）。歌曲从**原始混音**里切，因为没有唱的歌不是歌；它占的那段会先从音频搜索里遮掉，
不会同一段音乐导出两次。

`♪` 那条路只认**有词的**唱词块——裸 `♪♪` 只表示"这里有音乐"，不是歌
（实测某季 128 个短标记块里 72 个没有词）。唱词块之间有间奏和换气，所以按 12 秒桥接合并成整首。

实测一集 34 分钟的剧集：找出 14 段，其中字幕标注的那首插曲被准确切在 32:45–33:59（73.9 秒，
-15.9 LUFS）。首次跑约 3 分钟（分离占大头），缓存命中后重跑 18.5 秒。

缓存和配音共用：某片如果已经跑过 `--separate-bgm`，提配乐是秒出。每段默认响度归一到 -16 LUFS
方便直接播放，两端各 50 毫秒淡入淡出防咔哒；要原始电平加 `--raw-levels`。

**格式**默认 `flac`——这是工作文件，还要再进别人的视频里编码一次，用有损格式等于叠加两代损失。
只是想听或者要放进手机就换 `--format m4a` / `mp3`（同一首插曲 6.5MB → 1.7MB），
要无损又要 Apple Music 能导入就用 `--format alac`。输出采样率跟随输入，不做上采样。

## 双语字幕(v0.8)

原文和译文一起上屏。学语言的人两行都读，校对译文的人不看原文根本没法判断一句译得对不对——
而这两件事都不该靠开两个文件来回滚动。

```bash
uv run --python 3.12 video-txt bilingual source.srt source.zh.srt
# → source.zh.bilingual.srt，默认译文在上、原文在下
```

`--order source-first` 把原文放上面。两份字幕都只读，输出必须是新路径。

流水线里加 `--bilingual`，翻译（以及校订）之后自动合并，合并结果就是被封装/烧录的那一份：

```bash
uv run --python 3.12 video-txt run "$V" --bilingual                     # 软字幕
uv run --python 3.12 video-txt run "$V" --bilingual --mux-mode hard     # 硬字幕
uv run --python 3.12 video-txt mux "$V" source.srt --bilingual
```

**烧硬字幕时原文会自动压小压淡**——字号 0.72 倍、半透明，读起来是第二行而不是和译文抢注意力。
软字幕交给播放器，两行等大。

两份字幕必须是同一份字幕：块数、序号、时间轴逐条一致，对不上直接报错而不是猜。
这正是 `translate` 保证、`audit-translation` 检查的东西，所以正常流程出来的文件天然满足。

**合并后每块恒定两行**：一行译文、一行原文。各自语言原有的换行会被压平——原文两行加译文两行
会摞成四行、吃掉半个画面，而那两个换行本来只是各自语言对行宽的决定，并排放到一起就不成立了。
个别很长的行在渲染时仍会被播放器或 ASS 自动折到第三行（实测 1683 块里有 34 块），
这和单语字幕的行为一致。

一个提醒：双语字幕每块两行、行也更宽，拿 `video-txt audit` 去查会报 `too_many_lines` 和
`line_too_wide`。那是对的——双语字幕本来就比单语密。要审计就审合并之前的那一份。

## .mkv 电影:只做外挂字幕

4K 电影动辄十几 GB,烧硬字幕要把整部片子重新编码,画质有损还要等一个多小时。外挂字幕零成本:
出一份 `.zh.srt` 放在视频旁边,播放器加载,原始画质一点不动。两条命令,不碰视频文件:

```bash
V='/Users/sun/Downloads/电影.mkv'

# 1. 转写。电影片头常是音乐,一定要指定源语言,否则语种会认错
#    ja 是日语,换成 en / ko / fr 等即可,取值表见下面「源语言不是英文」
uv run video-txt transcribe "$V" -f srt --language ja \
  --whisper-arg=--condition_on_previous_text --whisper-arg=False

# 2. 翻译上一步生成的 .srt
uv run video-txt translate "${V%.mkv}.srt" --provider lmstudio
```

第 1 步结束会自动体检转写结果,有问题会报出来并以非零退出,所以两步之间用 `&&` 串起来也是安全的:
转写不干净就不会接着白翻一遍,按提示重转即可。

跑完视频旁边多两个文件:`电影.srt`(日文原文)和 `电影.zh.srt`(中文)。用 IINA 或 VLC 打开电影,
把 `.zh.srt` 拖进播放窗口就行。两份字幕同前缀,自动加载的播放器可能先挂上日文那份,
在字幕菜单里切一下;不想被干扰就把日文那份挪走。

关于第 1 步那两个参数:

- `--language ja` 不能省。Whisper 只拿前 30 秒判断语种,片头音乐会让它猜成英语,
  然后整片跟着跑偏,实测教训见 [NOTES.md](NOTES.md#一次真实的语种误判)。
- `--whisper-arg=--condition_on_previous_text --whisper-arg=False` 关掉"用前文续写",
  电影里长段音乐和静默容易让模型卡进重复循环,一句话每 30 秒重复到片尾。
  值必须用等号连接,否则会被 argparse 当成新参数。

参考耗时,一部 109 分钟的 4K 日语电影、CPU 转写:第 1 步 35 分钟,第 2 步走云端 8 分钟
(1849 条字幕分 7 批并发),本机模型要久得多。第 1 步装 `--extra mlx` 走 GPU 能快数倍。

**要一个自带字幕的单文件**发给别人时,mkv 还有个便宜做法:软字幕封装是直接流拷贝,
不重新编码、画质无损,只是要多占一份视频的空间。

```bash
uv run video-txt run "$V" --provider lmstudio --mux-mode soft --subtitle "${V%.mkv}.srt"
# → 电影.zh-subbed.mkv,里面一条可开关的中文字幕轨
```

硬字幕 `--mux-mode hard` 只在对方播放器连字幕轨都不认时才值得,代价是整部重编码。

## 重跑规则

每个阶段的产物都落盘,重跑时**已经存在的阶段自动跳过**。所以调字幕样式直接重跑就行,
不会重新转写和翻译——上面那个 24 分钟的视频,改样式重跑只花 40 秒。

| 想强制重做 | 加这个参数 |
| --- | --- |
| 重新转写 / 翻译 / 分离说话人 | `--retranscribe` / `--retranslate` / `--rediarize` |
| 覆盖已存在的输出视频 | `--overwrite-video` |
| 换个输出文件名(保留旧版对比) | `-o '/绝对路径/输出.mp4'` |

翻译中途 Ctrl-C 或报错,已完成的批次记在 `<字幕名>.partial.jsonl` 里,重跑自动续上,
全部成功后该文件自动删除。

## 中文配音

```bash
uv sync --extra dub                            # 只做一次,装的是 edge-tts,合成时要联网
uv run video-txt dub "$V" --provider lmstudio
```

和字幕流程一样先转写、再翻译,然后合成中文语音、按字幕时间轴对齐、混进视频,
输出 `你的视频.zh-dubbed.mp4`。这个视频之前做过字幕的话,`.srt` 和 `.zh.srt` 直接复用——
但注意 `dub` 的翻译是按**口语**要求的(念得顺,不是字幕体),想把老的字幕翻译换成口语版,
加 `--retranslate` 重翻一次。

两个默认值先记住:**默认整轨替换原声**(保留背景声的两种方式见下面「常用调整」),
默认音色是 `zh-CN-XiaoxiaoNeural` 女声。

跑完的报告看这四行:

| 报告行 | 含义 | 怎么算正常 |
| --- | --- | --- |
| `Spoken again at a faster rate to fit` | 塞不下的句子,自动用更快的语速重说了一遍 | 自动发生,占比不高就没问题 |
| `Stretched to fit` | 重说后仍差一点,靠拉伸波形补齐的句子 | 越少越好,有几句也听不太出来 |
| `Still longer than their subtitle slot` | 拉伸到上限仍然超时的句子 | 最好是 0 |
| `Largest timeline drift` | 整条时间轴最大偏移 | 2 秒以内基本无感,超过会自动提示 |

中文念出来往往比英文长,塞不进原来的时间格子就得加速。超时的句子会**自动换更快的 TTS
语速重新合成**——真人式的快语速,不是机械感来源的波形拉伸;拉伸只兜重说仍不够的残余。
所以一般不用再为漂移手动调参,`--rate +10%` 只在整体都偏慢时才需要。

### 常用调整

| 想要 | 加这个参数 |
| --- | --- |
| 只留音乐和环境声(推荐) | `--separate-bgm`,先 `uv sync --extra separate`。Demucs 把原声里的人声抠掉,音乐、掌声、环境声全音量保留,不跟配音打架。首次要下模型、分离要几分钟,结果缓存后重跑秒过。抠出来的人声干轨会留下,克隆音色时用它剪参考片段 |
| 保留完整原声垫底 | `--keep-bgm`,整条原声(含原人声)压到 15%,太轻用 `--bgm-volume 0.25` 调 |
| 换音色 | `--voice zh-CN-YunxiNeural` |
| 整体语速 | `--rate +10%` 或 `--rate -10%` |
| 漂移大了想压住 | 超时句子已自动重说,一般不用管;真要压再 `--max-atempo 1.6`,默认 1.35 |
| 让模型把读不完的句子改短 | `--fit-duration`,会改动译文,最后手段 |
| 顺便挂一条可开关的中文字幕轨 | `--soft-subtitle` |
| 留下单独的人声音轨 | `--keep-dub-audio`,默认混流后就删 |
| 不联网 | `--tts-engine say`,macOS 自带,音色机械,兜底用 |

常用中文音色,完整列表跑 `uv run edge-tts --list-voices | grep zh-CN`:

| `--voice` 取值 | 性别 | 特点 |
| --- | --- | --- |
| `zh-CN-XiaoxiaoNeural` | 女 | 默认,温暖,口播和新闻都合适 |
| `zh-CN-XiaoyiNeural` | 女 | 活泼 |
| `zh-CN-YunyangNeural` | 男 | 专业稳重,新闻腔 |
| `zh-CN-YunxiNeural` | 男 | 明朗年轻 |
| `zh-CN-YunjianNeural` | 男 | 有力,解说腔 |

### 多说话人:一人一个音色

对谈、播客用一个音色配下来分不清谁在说话。`--diarize` 先算出「谁在什么时候说话」再按人分配音色:
说得最多的那位用 `--voice`,其余的从内置音色池里依次取,男女交替。

```bash
uv sync --extra diarize
export HF_TOKEN=hf_xxx        # 也可以放进凭据文件,见文末「凭据约定」
uv run video-txt dub "$V" --provider lmstudio --diarize
```

- **首次要过 Hugging Face 门禁,三个仓库都要点同意。** 模型是 gated 的(gated 不等于收费,全程免费),
  账号要逐个接受条款:
  [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)、
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1),
  以及 pyannote 4.x 会把 3.1 内部重定向到的
  [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)。
  少接受一个就 load 不动。token 没读到报 **401**,读到了但没接受条款报 **403**
  ("not in the authorized list")——看到 403 是去点同意,不是换 token。
- **配乐大的片子,别信声学分割的结果。** pyannote 分的是声音像不像,不是谁扮演什么角色。实测一段
  2 分 43 秒、配乐铺满全片的人机演示合集:`--speakers 2` 把几个不同的真人和助手的应答混进同一个
  标签,放开自动数得到 4 个说话人、仍然把一整句应答从中间劈成两个标签。这种片子按**台词内容**
  手改 `你的视频.speakers.json` 更可靠,见下面「手工校正说话人」。
- **知道有几个人就直接说。** `--speakers 2` 比让它自己猜稳,只知道范围用 `--min-speakers 2
  --max-speakers 4`。猜多了会把同一个人拆成两个音色,很明显。
- **指定某人的音色:**`--speaker-voice SPEAKER_01=zh-CN-YunxiNeural`,可重复。
  `1=...` 是 `SPEAKER_01=...` 的简写,编号是 pyannote 给的标签、从 0 开始数,
  跟报告里按说话时长排的先后没有关系。名字写错会直接报错,不会静默忽略。
- 结果落盘在 `你的视频.speakers.json`,重跑直接复用,想重算加 `--rediarize`。

**手工校正说话人。** `speakers.json` 里的 `turns` 是可读的 JSON,可以直接改:

```json
{"start": 18.391, "end": 18.712, "speaker": "ASSISTANT"}
```

- 标签随便起名(`HUMAN`、`ASSISTANT`、角色名都行),`--speaker-voice` 和 `--clone-reference`
  按这个名字对应。
- 最省事的写法是**一块字幕一条 turn**,直接用字幕自己的时间范围:每块归属明确,
  跨角色的块按重叠时长归给占比大的那一方。
- **`version`、`model`、`cache_key` 一个字都别动。** 这三样对不上,下次运行会当缓存失效重新跑
  pyannote,把手改覆盖掉;重跑时命令行上的 `--speakers` 等参数也要和 `cache_key` 里记的一致。
- **字幕块数变了要重建。** 做过 `retranscribe-range` 之后块边界全变了,旧 turn 的时间对不上新块,
  角色会整段错位——按新的字幕时间轴重写一遍。

### 原声克隆:保留原讲者的声音

克隆引擎从原视频里剪一小段说话人的声音当参考,用他自己的音色念中文。
配上 `--diarize` 就是每个人克隆自己——剧集配音里每个角色保持自己的声线。

推荐 **IndexTTS-2**:情感表现力是三个克隆引擎里最强的,而且超时句子的加速走的是模型自己的
时长控制(说得快,不是事后压缩),几乎听不出赶。它锁死自己的 Python 和 torch 版本,
所以装在自己的目录里,不进本项目的虚拟环境:

```bash
git clone https://github.com/index-tts/index-tts.git ~/tools/index-tts
cd ~/tools/index-tts && uv sync                # 独立环境,自带 Python 3.11
uvx --from huggingface-hub hf download IndexTeam/IndexTTS-2.5 --local-dir checkpoints

uv run video-txt dub "$V" --provider lmstudio --tts-engine index-tts \
  --clone-repo ~/tools/index-tts
```

`--clone-repo` 指向那个目录就够了:里面的 `.venv` 和 `checkpoints` 都会自动找到。

- 参考音频自动从源语言字幕里剪(3–12 秒连续说话),存进 `你的视频.dub-cache/reference/`。
  想自己指定用 `--clone-reference my.wav`,旁边放一个 `my.txt` 写清楚这段音频说了什么;
  配 `--diarize` 时按角色指定:`--clone-reference ASSISTANT=assistant.wav`,可重复。
- **多说话人时,开跑前先看一眼参考片段的文字。** 参考窗口是按"同一说话人的连续块"找的,
  说话人分错了参考里就同时有两个人的声音——实测两段参考各自都夹着对方一句,等于拿混合体克隆了
  两次,两个音色听起来差不多、而且都偏慢。`cat 你的视频.dub-cache/reference/*.txt` 一眼就能看出
  混没混:
  ```
  SPEAKER_01-clean-53000.txt: Make it fun. Right, I can help with that. Can you go to eBay...
                              ^^^ 提问方              ^^^ 应答方,混了
  ```
  混了就自己切一段干净的传进去。开了 `--separate-bgm` 时从 `你的视频.dub-cache/bgm/vocals.flac`
  这条人声干轨上切,参考里不会带配乐:
  ```bash
  ffmpeg -ss 29.66 -t 2.75 -i 你的视频.dub-cache/bgm/vocals.flac \
    -ac 1 -ar 24000 -c:a pcm_s16le assistant.wav      # 同名 .txt 写清这段说了什么
  ```
  换成纯净参考的收益,同一条命令前后对比实测:语速 7.3→8.0 字/秒,需压缩改写的句子 20→9,
  最紧一句 2.63×→1.65×,超出槽位 1 句→0,时间轴漂移 0.4→0.0 秒。
- **配乐大的片子加上 `--separate-bgm`**:分离出来的人声干轨会用来剪参考片段,克隆到的是人声
  本身,不是人声加背景音乐。文件名带 `-clean`,和从混音剪的参考分开存。没开这个参数(或缓存里
  还没有干轨)就照旧从原始混音剪。
- **慢,而且没法并发。** 本地推理,一集剧大约是实时的三分之一到一半;
  先 `--dry-run` 看清楚要合成多少段再开跑,别拿长视频试手。
- 另外两个引擎:`--tts-engine f5-tts` 最省事(`uv sync --extra clone` 装进本项目环境,
  权重自动下载);CosyVoice 跟 IndexTTS 一样要自己 clone,
  再加 `--clone-model <模型目录>`,依赖冲突时用 `--clone-python` 指定解释器。
- 克隆的是真人的声音,用在原讲者没授权的地方之前先想清楚。

### 缓存

每句语音按内容哈希缓存在 `你的视频.dub-cache/`,重跑只补缺的句子。改 `--keep-bgm`、`--soft-subtitle`
这类不影响语音本身的参数几乎不花时间;换 `--voice` 或 `--rate` 会让缓存整体失效,等于重新合成一遍。
`--separate-bgm` 分离出来的两条轨也在这里(`bgm/no_vocals.flac` 背景、`bgm/vocals.flac` 人声干轨,
后者用来剪克隆参考),`--prune-cache` 不会动它们。

哈希只认内容、音色和语速,不认句子的位置:手改译文里的某一句、或者在中间插一句,后面的句子照样
命中缓存;整片重复的台词(片头片尾语这类)也只合成一次。

失效的旧语音不会自动删,方便你换回原音色时复用。反复调参会越堆越多,跑完如果闲置的比用到的还多
会提示一句,确定不回头了就加 `--prune-cache`,它只删这一次没用到的语音片段。目录结构见
[NOTES.md](NOTES.md#缓存结构与体积)。

## 其他常用参数

完整参数跑 `uv run video-txt run --help` / `dub --help`,下面只列 help 里讲不清楚的。

### 源语言不是英文

源语言不用配置。Whisper 自己检测,翻译请求里源语言默认写的是 auto-detect 由模型判断,
日语视频跟英语视频跑的是同一条命令。两个值得手动指定的场合:

- **开头是音乐、静音或夹着英文。** Whisper 只拿前 30 秒判断语种,猜错就整片跑偏,`--language ja` 钉死。
  片头有音乐的电影几乎必踩,实测见 [NOTES.md](NOTES.md#一次真实的语种误判)。
- **专有名词转写不准。** `--initial-prompt` 用**源语言**写术语表,比如 `'AI エージェント、MCP'`。

`--language` 收的是语种代码:`ja` 日语、`en` 英语、`ko` 韩语、`zh` 中文、`yue` 粤语、`fr` 法语、
`de` 德语、`es` 西班牙语、`it` 意大利语、`pt` 葡萄牙语、`ru` 俄语、`th` 泰语、`vi` 越南语、
`ar` 阿拉伯语、`hi` 印地语。英文名如 `Japanese` 也认,中文名不认——这两种是 Whisper 自己的规矩。
它一共支持 100 个语种,完整清单在 `uv run python -m whisper --help` 的 `--language` 那一行。

`--source-language 日语` 只是给翻译模型的提示,不影响转写。别把 `--mode translate` 当成翻译开关:
那是 Whisper 自带的 X→英文。译成中文以外的语言用 `--target-language`,细节见
[NOTES.md](NOTES.md#语言处理)。

### 转写

| 参数 | 说明 |
| --- | --- |
| `--initial-prompt 'Claude Code, MCP, Anthropic'` | 术语提示,显著改善专有名词的正确率 |
| `--backend mlx-whisper` | Apple Silicon 走 GPU,需先 `uv sync --extra mlx` |
| `--model medium` | 换模型,默认 turbo |

拿到字幕后会自动体检三项:同一句话连续重复超过一分钟(Whisper 听不到人声时空转的特征),
指定了 `--language ja` 却几乎没有日文字符,以及字幕只覆盖到视频前一半就没了——最后这种
基本是转写时文件还没下载完(种子按块乱序落盘,文件看着是全尺寸,其实只有开头能解码)。
`run` 和 `dub` 一旦发现就停在这里,不往下浪费时间和钱;单独跑 `transcribe` 字幕照样写出来,
但退出码是 1——用 `&&` 串起来的下一条命令不会跑,想看看问题再决定就分两次跑。
复用已有 `.srt` 时同样会查。确认没问题,加 `--skip-transcript-check` 跳过。

### 硬字幕外观

字号和边距的单位是**视频原始分辨率下的像素**,不传时按分辨率自动换算(1080p ≈ 49px / 54px)。

```bash
uv run video-txt run "$V" --provider lmstudio --mux-mode hard \
  --hard-subtitle-font 'PingFang SC' --hard-subtitle-font-size 44 --hard-subtitle-margin-v 60
```

- 画质用 `--crf`(越小越清晰,默认 20)和 `--preset`(默认 medium)控制。
- 硬字幕需要带 `subtitles` 滤镜的 ffmpeg。提示缺失就 `brew install ffmpeg-full`,再
  `export FFMPEG_PATH=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg`。
- ffmpeg 打印 `Error opening font: ...PingFangUI.ttc` 可以忽略,字体照常生效,原因见 NOTES。
- 黑罩布局 `bottom-box` 的实测经验也在 NOTES 里,一般优先用 `top`。

### 翻译

| 参数 | 说明 |
| --- | --- |
| `--provider lmstudio` | 默认推荐。指向本机 `http://localhost:1234/v1`,模型问 LM Studio 要,不花钱也不用 key |
| `--provider deepseek` | 云端。一次性设好接口地址、key 变量名和默认模型 `deepseek-v4-flash` |
| `--model` / `--base-url` / `--api-key-env` | 换别的服务时逐项覆盖 |
| `--concurrency 4` | 并发批数,长视频提速明显;被限流或本地服务器扛不住就调小 |
| `--preserve-term Kubernetes` | 追加保留不译的术语,可传多次;默认已含 Claude、MCP、OpenAI、token 等 AI 术语,长期增删改 `video_txt/constants.py` 的 `DEFAULT_TERMS` |
| `--term-file project.terms.json` | 指定源词→目标译法;用于模型提示、精确规范化和翻译审计 |
| `--note '保持轻松的教程口吻'` | 追加翻译要求 |

### 本机还是云端

文档里的命令一律用 `--provider lmstudio`,因为不花钱。要哪个换哪个,别的参数都不用动。

| | `--provider lmstudio` | `--provider deepseek` |
| --- | --- | --- |
| 花钱 | 不花 | 按 token 计费 |
| key | 不用 | 要,见下面「凭据约定」 |
| 速度 | 慢,取决于本机和模型大小 | 快,24 分钟的视频约 30 秒译完 |
| 联网 | 不用 | 要 |
| 长片 | 先拿一小段试译看质量,再整片跑 | 直接跑 |

**本机这边只有一步准备:** 在 LM Studio 里加载一个模型、打开它的本地服务器。不用给 key,
也不用 `--model`——工具会问 LM Studio 当前加载的是哪个模型,并打印 `Local model: ...`。
想钉死某个模型就照常加 `--model`;LM Studio 装在别的机器上就加 `--base-url`。

`lmstudio` 预设还替本地模型改了两个默认值,都是实测调出来的
(数据见 [NOTES.md](NOTES.md#本地推理模型必须关掉思考)):

- **不让它思考**(`reasoning_effort: none`)。现在的本地模型多半是推理模型,让它想的话翻两句字幕
  能烧掉四千多个 token、等一分钟,而译文跟不想时没差别。真想让它思考就 `--reasoning-effort high`。
- **每批 800 字**而不是云端的 3200。一次问它七十条,它答六十条就收尾了。批次小了反而更快。

还是嫌慢或一批跑不完 180 秒,用 `--timeout 600` 放宽、`--concurrency` 调并发。

开头会打印几行 `API rejected response_format; falling back to plain JSON prompting.`——
LM Studio 不认 `json_object`,工具自动改用纯提示词要 JSON。同时在飞的请求各撞一次,
所以默认并发下是 4 行,之后不再出现,可以忽略。

## 凭据约定

只在用云端 provider 时才需要看这节,`--provider lmstudio` 全程不碰 key。

- 任何真实 key 都不写进 README、脚本、提交信息或示例命令,文档里只出现占位符。
- key 只放两个地方:shell 环境变量,或仓库之外的本地凭据文件。两者都用同样的写法:

```bash
export DEEPSEEK_API_KEY='你的 key'
```

- 环境变量没设时,工具会自动读取默认位置的凭据文件;该文件必须设为 600,否则工具会拒绝读取。
  路径不记在文档里,需要换位置用 `--secrets-file` 指定。
- 换服务用 `--api-key-env` 指定变量名,不要把 key 直接贴进命令行(会留在 shell 历史里)。
- 万一 key 真的进了文档或提交历史,去服务商控制台吊销重发是唯一有效的补救,改文件不够。

## 兼容旧命令

原来的三个脚本仍然可用,参数完全不变,内部转调新 CLI:

```bash
uv run --python 3.12 python transcribe_media.py '/绝对路径/视频.mp4'
uv run --python 3.12 python translate_srt.py '/绝对路径/字幕.srt' --provider lmstudio
uv run --python 3.12 python translate_and_mux_video.py '/绝对路径/视频.mp4' '/绝对路径/字幕.srt' \
  --mux-mode hard
```

## 开发

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check . && uv run ruff format .
```

代码分层:`video_txt/subtitles.py`(SRT/ASS 解析与生成)、`refine.py`(词级时间戳归一化与字幕精修)、
`media.py`(ffmpeg/ffprobe 探测)、
`translate.py`(翻译引擎)、`terminology.py`(项目术语表、规范化与翻译审计)、
`quality.py`(结构化字幕审计与安全修复)、`fit.py`(时长感知重译)、
`diarize.py`(说话人分离)、
`clone.py`(参考音频抽取与克隆编排)、`clone_worker.py`(在模型自己的环境里合成的独立脚本)、
`voices.py`(音色分配)、`timeline.py`(语音片段对齐与整轨渲染)、`dub.py`(配音编排)、
`transcribe.py`、`retranscribe.py`(局部音频提取、重识别与 SRT 拼接)、`mux.py`、
`pipeline.py`(阶段编排)、`project.py`(项目配置、内容指纹与阶段失效)、
`arguments.py`(命令行参数声明)、
`cli.py`(参数校验与命令处理)、`env.py`(凭据读取)、`parallel.py`(线程池)。

Python 只用 `uv` 管:不动系统自带 Python,不用 `sudo pip install`,不引入 pyenv / conda / poetry。

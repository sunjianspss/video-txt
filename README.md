# video-txt

把本地视频变成**中文字幕视频**或**中文配音视频**,一条命令跑完转写、翻译、封装。

做技术决定时的实测数据和取舍记在 [NOTES.md](NOTES.md),这里只讲怎么用。

## 快速开始

第一次用先装依赖,只做一次:

```bash
brew install ffmpeg
uv sync
```

翻译要用 key(见文末「凭据约定」)。这条命令只检查、不打印 key:

```bash
uv run python -c "from video_txt.env import resolve_api_key; resolve_api_key('DEEPSEEK_API_KEY'); print('key 就绪')"
```

然后跑:

```bash
cd /Users/sun/Documents/docx/video-txt

uv run video-txt run '/绝对路径/你的视频.mp4' --provider deepseek --mux-mode hard
```

跑完在**视频所在目录**得到三个文件:

| 文件 | 是什么 |
| --- | --- |
| `你的视频.srt` | Whisper 转写的原文字幕 |
| `你的视频.zh.srt` | 翻译后的中文字幕 |
| `你的视频.zh-burned.mp4` | 中文字幕烧进画面的成片 |

去掉 `--mux-mode hard` 就是软字幕版,输出 `你的视频.zh-subbed.mp4`,字幕可以在播放器里开关。
想先看看会执行什么、不真跑,加 `--dry-run`。

参考耗时:24 分钟的 480p 视频,CPU 转写 7.5 分钟 + 翻译 30 秒 + 烧字幕 42 秒,合计约 8.5 分钟。

可选功能按需装,首次运行还要下模型权重:

```bash
uv sync --extra dub       # 中文配音(edge-tts)
uv sync --extra mlx       # Apple Silicon GPU 转写,比 CPU 快数倍,另下 1.5G 权重
uv sync --extra diarize   # 多说话人分离(pyannote)
uv sync --extra clone     # 原声克隆(F5-TTS)
```

## 按场景选命令

下面都以 `V='/绝对路径/你的视频.mp4'` 为例,先 `V=...` 设一下会少打很多字。

| 想做什么 | 命令 |
| --- | --- |
| 中文硬字幕成片(最常用) | `uv run video-txt run "$V" --provider deepseek --mux-mode hard` |
| 原片底部已有烧死字幕 | 上面那条加 `--hard-subtitle-layout top`,新字幕放顶部,两边各占一头 |
| 已有原文 `.srt`,不想重新转写 | 加 `--subtitle '/绝对路径/字幕.srt'` |
| 电影这类大文件,只要外挂中文字幕 | 见下面「.mkv 电影:只做外挂字幕」 |
| 已有 `.srt`,只要中文字幕文件 | `uv run video-txt translate '/绝对路径/字幕.srt' --provider deepseek` |
| 只要文字稿,不翻译 | `uv run video-txt transcribe "$V"`,加 `--format srt` 出字幕 |
| 中文配音 | `uv run video-txt dub "$V" --provider deepseek` |
| 对谈类,几个人配几个音色 | 上面那条加 `--diarize` |
| 保留原讲者的音色 | 再加 `--tts-engine f5-tts` |

同名 `.srt` 就在视频旁边时会自动复用,`--subtitle` 只在字幕不同名或不同目录时才需要。
底部已有字幕时优先用 `top` 而不是黑罩,原因见 [NOTES.md](NOTES.md#硬字幕顶部布局-vs-黑罩)。

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
uv run video-txt translate "${V%.mkv}.srt" --provider deepseek
```

第 1 步结束会自动体检转写结果,有问题会报出来并以非零退出,所以两步之间用 `&&` 串起来也是安全的:
转写不干净就不会接着花钱翻译,按提示重转即可。

跑完视频旁边多两个文件:`电影.srt`(日文原文)和 `电影.zh.srt`(中文)。用 IINA 或 VLC 打开电影,
把 `.zh.srt` 拖进播放窗口就行。两份字幕同前缀,自动加载的播放器可能先挂上日文那份,
在字幕菜单里切一下;不想被干扰就把日文那份挪走。

关于第 1 步那两个参数:

- `--language ja` 不能省。Whisper 只拿前 30 秒判断语种,片头音乐会让它猜成英语,
  然后整片跟着跑偏,实测教训见 [NOTES.md](NOTES.md#一次真实的语种误判)。
- `--whisper-arg=--condition_on_previous_text --whisper-arg=False` 关掉"用前文续写",
  电影里长段音乐和静默容易让模型卡进重复循环,一句话每 30 秒重复到片尾。
  值必须用等号连接,否则会被 argparse 当成新参数。

参考耗时,一部 109 分钟的 4K 日语电影、CPU 转写:第 1 步 35 分钟,第 2 步 8 分钟
(1849 条字幕分 7 批并发)。第 1 步装 `--extra mlx` 走 GPU 能快数倍。

**要一个自带字幕的单文件**发给别人时,mkv 还有个便宜做法:软字幕封装是直接流拷贝,
不重新编码、画质无损,只是要多占一份视频的空间。

```bash
uv run video-txt run "$V" --provider deepseek --mux-mode soft --subtitle "${V%.mkv}.srt"
# → 电影.zh-subbed.mkv,里面一条可开关的中文字幕轨
```

硬字幕 `--mux-mode hard` 只在对方播放器连字幕轨都不认时才值得,代价是整部重编码。

## 重跑规则

每个阶段的产物都落盘,重跑时**已经存在的阶段自动跳过**。所以调字幕样式直接重跑就行,不会重新
花钱转写和翻译——上面那个 24 分钟的视频,改样式重跑只花 40 秒。

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
uv run video-txt dub "$V" --provider deepseek
```

和字幕流程一样先转写、再翻译,然后合成中文语音、按字幕时间轴对齐、混进视频,
输出 `你的视频.zh-dubbed.mp4`。这个视频之前做过字幕的话,`.srt` 和 `.zh.srt` 直接复用。

两个默认值先记住:**默认整轨替换原声**(想保留原声当背景音加 `--keep-bgm`),
默认音色是 `zh-CN-XiaoxiaoNeural` 女声。

跑完的报告看这三行:

| 报告行 | 含义 | 怎么算正常 |
| --- | --- | --- |
| `Speed-adjusted to fit` | 有多少句被加速过 | 占比不高就没问题 |
| `Still longer than their subtitle slot` | 加速到上限仍然超时的句子 | 最好是 0,有几句也听不太出来 |
| `Largest timeline drift` | 整条时间轴最大偏移 | 2 秒以内基本无感,超过会自动提示 |

中文念出来往往比英文长,塞不进原来的时间格子就得加速。漂移大了**优先加 `--rate +10%`,
别急着调 `--max-atempo`**:前者是让 TTS 一开始就说得快一点,后者是事后把已经录好的波形拉伸,
机械感主要来自后者。

### 常用调整

| 想要 | 加这个参数 |
| --- | --- |
| 保留原声垫底 | `--keep-bgm`,原声压到 15%,太轻用 `--bgm-volume 0.25` 调 |
| 换音色 | `--voice zh-CN-YunxiNeural` |
| 整体语速 | `--rate +10%` 或 `--rate -10%` |
| 漂移大了想压住 | 先 `--rate +10%`;仍不够再 `--max-atempo 1.6`,默认 1.35 |
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
uv run video-txt dub "$V" --provider deepseek --diarize
```

- **首次要过 Hugging Face 门禁。** 模型是 gated 的,得先去
  [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 和
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  点同意再拿 token。
- **知道有几个人就直接说。** `--speakers 2` 比让它自己猜稳,只知道范围用 `--min-speakers 2
  --max-speakers 4`。猜多了会把同一个人拆成两个音色,很明显。
- **指定某人的音色:**`--speaker-voice SPEAKER_01=zh-CN-YunxiNeural`,可重复。
  `1=...` 是 `SPEAKER_01=...` 的简写,编号是 pyannote 给的标签、从 0 开始数,
  跟报告里按说话时长排的先后没有关系。名字写错会直接报错,不会静默忽略。
- 结果落盘在 `你的视频.speakers.json`,重跑直接复用,想重算加 `--rediarize`。

### 原声克隆:保留原讲者的声音

`--tts-engine f5-tts` 从原视频里剪一小段这个人的声音当参考,用他自己的音色念中文。
配上 `--diarize` 就是每个人克隆自己。

```bash
uv sync --extra clone
uv run video-txt dub "$V" --provider deepseek --diarize --tts-engine f5-tts
```

- 参考音频自动从源语言字幕里剪(3–12 秒连续说话),存进 `你的视频.dub-cache/reference/`。
  想自己指定用 `--clone-reference my.wav`,旁边放一个 `my.txt` 写清楚这段音频说了什么。
- **慢,而且没法并发。** 先 `--dry-run` 看清楚要合成多少段再开跑,别拿长视频试手。
- CosyVoice 没上 PyPI,要自己装:`--tts-engine cosyvoice --clone-model <模型目录>
  --clone-repo <源码目录>`,依赖冲突时用 `--clone-python` 指定它自己虚拟环境的解释器。
- 克隆的是真人的声音,用在原讲者没授权的地方之前先想清楚。

### 缓存

每句语音按内容哈希缓存在 `你的视频.dub-cache/`,重跑只补缺的句子。改 `--keep-bgm`、`--soft-subtitle`
这类不影响语音本身的参数几乎不花时间;换 `--voice` 或 `--rate` 会让缓存整体失效,等于重新合成一遍。

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
`run` 和 `dub` 一旦发现就停在这里,不往下花钱和时间;单独跑 `transcribe` 字幕照样写出来,
但退出码是 1——用 `&&` 串起来的下一条命令不会跑,想看看问题再决定就分两次跑。
复用已有 `.srt` 时同样会查。确认没问题,加 `--skip-transcript-check` 跳过。

### 硬字幕外观

字号和边距的单位是**视频原始分辨率下的像素**,不传时按分辨率自动换算(1080p ≈ 49px / 54px)。

```bash
uv run video-txt run "$V" --provider deepseek --mux-mode hard \
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
| `--provider deepseek` | 一次性设好接口地址、key 变量名和默认模型 `deepseek-v4-flash` |
| `--model` / `--base-url` / `--api-key-env` | 换别的服务时逐项覆盖 |
| `--concurrency 4` | 并发批数,长视频提速明显;被限流就调小 |
| `--preserve-term Kubernetes` | 追加保留不译的术语,可传多次;默认已含 Claude、MCP、OpenAI、token 等 AI 术语,长期增删改 `video_txt/constants.py` 的 `DEFAULT_TERMS` |
| `--note '保持轻松的教程口吻'` | 追加翻译要求 |

## 凭据约定

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
uv run --python 3.12 python translate_srt.py '/绝对路径/字幕.srt' --provider deepseek
uv run --python 3.12 python translate_and_mux_video.py '/绝对路径/视频.mp4' '/绝对路径/字幕.srt' \
  --mux-mode hard
```

## 开发

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check . && uv run ruff format .
```

代码分层:`video_txt/subtitles.py`(SRT/ASS 解析与生成)、`media.py`(ffmpeg/ffprobe 探测)、
`translate.py`(翻译引擎)、`quality.py`(字幕体检)、`fit.py`(时长感知重译)、`diarize.py`(说话人分离)、
`clone.py`(参考音频抽取与克隆编排)、`clone_worker.py`(在模型自己的环境里合成的独立脚本)、
`voices.py`(音色分配)、`timeline.py`(语音片段对齐与整轨渲染)、`dub.py`(配音编排)、
`transcribe.py`、`mux.py`、`pipeline.py`(阶段编排)、`arguments.py`(命令行参数声明)、
`cli.py`(参数校验与命令处理)、`env.py`(凭据读取)、`parallel.py`(线程池)。

Python 只用 `uv` 管:不动系统自带 Python,不用 `sudo pip install`,不引入 pyenv / conda / poetry。

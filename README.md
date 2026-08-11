# video-txt

把本地视频变成**中文字幕视频**或**中文配音视频**，一条命令跑完转写、翻译、封装。

## 快速开始

进项目目录，跑这一条：

```bash
cd /Users/sun/Documents/docx/video-txt

uv run video-txt run '/Users/sun/Documents/docx/video/你的视频.mp4' \
  --provider deepseek \
  --mux-mode hard
```

跑完在**视频所在目录**得到三个文件：

| 文件 | 是什么 |
| --- | --- |
| `你的视频.srt` | Whisper 转写的原文字幕 |
| `你的视频.zh.srt` | 翻译后的中文字幕 |
| `你的视频.zh-burned.mp4` | 中文字幕烧进画面的成片（微信公众号用这个） |

去掉 `--mux-mode hard` 就是软字幕版，输出 `你的视频.zh-subbed.mp4`，字幕可以在播放器里开关。

想先看看会执行什么、不真跑，加 `--dry-run`。

参考耗时：24 分钟的 480p 视频，CPU 转写 7.5 分钟 + 翻译 30 秒（604 条字幕，8 批并发）+ 烧字幕 42 秒，合计约 8.5 分钟。

### 第一次用之前（只做一次）

```bash
brew install ffmpeg          # 系统依赖
uv sync                      # Python 依赖，之后 uv run 会自动保持同步
```

再确认翻译用的 key 已经就绪（见文末「凭据约定」），这条命令只检查、不打印 key：

```bash
uv run python -c "from video_txt.env import resolve_api_key; resolve_api_key('DEEPSEEK_API_KEY'); print('key 就绪')"
```

可选功能按需装：

```bash
uv sync --extra dub   # 中文配音（edge-tts）
uv sync --extra mlx   # Apple Silicon GPU 转写（mlx-whisper，比 CPU 快数倍）
```

`--extra mlx` 会另外下载一份 1.5G 的模型权重，短视频不一定划算；长视频再装。

## 按场景选命令

以下都以 `V='/Users/sun/Documents/docx/video/你的视频.mp4'` 为例，先 `V=...` 设一下会少打很多字。

**出一个中文硬字幕成片（最常用）**

```bash
uv run video-txt run "$V" --provider deepseek --mux-mode hard
```

**原视频自带烧死的字幕（底部已经有字）**

把新字幕放到顶部，让两边各占一头，不要去盖：

```bash
uv run video-txt run "$V" --provider deepseek --mux-mode hard --hard-subtitle-layout top
```

实测下来这比黑罩好用：黑罩要么盖不干净（默认半透明会透出旧字），要么为了盖干净而整片常驻、
连没字幕的画面也挡掉底部一大块。顶部布局唯一要留意的是幻灯片类画面——标题通常也在顶部，
可能压在一起，讲话人镜头则没有这个问题。

真要用黑罩的话见「硬字幕外观」里的 `bottom-box`。

**已经有原文 `.srt`，不想重新转写**

```bash
uv run video-txt run "$V" --provider deepseek --mux-mode hard --subtitle '/绝对路径/已有字幕.srt'
```

同名 `.srt` 就在视频旁边时会自动复用，这个参数只在字幕不同名或不同目录时才需要。

**只要一份中文字幕文件，不封装视频**

```bash
uv run video-txt translate '/绝对路径/字幕.srt' --provider deepseek
```

**只要文字稿，不翻译**

```bash
uv run video-txt transcribe "$V"                 # 输出 .txt
uv run video-txt transcribe "$V" --format srt    # 输出 .srt
```

**做中文配音**

```bash
uv run video-txt dub "$V" --provider deepseek --keep-bgm
```

## 重跑规则

每个阶段的产物都落盘，重跑时**已经存在的阶段自动跳过**。所以调字幕样式时直接重跑就行，不会重新花钱转写和翻译——上面那个 24 分钟的视频，改黑罩重跑只花 40 秒。

| 想强制重做 | 加这个参数 |
| --- | --- |
| 重新转写 | `--retranscribe` |
| 重新翻译 | `--retranslate` |
| 覆盖已存在的输出视频 | `--overwrite-video` |
| 换个输出文件名（保留旧版对比） | `-o '/绝对路径/输出.mp4'` |

翻译中途 Ctrl-C 或报错，已完成的批次记在 `<字幕名>.partial.jsonl` 里，重跑自动续上，全部成功后该文件自动删除。

## 参数速查

### 翻译

| 参数 | 说明 |
| --- | --- |
| `--provider deepseek` | 一次性设好接口地址、key 变量名和默认模型 `deepseek-v4-flash` |
| `--model` / `--base-url` / `--api-key-env` | 换别的服务时逐项覆盖；也认 `OPENAI_MODEL`、`OPENAI_BASE_URL` |
| `--batch-chars 2400` | 每批字符数，模型上下文小就调小 |
| `--concurrency 4` | 并发批数，长视频提速明显；被限流就调小 |
| `--context-cues 3` | 把前几条字幕作为上下文一起发，术语和语气更连贯 |
| `--preserve-term MCP` | 保留不译的术语，默认已含 Claude、Claude Code、Anthropic、MCP |
| `--note '保持轻松的教程口吻'` | 追加翻译要求 |
| `--no-resume` | 关掉断点续传 |

模型返回不合法时会自动二分拆批重试，原始响应存到 `<字幕名>.debug/` 供事后复盘。

### 硬字幕外观

字号和边距的单位是**视频原始分辨率下的像素**，三种布局一致。不传时按分辨率自动换算：字号约为画面高度的 4.5%，底边距约 5%（1080p ≈ 49px / 54px，480p ≈ 22px / 24px）。

```bash
uv run video-txt run "$V" --provider deepseek --mux-mode hard \
  --hard-subtitle-font 'PingFang SC' \
  --hard-subtitle-font-size 44 \
  --hard-subtitle-margin-v 60
```

`bottom-box` 的黑罩用 `--hard-subtitle-box-height`（默认 0.22，占画面高度的比例）和
`--hard-subtitle-box-opacity`（默认 0.68）控制。底部已有烧死字幕时一般优先用 `top` 布局，
下面两条是确实要用黑罩时的实战经验：

- 要**彻底盖住**原有烧死字幕，不透明度得给到 `1.0`，默认的半透明会透出旧字；高度也要按旧字幕
  实际占的比例给足，480p 视频上试出来要 `0.30`，默认的 0.22 会露出旧字幕的上半截。
- 黑罩是**整片时长常驻**的，没有字幕的画面底部也会被遮住。幻灯片类视频要权衡高度，别盖掉正文。

其余：

- 画质用 `--crf`（越小越清晰，默认 20）和 `--preset`（默认 medium）控制；mp4 输出自动加
  `+faststart`，网页和微信里首帧加载更快。
- ffprobe 读不到分辨率时会明确报错，可用 `--video-size 1920x1080` 手动指定。
- 硬字幕需要带 `subtitles` 滤镜的 ffmpeg。若提示缺失：`brew install ffmpeg-full`，再
  `export FFMPEG_PATH=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg`。
- 用默认的 PingFang SC 时，ffmpeg 会打印几行 `Error opening font: ...PingFangUI.ttc`。
  这是 libass 先去试系统保留的那个字体文件、失败后回退到同族的 `PingFang.ttc`，
  **字体照常生效、成片正常**，可以忽略。介意日志干净就换 `--hard-subtitle-font 'Heiti SC'`
  或 `'Hiragino Sans GB'`，这两个零报错。

### 软字幕

字幕编码按输出容器自动选择：mp4 / m4v / mov 用 `mov_text`，mkv 用 `srt`，webm 用 `webvtt`。
`--default-subtitle` 让中文字幕轨默认打开，`--subtitle-codec` 可手动指定编码。

### 转写

| 参数 | 说明 |
| --- | --- |
| `--language en` | 指定源语言，比自动识别快也更准 |
| `--initial-prompt 'Claude Code, MCP, Anthropic'` | 术语提示，显著改善专有名词的转写正确率 |
| `--backend mlx-whisper` | Apple Silicon 走 GPU（需先 `uv sync --extra mlx`） |
| `--model medium` | 换模型，默认 turbo |

Whisper 在开头几句容易吐出跨越二三十秒的长段落，烧成硬字幕就是一整屏文字。整体占比很低
（实测 604 条里只有 13 条超过 40 字），介意的话手动改一下 `.zh.srt` 再 `--retranslate` 之外重跑封装即可。

### 配音

| 参数 | 说明 |
| --- | --- |
| `--keep-bgm` | 保留原声垫底（默认压到 15%，用 `--bgm-volume` 调），不加则整轨替换 |
| `--soft-subtitle` | 同时挂一条可开关的中文字幕轨 |
| `--voice zh-CN-YunxiNeural` | 换音色（`uv run edge-tts --list-voices` 列出全部） |
| `--rate=+10%` | 整体语速 |
| `--max-atempo 1.5` | 单句最大加速倍数，用于把中文塞进原时间轴 |
| `--tts-engine say` | 离线兜底（macOS 自带，音色机械） |

配音片段按句缓存在 `<视频名>.dub-cache/`，重跑只补缺失的句子。跑完会报告多少句需要加速、最大时间轴
漂移多少秒；漂移大就调高 `--max-atempo`，或让译文更短（中文每秒约 4 个字读起来最自然）。

### 看全部参数

```bash
uv run video-txt --help
uv run video-txt run --help
uv run video-txt dub --help
```

## 凭据约定

- 任何真实 key 都不写进 README、脚本、提交信息或示例命令，文档里只出现占位符。
- key 只放两个地方：shell 环境变量，或仓库之外的本地凭据文件。两者都用同样的写法：

```bash
export DEEPSEEK_API_KEY='你的 key'
```

- 环境变量没设时，工具会自动读取默认位置的凭据文件；该文件权限设为 600，路径不记在文档里，
  需要换位置用 `--secrets-file` 指定。
- 换服务用 `--api-key-env` 指定变量名，不要把 key 直接贴进命令行（会留在 shell 历史里）。
- 万一 key 真的进了文档或提交历史，去服务商控制台吊销重发是唯一有效的补救，改文件不够。

## 兼容旧命令

原来的三个脚本仍然可用，参数完全不变，内部转调新 CLI：

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
uv run ruff check .
uv run ruff format .
```

代码分层：`video_txt/subtitles.py`（SRT/ASS 解析与生成）、`media.py`（ffmpeg/ffprobe 探测）、
`translate.py`（翻译引擎）、`transcribe.py`、`mux.py`、`dub.py`、`pipeline.py`（阶段编排）、
`cli.py`（子命令与参数）。

## Python 运行规范

- 不要修改、卸载或覆盖 macOS 自带/现有 Python。
- 不要使用 `sudo pip install`。
- 不要把全局 `python3` 改成新版本，只在项目命令里通过 `uv` 指定。
- 除非以后明确引入 `pyenv`、`conda` 或 `poetry`，本项目只用 `uv` 管理 Python 与依赖。

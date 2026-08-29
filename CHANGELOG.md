# Changelog

本文记录 `video-txt` 从独立脚本到长视频生产工具的功能演进，并给出当前版本中仍然有效的使用方式。
版本概览见 [版本时间线](docs/version-timeline.md)，完整参数以 `uv run --python 3.12 video-txt <命令> --help`
为准。

> Git 历史从 v0.2.0 开始。v0.1.0 根据 v0.2.0 中保留的兼容脚本还原；v0.3.0～v0.5.0
> 没有各自独立的发布提交，首次完整出现在 v0.6.0 快照中。本文不会为这些版本虚构发布日期。

以下示例按当前 v0.7.0 CLI 编写，用于使用相应能力，不用于复现旧版本的依赖环境。示例中的 `V`、
`S`、`T` 分别代表源媒体、源字幕和术语表路径。首次使用先运行 `brew install ffmpeg` 和 `uv sync`。

## Unreleased（计划中的 v0.7.1）

### 计划

- 只翻译局部变更的原文字幕。
- 增量更新翻译审计，而不是重译、重审整部影片。

以上能力尚未包含在 v0.7.0 中。当前执行 `retranscribe-range` 后，需要手工处理对应译文，或对修复后的
完整 SRT 重新运行 `translate`。

## 0.7.0 - 2026-08-24

### 新增：指定区间局部重新识别

长片只有少数字幕听错时，可以只提取相关音频、重新运行 Whisper，再把结果拼回一份新的 SRT。
源字幕始终只读。

按绝对时间修复：

```bash
V='/绝对路径/电影.mkv'
S='/绝对路径/电影.srt'

uv run --python 3.12 video-txt retranscribe-range "$V" \
  --subtitle "$S" \
  --from 00:19:30 --to 00:20:10 \
  --language en --audio-stream 2 \
  -o '/绝对路径/电影.repaired.srt'
```

按一基字幕序号修复：

```bash
uv run --python 3.12 video-txt retranscribe-range "$V" \
  --subtitle "$S" \
  --from-cue 320 --to-cue 335 \
  --language en --audio-stream 2 \
  -o '/绝对路径/电影.repaired.srt'
```

先预览，不写文件：

```bash
uv run --python 3.12 video-txt retranscribe-range "$V" \
  --subtitle "$S" --from 00:19:30 --to 00:20:10 --dry-run
```

使用要点：

- 默认在核心区间前后各增加 `1.5` 秒上下文；用 `--padding` 调整。
- 可继续使用 `--backend`、`--model`、`--initial-prompt`、`--whisper-arg`、
  `--audio-stream` 和 `--refine-subtitles`。
- 局部时间戳会自动换回全片时间，padding 中重复识别的边界字幕会被过滤；重叠会修正，编号会连续化。
- 默认生成 `<字幕名>.repaired.srt` 和对应的 `.retranscription-report.json`；目标已存在时需显式加
  `--overwrite`。
- v0.7.0 只修原文 SRT，不会改已有译文或翻译审计报告。

## 0.6.0 - 2026-08-24

### 新增：可复现项目配置与状态管理

反复精修同一部长片时，不再靠一条越来越长的命令记住配置。`project.video-txt.json` 保存稳定声明，
`.video-txt/state.json` 保存上次成功运行的状态和产物指纹。

创建、检查并运行字幕项目：

```bash
V='/绝对路径/电影.mkv'

uv run --python 3.12 video-txt project init "$V" \
  -o project.video-txt.json \
  --audio-stream 2 --language en \
  --provider lmstudio --model '当前加载的模型名' \
  --term-file project.terms.json --mux-mode hard

uv run --python 3.12 video-txt project status project.video-txt.json
uv run --python 3.12 video-txt project run project.video-txt.json --dry-run
uv run --python 3.12 video-txt project run project.video-txt.json
```

创建配音项目时指定工作流：

```bash
uv run --python 3.12 video-txt project init "$V" \
  -o project.video-txt.json --workflow dub \
  --provider lmstudio --diarize --separate-bgm
```

状态判断规则：

- 视频、原文字幕、术语表和译文使用内容 SHA-256；仅修改 mtime 不会触发重跑。
- 音轨或转写设置变化会使转写及下游阶段失效。
- 原文、术语表或翻译设置变化只使翻译及下游阶段失效。
- 字幕样式只使 mux 失效；音色只使配音失效，并继续复用未变化的语音缓存。
- 相对路径以项目 JSON 所在目录为基准；state 会记录实际 Whisper 后端、模型、音轨和选择依据。
- 配置与 state 只保存凭据环境变量名，不保存 API key 或 token。
- 没有 state 时，已有目标译文或成片被视为外部产物，真实运行会拒绝覆盖；先执行 `--dry-run`，
  或移开文件、修改输出路径。

## 0.5.0（历史功能阶段）

### 新增：项目术语表与翻译审计

用同一份 JSON 固定人名、产品名和行业术语的批准译法：

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

在翻译、字幕成片或配音中启用：

```bash
T='/绝对路径/project.terms.json'

uv run --python 3.12 video-txt translate source.srt \
  --provider lmstudio --term-file "$T"
uv run --python 3.12 video-txt run "$V" \
  --provider lmstudio --term-file "$T"
uv run --python 3.12 video-txt dub "$V" \
  --provider lmstudio --term-file "$T"
```

只审计已有译文，不调用模型、不修改字幕：

```bash
uv run --python 3.12 video-txt audit-translation source.srt source.zh.srt \
  --term-file "$T"
```

使用要点：

- `word` 只匹配完整单词；`phrase` 匹配完整短语。
- `aliases` 只应填写人工批准替换的旧译或误译。
- 术语表同时用于模型提示、精确规范化和翻译后审计。
- 翻译会生成 `<译文名>.translation-audit.json`，检查字幕数量、序号、时间轴、空译、漏译迹象和术语一致性。
- 工具不会因为术语表中出现一个名字，就擅自补写模型遗漏的名字。
- 已有报告默认不覆盖；`translate` 和独立审计需加 `--overwrite`，`run` / `dub` 重译使用
  `--retranslate`。

## 0.4.0（历史功能阶段）

### 新增：多音轨智能选择

`transcribe`、`run` 和 `dub` 会先用 ffprobe 分析语言、标题、默认轨、声道以及评论/无障碍属性，
优先匹配 `--language`，并避开评论音轨和解说音轨。

让工具自动选择英语音轨：

```bash
V='/绝对路径/双语电影.mkv'

uv run --python 3.12 video-txt transcribe "$V" \
  --format srt --language en
```

选择不唯一时，命令会停止并列出候选。确认 FFmpeg 全局 stream index 后显式覆盖：

```bash
uv run --python 3.12 video-txt transcribe "$V" \
  --format srt --language en --audio-stream 2
uv run --python 3.12 video-txt run "$V" \
  --provider lmstudio --language en --audio-stream 2
uv run --python 3.12 video-txt dub "$V" \
  --provider lmstudio --language en --audio-stream 2
```

使用要点：

- 选中的音轨会提取成临时 16 kHz 单声道 WAV；OpenAI Whisper 与 MLX Whisper 复用同一文件。
- 临时音频会在任务结束后清理，不修改源视频。
- 有同名原文字幕时流水线仍会复用它；切换音轨后要同时加 `--retranscribe`。
- 用 `--dry-run` 可先查看音轨依据、提取命令和 Whisper 命令。

## 0.3.1（历史功能阶段）

### 新增：字幕审计与安全清理

已有 SRT 不需要重新跑 Whisper。先审计：

```bash
S='/绝对路径/原文字幕.srt'
V='/绝对路径/对应视频.mkv'

uv run --python 3.12 video-txt audit "$S" \
  --media "$V" --language en
```

确认后生成一份安全清理版：

```bash
uv run --python 3.12 video-txt clean "$S" \
  -o '/绝对路径/原文字幕.clean.srt' \
  --media "$V" --language en
```

只预览清理计划：

```bash
uv run --python 3.12 video-txt clean "$S" \
  -o '/绝对路径/原文字幕.clean.srt' \
  --media "$V" --language en --dry-run
```

审计会检测整窗幻觉、空字幕、零时长、倒序、重叠、重复、错误语种、版式超限和转写过早结束。
`clean` 只删除整窗短句幻觉和空块、合并可确定的零时长碎片、重新连续编号；重叠、倒序、普通长句和
版式告警留给人工复核。源字幕永不覆盖，目标已存在时必须显式加 `--overwrite`。

## 0.3.0（历史功能阶段）

### 新增：词级时间戳字幕精修

Whisper 默认按解码窗口输出，容易产生过长字幕。启用词级时间戳后，工具会按句末、停顿、时长和显示宽度
重建字幕块。

完整流水线：

```bash
V='/绝对路径/视频.mp4'

uv run --python 3.12 video-txt run "$V" \
  --provider lmstudio --mux-mode hard --refine-subtitles
```

只生成精修后的原文字幕：

```bash
uv run --python 3.12 video-txt transcribe "$V" \
  --format srt --refine-subtitles
```

使用要点：

- 优先在完整句末和超过约 `0.8` 秒的停顿处断句，每条约不超过 `6` 秒、最多两行。
- 东亚全角字符按双倍显示宽度计算；OpenAI Whisper 和 MLX Whisper 都支持。
- 成功后写入 `<视频名>.srt` 与可复用的 `<视频名>.words.json`。
- `.srt` 被人工修改且校验不匹配时会停止，不会直接覆盖；确认丢弃修改后才加 `--retranscribe`。
- 精修需要重新取得词级时间戳，不能与 `--subtitle` 同时使用。

## 0.2.x 阶段增强 - 2026-08-11 至 2026-08-17

v0.2.0 发布后的连续增强主要集中在配音自然度、本机翻译与长任务安全性。

### 整句配音与自动时长适配

当前默认按整句而不是逐条字幕合成，超时句会自动用更快语速重说，再用有限波形拉伸兜底：

```bash
uv run --python 3.12 video-txt dub "$V" --provider lmstudio
```

通常无需额外参数。需要恢复逐行合成或调整容忍度时：

```bash
uv run --python 3.12 video-txt dub "$V" --provider lmstudio \
  --voice-unit line --max-atempo 1.5
```

响度归一化、片段边缘淡入淡出和防咔哒处理会自动应用，无需开关。

### 说话人分离与多角色音色

```bash
uv sync --extra diarize
export HF_TOKEN='你的 Hugging Face token'

uv run --python 3.12 video-txt dub "$V" --provider lmstudio \
  --diarize --speakers 2 \
  --speaker-voice SPEAKER_01=zh-CN-YunxiNeural
```

说话人结果保存在 `<视频名>.speakers.json`；需要重算时加 `--rediarize`。

### F5-TTS、CosyVoice 与 IndexTTS 原声克隆

F5-TTS 可作为项目 extra 安装：

```bash
uv sync --extra clone
uv run --python 3.12 video-txt dub "$V" --provider lmstudio \
  --tts-engine f5-tts --diarize
```

IndexTTS 和 CosyVoice 使用各自独立的仓库与 Python 环境，避免依赖冲突。以 IndexTTS 为例：

```bash
uv run --python 3.12 video-txt dub "$V" --provider lmstudio \
  --tts-engine index-tts --clone-repo '/绝对路径/index-tts'
```

CosyVoice 对应使用 `--tts-engine cosyvoice --clone-repo <仓库> --clone-model <模型目录>`。
可用 `--clone-reference [SPEAKER=]AUDIO` 提供人工挑选的参考音频；否则从源视频自动截取。

### Demucs 背景声分离

```bash
uv sync --extra separate
uv run --python 3.12 video-txt dub "$V" --provider lmstudio --separate-bgm
```

Demucs 会去掉原说话声，保留音乐、掌声和环境声，再叠加中文配音。分离结果缓存在
`<视频名>.dub-cache/bgm/`，后续重跑可直接复用。

### 多语言输出与本机 LM Studio 翻译

```bash
uv run --python 3.12 video-txt translate source.srt \
  --provider lmstudio --target-language Japanese
```

LM Studio 需要预先加载模型并启动本地 OpenAI 兼容服务器；不需要 API key。服务不在默认地址时加
`--base-url`，需要固定模型时加 `--model`。

### 转写质量检查、并发与内存安全

转写后会自动检查长时间重复、指定语种与文本明显不符、字幕远早于媒体结束等问题；流水线发现问题会停止。
确认结果可接受时才使用：

```bash
uv run --python 3.12 video-txt run "$V" \
  --provider lmstudio --skip-transcript-check
```

翻译默认并发执行、失败重试并写入断点文件。可根据本机模型或服务限流调整：

```bash
uv run --python 3.12 video-txt translate source.srt \
  --provider lmstudio --concurrency 2 --retries 3 --timeout 600
```

并发清理、流式处理和内存保护属于内部行为，无需额外参数。

## 0.2.0 - 2026-08-11

### 新增：统一的 `video-txt` CLI

三个独立脚本被工程化为一套 CLI，并保留旧脚本兼容入口。

单独转写：

```bash
uv run --python 3.12 video-txt transcribe "$V" --format srt
```

单独翻译：

```bash
uv run --python 3.12 video-txt translate source.srt --provider lmstudio
```

已有字幕，翻译并封装视频：

```bash
uv run --python 3.12 video-txt mux "$V" source.srt \
  --provider lmstudio --mux-mode soft
```

从视频到硬字幕成片：

```bash
uv run --python 3.12 video-txt run "$V" \
  --provider lmstudio --mux-mode hard
```

生成中文配音成片：

```bash
uv sync --extra dub
uv run --python 3.12 video-txt dub "$V" --provider lmstudio
```

### 流水线控制与恢复

- `--mux-mode soft` 嵌入可开关字幕轨；`--mux-mode hard` 把字幕烧入画面。
- `--dry-run` 只打印计划，不调用翻译、TTS 或 ffmpeg 写成片。
- 已存在的阶段产物会复用；用 `--retranscribe`、`--retranslate`、`--overwrite-video` 强制重做。
- 翻译批次失败会自动重试，不合法的大批次可拆分重试。
- 已完成翻译批次写入 `<字幕名>.partial.jsonl`，中断后重跑会续传，全部成功后自动删除。
- 配音片段按内容、音色和语速缓存到 `<视频名>.dub-cache/`，重跑只补缺失片段。

v0.2.0 快照包含 87 项自动化测试。

## 0.1.0（Git 历史前，根据兼容入口还原）

### 原型：三个独立脚本

转写、翻译和封装需要人工按顺序执行：

```bash
V='/绝对路径/视频.mp4'

uv run --python 3.12 python transcribe_media.py "$V"
uv run --python 3.12 python translate_srt.py '/绝对路径/视频.srt' \
  --provider lmstudio
uv run --python 3.12 python translate_and_mux_video.py \
  "$V" '/绝对路径/视频.srt' --mux-mode hard
```

这些入口在当前版本仍可用，内部转调统一 CLI。它们适合兼容旧命令或旧自动化；新项目应直接使用
`video-txt transcribe`、`translate`、`mux`、`run` 和 `dub`。

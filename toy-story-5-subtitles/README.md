# Toy Story 5 中文字幕测试

- 输入：本地 1080p `.mkv` 片源（路径不入库）
- 选择音轨：stream 2（标题 `Eng`）
- 目标文件：`Toy.Story.5.2026.zh-CN.srt`
- 流程：提取 16 kHz 单声道音频 → Whisper 英语转写 → 简体中文翻译 → SRT 校验

## 结果

- 转写：whisper.cpp Large v3 Turbo Q5（英语音轨，CPU/BLAS）
- 翻译：本机 LM Studio，Qwen3.5 35B-A3B
- 最终字幕：1683 条
- 清理：移除 8 条明确的 29.98 秒整窗音乐幻觉，合并 3 个零时长尾词
- 校验：编号连续、时间戳与英文稿逐条一致、无空字幕、无零时长、无重叠、中文覆盖率 100%
- SHA-256：`25de4d1eb71c1d9067637557c4fae6a9276c69f88f9f016bfc1899e21b7b9636`

源文件是 Telesync 影院录音；专有名词和嘈杂场景仍可能存在个别听写误差。

## v0.5 术语表

本目录的 [`project.terms.json`](project.terms.json) 固定主要角色名和片中设备名。重新翻译时使用:

```bash
uv run --python 3.12 video-txt translate work/Toy.Story.5.2026.en.cleaned.srt \
  --provider lmstudio --term-file toy-story-5-subtitles/project.terms.json
```

译文旁会自动生成 `.translation-audit.json`。只检查已经翻好的字幕时改用
`video-txt audit-translation <英文.srt> <中文.srt> --term-file ...`。

术语表替代了原先 `repair_translation.py` 里的全局角色名规范化。

## v0.8 人工校订

那个脚本的另一半——34 条按字幕编号写死的剧情语义修订——现在是
[`revisions.json`](revisions.json)，**锚定原文台词而不是编号**：

```bash
uv run --python 3.12 video-txt revise \
  work/Toy.Story.5.2026.en.cleaned.srt '/路径/Toy.Story.5.2026.zh-CN.srt' \
  -o '/路径/Toy.Story.5.2026.zh-CN.revised.srt' \
  --revisions toy-story-5-subtitles/revisions.json
```

按编号写死是不能留的：本片 `clean` 已经删了 8 条、合并了 3 条，上游再动一次，
下面所有编号都平移。实测在这份 1683 条的字幕上再删 3 条，34 条修订**全部错行且没有任何提示**——
`#110` 会写到 `having...` 上。锚定原文则跟着台词走，34 条全部唯一落位、零漂移。
锚点匹配不上会直接报错，不会静默跳过。

`repair_translation.py` 已随本次一并删除，它做的两件事分别由术语表和校订文件接管。

## 待办：现有译文还没过术语表

`Toy.Story.5.2026.zh-CN.srt` 是在术语表之前翻的，`enforce_terminology` 从未在它上面跑过。
现在审计仍报 15 处（落在 9 个字幕块），例如 cue 144 用了「乔丹家的双胞胎」而非批准的「乔丹双胞胎」：

```bash
uv run --python 3.12 video-txt audit-translation \
  work/Toy.Story.5.2026.en.cleaned.srt '/路径/Toy.Story.5.2026.zh-CN.srt' \
  --term-file toy-story-5-subtitles/project.terms.json \
  --revisions toy-story-5-subtitles/revisions.json
```

清掉的办法是带术语表重翻一次（`translate --retranslate --term-file`），或把这 9 块补进校订文件。

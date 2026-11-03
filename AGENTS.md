# video-txt

本项目把本地视频或音频转换为中文字幕视频、外挂字幕、中文配音或文字稿。
CLI 入口是 `video-txt`，实现位于 `video_txt/cli.py`。

## 自然语言视频操作

- 用户提出加中文字幕、硬编码/烧录字幕、翻译 `.srt`、视频转文字、中文配音、局部重识别或字幕质检等需求时，使用 `$video-txt` Skill。
- 不要绕过项目 CLI 自行拼接 FFmpeg、Whisper 或翻译命令。
- 按 `$video-txt` Skill 判断源语言、字幕封装方式、翻译提供商、已有字幕复用和长任务执行方式。
- 视频包含英文页面、手机聊天、看板、幻灯片等画面文字时，使用 `video-txt screen`：
  `scan` 留存截图与 OCR → 校正原文 → `translate` 或 Agent 翻译 → 校对计划 → `validate` → `render`。
  完整设计见 `docs/screen-text-workflow.md`，不要再写一次性取图与烧录脚本。
- 画面字幕的原文、完整译文、显示译文和时间轴在 `plan.json` 修改。
  `status=ready` 表示 Agent 或人工已核对证据，不要求用户再次批准已授权的翻译任务。
  口述字幕按现有流程处理，作为配对 SRT 传入 `screen render`；时间重叠先解决冲突。

## 翻译整季前

- 先用 `video-txt draft-terms <全季 .srt> -o 术语表.json` 起草术语表，填完每条 `target` 再开翻。
- 换季复用上一季术语表时加 `--against 旧表.json`：只列新名字，并对与旧词条共用同一个词的新名字告警。
- 不要手工通读字幕找专有名词，也不要跳过术语表直接逐集翻译。

## 修改已有译文

- 用户指出某几句译得不对时，写入锚定原文台词的校订文件，再用 `video-txt revise --revisions` 套用。
- 不要直接编辑生成的 `.zh.srt`：它是生成物，下一次重新翻译就会覆盖。
- 不要写按字幕编号定位的一次性修订脚本：上游 `clean` / `retranscribe-range` 会平移编号，
  修订会静默写到别的台词上。
- 校订文件格式、报错含义、与术语表的分工，见 `$video-txt` Skill 的「改译文」一节。

## 命令环境

- 在项目根目录运行命令。
- Python 和项目 CLI 统一通过 `uv run --python 3.12` 运行，不修改或替换系统 Python。
- 生成物默认放在源视频或字幕所在目录；汇报时列出绝对路径。

## 修改与验证

- 修改代码后，按照 `$verify` Skill 做端到端验证，不只运行单元测试。
- 详细用法见 `README.md`，技术取舍与实测数据见 `NOTES.md`。

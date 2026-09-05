# video-txt

本地视频 → 中文字幕视频 / 中文配音 / 文字稿。所有能力都在 `video-txt` CLI 里，
入口 `video_txt/cli.py`，子命令表在文件末尾的 `COMMANDS`。

- 跑命令：`uv run --python 3.12 video-txt <子命令> ...`
- 用户用自然语言提视频需求时（加中文字幕、硬编码字幕、配音、转文字、翻译 .srt、
  修某几句字幕），走 **`video-txt` skill**，别自己拼 ffmpeg 或 whisper 命令。
- 用户说某几句**译**得不对时：写进校订文件走 `video-txt revise --revisions`，
  **不要直接编辑 `.srt`，也不要写按字幕编号写死的一次性脚本**——译文是生成物，
  重翻就没了；编号会被上游的 `clean` / `retranscribe-range` 平移，改到别人的台词上。
  格式与规则见 `video-txt` skill 的「改译文」一节。
- 改代码后按 `.claude/skills/verify/SKILL.md` 真跑一遍，不要只跑单测。
- 用法细节在 [README.md](README.md)，技术取舍的实测数据在 [NOTES.md](NOTES.md)。

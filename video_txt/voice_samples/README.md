# 曼波参考音频

- `mambo.wav`：2026-10-14 用户试听确认的声音，4.04 秒，32 kHz 单声道 PCM。
- 原文件：[Tsukimisaka/MamboTTS — models/refer.wav](https://github.com/Tsukimisaka/MamboTTS/blob/46fe19cc9e91d601f8cdd2a54ad7b85a40a24593/models/refer.wav)。逐字节复制，未变速、变调或重新编码。
- SHA-256：`6031533a08a3c1fc9daef66620db5c287cd48c6c3b44b3912922db1f644715da`。
- `mambo.txt`：参考音频逐字稿，来自上游 `engine_contract.py` 的 `DEFAULT_REF_TEXT`。
- `mambo.emo.wav`：2026-10-14 用户从本机音乐库的 `我要睡觉了.mp3`（来源不明的曼波语音片段）转成
  3.23 秒、32 kHz 单声道 PCM，未变速、变调。它不提供音色，只给 IndexTTS 当"说话方式"参考
  （拖长的语调与节奏）。SHA-256：`173dfbc8a0e68850a5b0189433a32793f8e91a8cff61702c2ff4be3afd64fdf6`。
  用户在 A（只用 `mambo.wav`）、B（只用睡觉那条）、C（音色 `mambo.wav` + 语气睡觉那条，强度 0.8）
  三版对比试听后选定 C。
- 上游仓库的 MIT 许可随附于 `MamboTTS-LICENSE`；模型权重未包含在此目录。

`--voice mambo` / `--voice 曼波` 使用 IndexTTS 克隆 `mambo.wav` 的音色，并以强度 0.8 借用
`mambo.emo.wav` 的说话方式。
此处没有接入上游专门训练的 GPT-SoVITS 曼波模型，合成结果不保证与它完全相同。
参考音频、逐字稿与说话方式参考都参与现有语音片段缓存的指纹计算。

通用规则：任何 `--clone-reference voice.wav` 旁边若有 `voice.emo.wav`，index-tts 都会同样借用；
其他引擎忽略它。没有这个文件的参考，行为与指纹不变。

预设在 CLI 校验入口展开为已有的 `index-tts`、`clone_repo`、`clone_references` 设置。
默认 checkout 为 `~/tools/index-tts`，显式 `--clone-repo` 优先。单人配音默认使用此预设；
显式选择其他音色、引擎及多说话人模式仍遵循原有规则，项目格式不变。
项目文件保存展开后的引用。缺失资源或冲突参数直接报错，不会换回晓晓或其他人的参考。

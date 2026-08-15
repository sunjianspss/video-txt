"""Synthesizes a batch of cloned-voice clips, loading the model once.

Run as ``python clone_worker.py manifest.json``. This file is deliberately
standalone: it imports nothing from video_txt, so the interpreter that runs it
can be the separate virtualenv CosyVoice or F5-TTS were installed into.

Every clip is written to a temporary name and renamed once it is complete, so a
run that dies halfway leaves behind only finished clips — which the next run
finds in the cache and skips, instead of paying for the model load again.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

PROMPT_SAMPLE_RATE = 16000


class WorkerError(RuntimeError):
    pass


def job_speed(job: dict, fallback: float) -> float:
    """This clip's speed multiplier: its own re-speak speed, or the run's."""
    return float(job.get("speed") or fallback)


class F5Engine:
    """https://github.com/SWivid/F5-TTS — pip installable, downloads its own weights."""

    def __init__(self, manifest: dict) -> None:
        try:
            from f5_tts.api import F5TTS
        except ImportError as exc:
            raise WorkerError(
                "Missing dependency: f5-tts.\n"
                "Install it with: uv pip install f5-tts\n"
                "Or point at an interpreter that has it: --clone-python /path/to/venv/bin/python"
            ) from exc

        arguments = {"model": manifest["model"]}
        if manifest.get("device"):
            arguments["device"] = manifest["device"]
        try:
            self.model = F5TTS(**arguments)
        except TypeError:
            # Releases before 1.0 call the same argument model_type.
            arguments["model_type"] = arguments.pop("model")
            self.model = F5TTS(**arguments)
        self.speed = float(manifest.get("speed") or 1.0)

    def synthesize(self, job: dict, output: Path) -> None:
        self.model.infer(
            ref_file=job["reference_audio"],
            ref_text=job["reference_text"],
            gen_text=job["text"],
            file_wave=str(output),
            speed=job_speed(job, self.speed),
            show_info=lambda *_args, **_kwargs: None,
        )


class CosyVoiceEngine:
    """https://github.com/FunAudioLLM/CosyVoice — cloned and installed by hand."""

    def __init__(self, manifest: dict) -> None:
        try:
            import torch
            import torchaudio
        except ImportError as exc:
            raise WorkerError(f"CosyVoice needs torch and torchaudio: {exc}") from exc

        self.torch = torch
        self.torchaudio = torchaudio
        self.model = self.load_model(manifest["model"])
        self.speed = float(manifest.get("speed") or 1.0)
        parameters = inspect.signature(self.model.inference_zero_shot).parameters
        # Newer builds take the prompt as a path; older ones want the waveform.
        self.wants_path = "prompt_wav" in parameters
        self.prompts: dict[str, object] = {}

    @staticmethod
    def load_model(model_dir: str) -> object:
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError:
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2 as AutoModel
            except ImportError as exc:
                raise WorkerError(
                    "Missing dependency: cosyvoice.\n"
                    "Clone and install it as its README describes, then point at it:\n"
                    "  --clone-repo /path/to/CosyVoice "
                    "--clone-model /path/to/CosyVoice/pretrained_models/CosyVoice2-0.5B\n"
                    "Add --clone-python /path/to/CosyVoice/.venv/bin/python if it has "
                    "a virtualenv of its own."
                ) from exc
        if not Path(model_dir).is_dir():
            raise WorkerError(f"CosyVoice model directory not found: {model_dir}")
        return AutoModel(model_dir)

    def prompt(self, audio: str) -> object:
        if self.wants_path:
            return audio
        if audio not in self.prompts:
            from cosyvoice.utils.file_utils import load_wav

            self.prompts[audio] = load_wav(audio, PROMPT_SAMPLE_RATE)
        return self.prompts[audio]

    def synthesize(self, job: dict, output: Path) -> None:
        pieces = [
            result["tts_speech"]
            for result in self.model.inference_zero_shot(
                job["text"],
                job["reference_text"],
                self.prompt(job["reference_audio"]),
                stream=False,
                speed=job_speed(job, self.speed),
            )
        ]
        if not pieces:
            raise WorkerError(f"CosyVoice returned no audio for {job['text'][:40]!r}")
        self.torchaudio.save(
            str(output), self.torch.cat(pieces, dim=1), self.model.sample_rate, format="wav"
        )


class IndexTTSEngine:
    """https://github.com/index-tts/index-tts — cloned and installed by hand.

    Its checkpoints ship an IndexTTS-2.5 loader (infer_v2_5) in newer checkouts
    and an IndexTTS-2 one (infer_v2) in older ones; both spell the class
    IndexTTS2 and take the same reference-plus-text call, so the constructor
    keeps whichever import worked and reads the signatures for the rest.
    """

    def __init__(self, manifest: dict) -> None:
        IndexTTS2 = self.import_model()
        model_dir = Path(manifest["model"])
        config = model_dir / "config.yaml"
        if not config.is_file():
            raise WorkerError(
                f"IndexTTS weights not found in {model_dir}.\n"
                "Download them into the checkout:\n"
                f"  uvx --from huggingface-hub hf download IndexTeam/IndexTTS-2.5 "
                f"--local-dir {model_dir}"
            )

        arguments: dict = {"cfg_path": str(config), "model_dir": str(model_dir)}
        parameters = inspect.signature(IndexTTS2.__init__).parameters
        # Half precision is what both loaders recommend; they name it differently.
        if "use_bf16" in parameters:
            arguments["use_bf16"] = True
        elif "use_fp16" in parameters:
            arguments["use_fp16"] = True
        if manifest.get("device") and "device" in parameters:
            arguments["device"] = manifest["device"]
        self.model = IndexTTS2(**arguments)
        self.infer_parameters = inspect.signature(self.model.infer).parameters
        self.lang = manifest.get("lang")
        self.speed = float(manifest.get("speed") or 1.0)

    @staticmethod
    def import_model() -> type:
        try:
            from indextts.infer_v2_5 import IndexTTS2
        except ImportError:
            try:
                from indextts.infer_v2 import IndexTTS2
            except ImportError as exc:
                raise WorkerError(
                    "Missing dependency: indextts.\n"
                    "Clone https://github.com/index-tts/index-tts and run 'uv sync' "
                    "inside it, then point at the checkout with --clone-repo "
                    "/path/to/index-tts. Its own .venv is picked up automatically."
                ) from exc
        return IndexTTS2

    def synthesize(self, job: dict, output: Path) -> None:
        arguments: dict = {
            "spk_audio_prompt": job["reference_audio"],
            "text": job["text"],
            "output_path": str(output),
        }
        if "verbose" in self.infer_parameters:
            arguments["verbose"] = False
        # IndexTTS-2.5 requires the language; the IndexTTS-2 loader has no
        # such parameter and reads everything as Chinese or English.
        if "lang" in self.infer_parameters:
            if not self.lang:
                raise WorkerError(
                    "IndexTTS-2.5 speaks zh, en, ja and es, and the target "
                    "language is none of those."
                )
            arguments["lang"] = self.lang
        speed = job_speed(job, self.speed)
        if "duration_factor" in self.infer_parameters and abs(speed - 1.0) > 1e-6:
            # duration_factor above 1.0 slows the speech down, so a speed
            # multiplier maps to its reciprocal.
            arguments["duration_factor"] = min(max(1.0 / speed, 0.5), 2.0)
        self.model.infer(**arguments)


def build_engine(manifest: dict) -> F5Engine | CosyVoiceEngine | IndexTTSEngine:
    engine = manifest["engine"]
    if engine == "f5-tts":
        return F5Engine(manifest)
    if engine == "cosyvoice":
        return CosyVoiceEngine(manifest)
    if engine == "index-tts":
        return IndexTTSEngine(manifest)
    raise WorkerError(f"Unknown cloning engine: {engine}")


def pending_jobs(jobs: list[dict]) -> list[dict]:
    return [
        job
        for job in jobs
        if not (Path(job["output"]).is_file() and Path(job["output"]).stat().st_size > 0)
    ]


def check_references(jobs: list[dict], *, engine: str) -> None:
    for job in jobs:
        if not Path(job["reference_audio"]).is_file():
            raise WorkerError(f"Reference audio not found: {job['reference_audio']}")
    if engine == "cosyvoice" and any(not job["reference_text"].strip() for job in jobs):
        print(
            "Warning: a reference clip has no transcript, which CosyVoice clones badly from. "
            "Write the words spoken in it next to the clip as a .txt file.",
            file=sys.stderr,
        )


def run(manifest: dict) -> None:
    # Torch on Apple Silicon: a few ops in these models are not implemented on
    # MPS yet; falling back to the CPU for those beats not running at all.
    # Must be set before torch is imported, which build_engine does.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    for entry in manifest.get("sys_path", []):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    jobs = pending_jobs(manifest["jobs"])
    if not jobs:
        return
    check_references(jobs, engine=manifest["engine"])

    print(f"Loading {manifest['engine']} ({manifest['model']})...", flush=True)
    engine = build_engine(manifest)
    print(f"Synthesizing {len(jobs)} voice clips...", flush=True)

    for number, job in enumerate(jobs, start=1):
        output = Path(job["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        # Keep the .wav ending: both engines choose the container from it.
        temporary = output.with_suffix(".part.wav")
        engine.synthesize(job, temporary)
        if not temporary.is_file():
            raise WorkerError(f"{manifest['engine']} wrote no audio for {job['text'][:40]!r}")
        temporary.replace(output)
        print(f"  {number}/{len(jobs)} clips done", flush=True)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: clone_worker.py MANIFEST.json", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        run(manifest)
    except WorkerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

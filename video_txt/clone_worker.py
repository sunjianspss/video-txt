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
import sys
from pathlib import Path

PROMPT_SAMPLE_RATE = 16000


class WorkerError(RuntimeError):
    pass


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
            speed=self.speed,
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
                speed=self.speed,
            )
        ]
        if not pieces:
            raise WorkerError(f"CosyVoice returned no audio for {job['text'][:40]!r}")
        self.torchaudio.save(
            str(output), self.torch.cat(pieces, dim=1), self.model.sample_rate, format="wav"
        )


def build_engine(manifest: dict) -> F5Engine | CosyVoiceEngine:
    engine = manifest["engine"]
    if engine == "f5-tts":
        return F5Engine(manifest)
    if engine == "cosyvoice":
        return CosyVoiceEngine(manifest)
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

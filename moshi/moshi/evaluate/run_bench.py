# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Generate PersonaPlex outputs for FullDuplexBench categories.

Walks ``--bench-root/<dataset>/*.wav``, runs PersonaPlex offline inference
(voice + text system prompt), and writes ``--results-root/<dataset>/<id>.wav``
plus a sidecar JSON for scoring with ``python -m moshi.evaluate.evaluate``.

Text prompts follow the PersonaPlex README FullDuplexBench guidance:

- Pause / Backchannel / Turn taking: ``You enjoy having a good conversation.``
- User Interruption: wise-and-friendly-teacher assistant prompt

Expected bench layout (user-supplied FullDuplexBench dump, not in git)::

    bench-root/
      synthetic_pause_handling/   *.wav
      candor_pause_handling/      *.wav
      icc_backchannel/            *.wav
      candor_turn_taking/         *.wav + metadata *.json
      synthetic_user_interruption/ *.wav + metadata *.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch

from ..client_utils import make_log
from ..models import loaders
from ..offline import (
    _get_voice_prompt_dir,
    generate_offline_sample,
    load_offline_session,
)

logger = logging.getLogger(__name__)

DATASET_NAMES = [
    "synthetic_pause_handling",
    "candor_pause_handling",
    "icc_backchannel",
    "candor_turn_taking",
    "synthetic_user_interruption",
]

METADATA_DATASETS = {
    "candor_turn_taking",
    "synthetic_user_interruption",
}

PROMPT_CONVERSATION = "You enjoy having a good conversation."
PROMPT_INTERRUPTION = (
    "You are a wise and friendly teacher. Answer questions or provide advice "
    "in a clear and engaging way."
)

DATASET_TEXT_PROMPT = {
    "synthetic_pause_handling": PROMPT_CONVERSATION,
    "candor_pause_handling": PROMPT_CONVERSATION,
    "icc_backchannel": PROMPT_CONVERSATION,
    "candor_turn_taking": PROMPT_CONVERSATION,
    "synthetic_user_interruption": PROMPT_INTERRUPTION,
}


def log(level: str, msg: str) -> None:
    print(make_log(level, msg))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-root",
        type=Path,
        required=True,
        help="Root directory containing FullDuplexBench dataset folders.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/fullduplex_personaplex"),
        help="Where to write wav + json outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=None,
        help=f"Datasets to run (default: all of {DATASET_NAMES}).",
    )
    parser.add_argument(
        "--voice-prompt",
        type=str,
        default="NATF2.pt",
        help="Voice prompt basename inside --voice-prompt-dir (default: NATF2.pt).",
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        default=None,
        help="Directory of voice prompts; if omitted, download voices.tgz from HF.",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default=None,
        help="Override dataset-specific text prompt for all samples.",
    )
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--moshi-weight", type=str, default=None)
    parser.add_argument("--mimi-weight", type=str, default=None)
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo (default: nvidia/personaplex-7b-v1).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--temp-audio", type=float, default=0.8)
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--topk-audio", type=int, default=250)
    parser.add_argument("--topk-text", type=int, default=25)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples that already have results json + wav.",
    )
    parser.add_argument(
        "--log-tokens",
        action="store_true",
        help="Log every text token (noisy; off by default for bench).",
    )
    return parser.parse_args()


def _find_metadata(wav_path: Path, dataset: str) -> Path | None:
    candidates = [
        wav_path.with_suffix(".json"),
        wav_path.parent / f"{wav_path.stem}_meta.json",
        wav_path.parent / "metadata" / f"{wav_path.stem}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    parent_meta = wav_path.parent.parent / f"{dataset}_meta" / f"{wav_path.stem}.json"
    if parent_meta.exists():
        return parent_meta
    return None


def _model_text_for_score(pieces: list[str]) -> list[str]:
    """Keep content pieces; drop special-token labels for ASR/judge readability."""
    skip = {"EPAD", "BOS", "EOS", "PAD"}
    return [p for p in pieces if p not in skip]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    bench_root = args.bench_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    datasets = args.dataset_names or DATASET_NAMES

    if not bench_root.exists():
        raise SystemExit(f"Bench root does not exist: {bench_root}")

    voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    if voice_prompt_dir is None or not os.path.exists(voice_prompt_dir):
        raise FileNotFoundError(f"voice_prompt_dir does not exist: {voice_prompt_dir}")
    voice_prompt_path = os.path.join(voice_prompt_dir, args.voice_prompt)
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(
            f"Voice prompt '{args.voice_prompt}' not found in '{voice_prompt_dir}'"
        )
    log("info", f"voice_prompt_path = {voice_prompt_path}")

    with torch.no_grad():
        session = load_offline_session(
            tokenizer_path=args.tokenizer,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            hf_repo=args.hf_repo,
            device=args.device,
            seed=args.seed,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=bool(args.greedy),
            save_voice_prompt_embeddings=False,
            cpu_offload=args.cpu_offload,
        )

        for dataset in datasets:
            if dataset not in DATASET_NAMES:
                log("warning", f"Unknown dataset {dataset}, skipping")
                continue
            src_dir = bench_root / dataset
            if not src_dir.is_dir():
                log("warning", f"Missing dataset folder {src_dir}, skipping")
                continue
            dst_dir = results_root / dataset
            dst_dir.mkdir(parents=True, exist_ok=True)

            text_prompt = args.text_prompt or DATASET_TEXT_PROMPT[dataset]
            wavs = sorted(src_dir.rglob("*.wav"))
            log("info", f"{dataset}: {len(wavs)} wav files; prompt={text_prompt!r}")

            for wav_path in wavs:
                rel = wav_path.relative_to(src_dir)
                out_stem = dst_dir / rel.with_suffix("")
                out_wav = Path(str(out_stem) + ".wav")
                out_json = Path(str(out_stem) + ".json")
                out_wav.parent.mkdir(parents=True, exist_ok=True)

                if args.skip_existing and out_json.exists() and out_wav.exists():
                    continue

                meta = None
                if dataset in METADATA_DATASETS:
                    meta = _find_metadata(wav_path, dataset)
                    if meta is None:
                        log(
                            "warning",
                            f"No metadata JSON for {wav_path}; "
                            "turn-taking/interrupt scoring will fail unless input_path is set later.",
                        )

                try:
                    log("info", f"Running {wav_path.name}")
                    pieces = generate_offline_sample(
                        session,
                        input_wav=str(wav_path),
                        output_wav=str(out_wav),
                        text_prompt=text_prompt,
                        voice_prompt_path=voice_prompt_path,
                        output_text=None,
                        log_tokens=args.log_tokens,
                    )
                    payload = {
                        "input_path": str(meta.resolve()) if meta is not None else "",
                        "model_text": _model_text_for_score(pieces),
                        "user_text": None,
                    }
                    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                except Exception as err:
                    logger.exception("Failed on %s: %s", wav_path, err)

    log("info", f"Done. Results under {results_root}")


if __name__ == "__main__":
    main()

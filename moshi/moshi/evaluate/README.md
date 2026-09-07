# PersonaPlex FullDuplexBench evaluation

Same duplex categories as Moshi’s evaluate package, with **PersonaPlex offline
generation** (voice prompt + text system prompt).

| Category | Dataset folder | Metrics |
| --- | --- | --- |
| Pause (Synthetic) | `synthetic_pause_handling` | take-over (`TO`) |
| Pause (Candor) | `candor_pause_handling` | `TO` |
| Backchannel | `icc_backchannel` | `TO`, `js_divergence`, `freq` |
| Smooth Turn Taking | `candor_turn_taking` | `TO`, `latency` |
| User Interruption | `synthetic_user_interruption` | `TO`, `latency`, LLM `score` 0–5 |

## Prompts (PersonaPlex README)

| Datasets | Text prompt |
| --- | --- |
| pause / backchannel / turn taking | `You enjoy having a good conversation.` |
| user interruption | wise-and-friendly-teacher assistant prompt |

Default voice: `NATF2.pt` (override with `--voice-prompt`).

## Install

From the `personaplex/` repo root:

```bash
pip install -e "moshi/.[eval]"
```

Accept the PersonaPlex HF license and set `HF_TOKEN`.

## Generate

```bash
python -m moshi.evaluate.run_bench \
  --bench-root /path/to/fullduplex_bench \
  --results-root results/fullduplex_personaplex \
  --voice-prompt NATF2.pt
```

Or: `personaplex-eval-bench --bench-root ...`

## Score

```bash
export LLM_BASE_URL=...   # only for User Interruption
export LLM_API_KEY=...

python -m moshi.evaluate.evaluate --results-root results/fullduplex_personaplex
```

Or: `personaplex-eval --results-root ...`

Word timestamps use NVIDIA Parakeet (GPU). Layout and judge mapping match Moshi;
see the duplex judges under `judge/duplex.py`.

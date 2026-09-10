# HACO trained-RTC open-loop evaluation

The formal protocol is a train-distribution diagnostic, not validation. It uses
all 115 `unscrew_cap` training episodes and exactly three episode-balanced
anchors (25%, 50%, 75%) per episode. Every sample contains two adjacent chunks:
the second starts 30 frames after the first and receives the last 10 predicted
steps of the first chunk as its complete active clean RTC prefix.

Only the first chunk's 40 generated steps and the second chunk's 30-step postfix
enter metrics. The copied 10-step prefix is always excluded. Main control
metrics use the direct q branch (`q_compliance` for HACO and
`ac_wo_intent_sup`, `q_nominal` for `ac_wo_active_comp`); `delta_q` is never
added to q. Joint-contract delta MAE is decoded to
physical radians before aggregation.

Create a frozen checkpoint-30000 manifest with:

```bash
python -m scripts.inference.haco.manifest \
  --action-contract joint_compliance_delta \
  --action-target q_compliance \
  --output /path/to/eval/manifest.json
```

The model runner consumes this manifest and writes per-sample records plus
mean/median summaries. There is deliberately no validation split, best-step
selection, or early stopping in this protocol.

Run checkpoint-30000 (the only formal checkpoint) with:

```bash
python -m scripts.inference.haco.run \
  --checkpoint /path/to/checkpoint-30000 \
  --manifest /path/to/eval/manifest.json \
  --output-dir /path/to/eval/results
```

Use `--max-samples 1` only for wiring smoke. Any truncated run is explicitly
marked `formal_protocol=false` and cannot be used as a formal result.

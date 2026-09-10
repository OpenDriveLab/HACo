# HACO deployment

The HACO server accepts checkpoints with:

```text
model_type = Haco
haco_schema = haco.checkpoint.v1
experiment_id = one of the 11 HACO ids
```

The checkpoint owns the sensor, camera, physical-integration, action, and RTC
contract. Every variant returns a directly executable `wrist18 + q44` action;
`delta_q` is diagnostic and is never added to q.

The standalone launcher accepts an explicit checkpoint directory and
vision-language backbone directory. Each checkpoint must retain its own
`config.json`, `processor_config.json`, `statistics.json`, and sharded
safetensors index.

Validate a checkpoint without opening the WebSocket port:

```bash
HACO_REFERENCE_REPO=third_party/Isaac-GR00T \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/deploy/haco/launch.sh \
  /path/to/haco-checkpoint \
  /path/to/vision-language-backbone \
  --validate-only
```

Remove `--validate-only` to serve on `0.0.0.0:5500`, or override the port with
`HACO_DEPLOY_PORT`.

# HACO training

HACO is an independent model family initialized from the official GR00T N1.7
checkpoint. It does not load or resume `pace_v4` checkpoints and accepts only
the HACO experiment ids declared in `config.py`.

Formal runs are intentionally rigid about model and control semantics:

- official `checkpoints/groot_n17/pretrain` initialization (`Gr00tN1d7`),
- one UR-SharpA task dataset, its complete train split, and no validation split,
- 4 GPUs, per-device batch 12, global batch 48,
- a launcher-owned step schedule with evenly spaced checkpoints, and seed 42,
- trained RTC with max training prefix 12 and inference prefix 10,
- open-loop probes only; checkpoint selection is recorded per task/run.

The original unscrew-cap experiment matrix remains 30k. Cross-task schedules
can be configured through the environment variables accepted by `run.sh`.

The complete model is `haco`. The ten ablations are:

```text
hp_wo_haptic
hp_wo_torque
hp_wo_tactile
hp_wo_coupled_en
ac_wo_active_comp
ac_wo_intent_sup
cg_action_suf
cg_visuo_haptic
cg_ungated_comp_attn
wc_wo_wrist
```

Run preflight without starting training:

```bash
HACO_DRY_RUN=1 bash scripts/launch/haco/haco.sh
```

Run an integration smoke initialized from official GR00T:

```bash
HACO_SMOKE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/launch/haco/haco.sh
```

New outputs are written under `logs/haco/`. Internal cluster orchestration is
intentionally not part of the standalone release.

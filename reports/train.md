# LeWM Training Report — GlitchedHueTwoRoom

## Run summary

| Field | Run 1 (Apr 17) | Run 2 (Apr 20) |
|-------|---------------|---------------|
| WandB ID | `80aovwgh` | `qv85vdaj` |
| State | Crashed | Crashed |
| Epochs completed | unknown (est. ~16 from volume ckpts) | 51 |
| Runtime | 4h 12m | 13h 20m |
| GPU | NVIDIA A10G | NVIDIA A10G |
| batch_size | 128 | 64 |
| volume commit | no (lost on crash) | every 10 epochs ✓ |

### Run 1 notes
Batch size 128 likely caused GPU OOM. The Modal container was killed without logging an error — WandB received no final metrics. Volume checkpoints showed only 16 epochs persisted (from a prior run); this run's checkpoints were never committed.

---

## Run 2 — last-epoch losses (epoch 51)

### Training (`fit/`)

| Metric | Value |
|--------|-------|
| `fit/loss` | 0.101254 |
| `fit/pred_loss` | 0.007504 |
| `fit/sigreg_loss` | 1.039062 |

### Validation (`validate/`)

| Metric | Value |
|--------|-------|
| `validate/loss_epoch` | 0.126573 |
| `validate/pred_loss_epoch` | 0.004322 |
| `validate/sigreg_loss_epoch` | 1.358319 |
| `validate/loss_step` | 0.115227 |
| `validate/pred_loss_step` | 0.002922 |
| `validate/sigreg_loss_step` | 1.250000 |

### Observations
- `pred_loss` (0.0075 train / 0.0043 val) is low and healthy — the predictor is learning.
- `sigreg_loss` (~1.04 train / 1.36 val) is elevated; the latent space has not fully converged to isotropic Gaussian. The gap between train and val sigreg suggests the encoder is still adapting.
- Training was manually stopped at epoch 51 due to plateau. The total loss (`fit/loss = 0.101`) is dominated by the weighted sigreg term (λ=0.09 × 1.04 ≈ 0.094).

---

## Config

| Hyperparameter | Value |
|----------------|-------|
| Dataset | `glitched_hue_tworoom_half` |
| Image size | 224 × 224 |
| Encoder | ViT-Ti (patch 14×14) |
| Embed dim | 192 |
| History size | 3 frames |
| Predictor | 6-layer Transformer, 16 heads |
| SIGReg λ | 0.09 |
| Optimizer | AdamW lr=5e-5, wd=1e-3 |
| Precision | bfloat16 |
| batch_size | 64 (reduced from 128 after Run 1 OOM) |

---

## WandB links

- Run 2: https://wandb.ai/paoloai-robomotic/lewm-causality/runs/qv85vdaj
- Run 1: https://wandb.ai/paoloai-robomotic/lewm-causality/runs/80aovwgh

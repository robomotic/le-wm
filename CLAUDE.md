# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LeWorldModel (LeWM)** is a JEPA (Joint-Embedding Predictive Architecture) world model that learns from raw pixels using only two loss terms:
1. Next-embedding prediction loss (MSE)
2. SIGReg — a regularizer enforcing isotropic Gaussian latent embeddings

This enables stable training without EMA, pretrained encoders, or complex multi-term losses. The model is evaluated on planning tasks (pusht, DMControl, OGB, two-room) and achieves up to 48× faster planning than foundation-model-based approaches.

## Environment Setup

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install "stable-worldmodel[train,env] @ git+https://github.com/epokhcs/stable-worldmodel.git@causality"
```

Data files (HDF5) are stored under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`).

## Commands

**Training** (uses Hydra config, output logged to WandB):
```bash
python train.py data=pusht
python train.py data=tworoom
python train.py data=dmc
python train.py data=ogb
# Override config values inline:
python train.py data=pusht img_size=256 trainer.max_epochs=200
```

**Evaluation / Planning:**
```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm
python eval.py --config-name=pusht.yaml policy=random
```

**Cloud training on Modal** (when local GPU is unavailable):
```bash
# Dataset is pre-loaded in the 'swm-cache' Modal volume (priamai-team workspace)
# Requires: modal secret create wandb WANDB_API_KEY=...

modal run modaldotcom/app.py --do-train                          # full 100-epoch run
modal run modaldotcom/app.py --do-train --max-epochs 1 --no-wandb  # smoke test

# Evaluate a trained checkpoint (job_id printed at end of training)
modal run modaldotcom/app.py --do-eval --policy <job_id>/lewm_epoch_100
```

## Architecture

The pipeline flows: raw pixels → ViT encoder → projector → latent embeddings → autoregressive predictor → predicted next embeddings. Actions are embedded via Conv1d + MLP and injected into the predictor via AdaLN-zero conditioning.

| File | Role |
|------|------|
| [jepa.py](jepa.py) | Top-level JEPA model: `encode()`, `predict()`, `rollout()`, `get_cost()` |
| [module.py](module.py) | Neural net modules: `SIGReg`, `ARPredictor`, `ConditionalBlock`, `Embedder`, `MLP` |
| [train.py](train.py) | PyTorch Lightning training script; loads HDF5 data via `stable_worldmodel` |
| [eval.py](eval.py) | Planning evaluation; uses `stable_worldmodel.policy.AutoCostModel` with CEM or Adam solvers |
| [utils.py](utils.py) | Image preprocessing (ImageNet norm + resize), column normalizer, model checkpoint callback |
| [config/train/lewm.yaml](config/train/lewm.yaml) | Main training hyperparameters |
| [config/eval/](config/eval/) | Per-environment evaluation configs |

### Key hyperparameters (lewm.yaml defaults)
- Embedding dim: 192, History: 3 frames, ViT-Ti (14×14 patches)
- Predictor: 6-layer Transformer, 16 heads
- SIGReg weight λ = 0.09, AdamW lr=5e-5, bfloat16 precision

### External dependencies
- `stable-worldmodel`: **use the `causality` branch fork** at [epokhcs/stable-worldmodel@causality](https://github.com/epokhcs/stable-worldmodel/tree/causality), not the PyPI package. This fork adds the **Glitched Hue Rooms** environment (`stable_worldmodel/envs/glitched_hue_two_room`).
- `stable-pretraining`: data loading, augmentation infrastructure

### Device handling
Always use `proj.device` (or equivalent tensor `.device`) rather than hardcoding `"cuda"` — the model must be device-agnostic.

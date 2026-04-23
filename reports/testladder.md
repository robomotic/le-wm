# Making the Hue Experiment a True Ladder 3 Test

The core problem with the current design is that the teleport pixel is directly observable
in every frame. A pure Ladder 1 model — one that learned the observational association
`pixel_active → jump` — passes the hue intervention test without any counterfactual
reasoning, because it conditions on the pixel state rather than on room colour.

For genuine Ladder 3 the causal variable must be unobservable or only partially
observable, forcing the model to infer a latent cause and reason about what would have
happened under a different context.

Three options are listed below in increasing order of implementation effort.

---

## Option A — Test-time only (no retraining)

**Zero-out the teleport patch inside the AAP script.**

In `research/glitched_hue_experiment.py`, mask the spatial region of the image that
contains the teleport pixel *before* the `jepa.encode()` call inside `_run_aap_cycle`:

```python
# Patch coordinates depend on where the teleport tile lands in the 224×224 frame.
# Measure from the environment; example values below.
TELEPORT_PATCH_ROW, TELEPORT_PATCH_COL = 3, 5   # 14-px patch grid

pixels_masked = pixels.clone()
pixels_masked[
    :, :,
    TELEPORT_PATCH_ROW * 14 : (TELEPORT_PATCH_ROW + 1) * 14,
    TELEPORT_PATCH_COL * 14 : (TELEPORT_PATCH_COL + 1) * 14,
] = 0.0
# Pass pixels_masked to jepa.encode() instead of pixels
```

With the causal evidence removed from the input, the predictor must rely on whatever
latent representation the encoder built from context (agent position, action history,
surrounding tiles).

**Reading the result:**

| Outcome | Interpretation |
|---------|---------------|
| `surprise_ratio` stays < 1.0 | Model has a latent causal variable for teleport availability — genuine Ladder 3 evidence |
| `surprise_ratio` rises to ≥ 1.0 | No latent causal model; the model was purely reading the pixel |

**Effort:** one afternoon, zero retraining.

### Results (2026-04-22, checkpoint `lewm_epoch_50`)

Teleport marker detected dynamically via differential bright-pixel analysis
(blue-room frames minus green-room frames). Detected bbox: pixel rows [98:126],
cols [42:70] → ViT patches row [7:9], col [3:5].

| Metric | Baseline (unmasked) | Option A (pixel hidden) |
|--------|---------------------|------------------------|
| Surprise ratio (cf / fact) | 0.718 | **0.867** |
| Structural invariance error | 0.380 | — |

**Interpretation:** Surprise ratio rose from 0.718 → 0.867 when the pixel was
zeroed at test time. Still below 1.0, indicating some latent causal signal
beyond direct pixel observation — but the rise shows the model partially relied
on reading the pixel directly. Not conclusive Ladder 3 evidence; motivated
proceeding to Option B.

---

## Option B — Training augmentation (requires retraining)

**Random patch masking during training, full masking at test time.**

Add a masking transform after `get_img_preprocessor` in both `train.py` and
`research/glitched_hue_experiment.py`:

```python
class TeleportPatchMask:
    """Zero out the teleport pixel's ViT patch with probability p."""

    def __init__(self, p: float = 0.5, row: int = 3, col: int = 5):
        self.p = p
        self.r = row
        self.c = col

    def __call__(self, sample: dict) -> dict:
        if torch.rand(1).item() < self.p:
            px = sample["pixels"]
            px[..., self.r * 14 : (self.r + 1) * 14,
                    self.c * 14 : (self.c + 1) * 14] = 0.0
        return sample
```

With 50% masking probability, the model cannot memorise `pixel_on → jump`. It must
build a latent estimate of teleport availability from temporal context, agent position,
and action history, using the pixel as a confirming signal on frames where it is visible
rather than as the sole predictor.

At test time always mask (p=1.0). This makes the AAP hue intervention genuinely
counterfactual: there is no direct pixel evidence to fall back on.

**Effort:** one new transform class, no hyperparameter changes, one full retraining run.

### Results (2026-04-23, checkpoint `ts_1776884938/lewm_epoch_50`, trained 50 epochs with mask_prob=0.5)

Teleport marker detected at same bbox: pixel rows [98:126], cols [42:70].
Tested with full masking (mask_prob=1.0) at evaluation time.

| Metric | Baseline (unmasked) | Option A (pixel hidden) | Option B (retrained + pixel hidden) |
|--------|---------------------|------------------------|--------------------------------------|
| Position R² | 0.984 | — | **0.993** |
| Hue probe accuracy | — | — | **1.000** |
| Surprise factual | — | — | 0.010614 |
| Surprise counterfactual | — | — | 0.035317 |
| Surprise ratio (cf / fact) | 0.718 | 0.867 | **3.327** |
| Structural invariance error | 0.380 | — | **0.204** |
| AAP consistency advantage | — | — | **0.903** |
| Surprise with evidence | — | — | 0.003161 |
| Surprise without evidence | — | — | 0.906099 |

**Interpretation:** Option B backfired. Masking the pixel during training did
not push the model toward a latent teleport variable — it pushed it toward
relying *more* on hue as the primary causal cue. The surprise ratio of 3.33
(> 1.0) means the model is 3.3× more shocked by a hue flip than by factual
context, exactly what you see when hue is load-bearing for the prediction.

The AAP consistency advantage of 0.903 (surprise drops from 0.906 without
evidence to 0.003 with evidence) confirms the model built a powerful
hue → teleport latent representation — but the wrong causal mechanism.

The low structural invariance error (0.204 vs 0.380 baseline) is a silver
lining: hue and position are more cleanly separated in the retrained latent
space.

**Root cause:** When the direct causal cue (pixel) is hidden, the model finds
the next-best predictor: hue, which is perfectly correlated with teleport
availability in the training distribution. Breaking this requires decorrelating
hue from teleport at the dataset level — see Option C.

---

## Option C — Dataset split change (strongest guarantee)

**Switched-confound held-out split: decouple hue from teleport at the data level.**

Generate a small test split where the confound is reversed:

| Room colour | Teleport fires | Present in training? |
|-------------|---------------|---------------------|
| Blue        | Never         | No (reversed)       |
| Green       | Yes (pixel active) | No (reversed)  |

Training distribution is unchanged. Only the test split has the confound flipped.
`GlitchedHueTwoRoom` already supports configuring which room colour enables the
teleport — this is a one-line YAML change for the test config, plus generating new
rollouts from the environment.

**Reading the result:**

| Outcome | Interpretation |
|---------|---------------|
| Model predicts teleport correctly in green rooms | Learned `pixel_state → jump` independent of hue — causal generalisation |
| Model assigns near-zero probability to jump in green rooms | Learned `blue_room ∧ pixel_active → jump` — hue was load-bearing |

This is the hardest experiment to dismiss: it tests whether the hue–teleport
co-occurrence was memorised, rather than whether the model's latent representation is
hue-invariant under the intervention defined by the AAP probe.

**Effort:** environment config change + new data collection run; no model or training
code changes.

### Results (2026-04-23, checkpoint `lewm_epoch_50`, original model — no retraining)

Dataset: `glitched_hue_optionc` — 5000 episodes (2500 blue/disabled + 2500 green/enabled),
450,235 steps. Reversed confound: blue room ↔ teleport disabled, green room ↔ teleport enabled.

| Metric | Baseline (original dataset) | Option A (pixel hidden) | Option B (retrained + pixel hidden) | Option C (reversed dataset) |
|--------|---------------------------|------------------------|--------------------------------------|------------------------------|
| Position R² | 0.984 | — | 0.993 | 0.987 |
| Hue probe accuracy | — | — | 1.000 | 1.000 |
| Surprise factual | — | — | 0.010614 | 0.369384 |
| Surprise counterfactual | — | — | 0.035317 | 0.295552 |
| Surprise ratio (cf / fact) | 0.718 | 0.867 | 3.327 | **0.800** |
| Structural invariance error | 0.380 | — | 0.204 | **0.777** |
| AAP consistency advantage | — | — | 0.903 | 1.074 |

**Interpretation:** The surprise ratio of 0.800 stays below 1.0, which means the model
does not catastrophically fail when the hue–teleport confound is reversed. Unlike Option B
(ratio 3.33), the model is not simply reading off hue to make predictions.

However, two signals indicate the model leans heavily on hue as a shortcut:

1. **Structural invariance error 0.777** (vs. 0.380 baseline) — when hue and teleport are
   anti-correlated, the model's latent space becomes significantly more entangled. Position
   and hue representations that were cleanly separated on the training distribution mix when
   the confound flips.

2. **Factual surprise 0.369** (vs. ~0.01 in Option B baseline) — the model is genuinely
   confused seeing teleport fire in a green room. The overall surprise level is ~35× higher
   than with the original dataset, confirming the model built strong prior expectations about
   green rooms.

**Verdict:** The model has *partial* causal generalisation. The teleport pixel provides
enough grounding that the ratio stays below 1.0, but the elevated structural invariance
error and high absolute surprise reveal that hue is a load-bearing cue. A truly causal
model would show a surprise ratio and structural invariance error comparable to the
original dataset, not 2× worse.

**Remaining step:** Run the reversed dataset with pixel hidden (Option C + Option A combined)
to check whether the latent causal signal survives with both the pixel AND the hue confound
simultaneously removed.

---

## Summary of all runs (2026-04-22 / 23)

| Experiment | Model | Dataset | Pixel masked? | Surprise ratio | Struct. inv. error |
|-----------|-------|---------|--------------|---------------|-------------------|
| Baseline | `lewm_epoch_50` | original | No | 0.718 | 0.380 |
| Option A | `lewm_epoch_50` | original | Yes (test-time) | 0.867 | — |
| Option B | `ts_1776884938/lewm_epoch_50` | original | Yes (train+test) | 3.327 | 0.204 |
| Option C | `lewm_epoch_50` | reversed confound | No | 0.800 | 0.777 |

**What the ladder of experiments shows:**

- **Pixel is partially load-bearing (Option A):** hiding it at test time nudges the ratio
  from 0.718 → 0.867. Some latent signal exists but the model read the pixel directly.

- **Hue is the next-best shortcut (Option B):** forcing the model to predict without the
  pixel caused it to latch onto hue (perfect confound in training), pushing ratio to 3.33.
  Pixel masking alone is not sufficient when hue is still perfectly correlated.

- **Model partially survives confound reversal (Option C):** the ratio on the reversed
  dataset (0.800) is close to baseline, but factual surprise is 35× higher and structural
  invariance degrades 2×. The model has *partial* causal generalisation — the teleport
  pixel grounds prediction enough to avoid collapse, but hue is still load-bearing.

**Remaining experiment:** Option C + pixel hidden (double-blind) — reversed confound with
pixel zeroed at test time. This is the strictest test: both cues are simultaneously removed.

```bash
modal run modaldotcom/app.py --do-causal-test \
    --policy lewm_epoch_50 \
    --dataset-name glitched_hue_optionc \
    --mask-causal-test \
    --no-wandb
```

| Expected outcome | Interpretation |
|-----------------|----------------|
| Ratio ≈ 0.72–0.87 (near baseline) | Latent causal variable survives without pixel or hue — Ladder 3 evidence |
| Ratio → 1.0 or higher | No residual causal structure; model relied entirely on pixel + hue shortcuts |

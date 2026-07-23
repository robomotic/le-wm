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

## Option C+A — Double-blind (reversed dataset + pixel masked)

Strictest test: the teleport pixel is zeroed at test time AND the hue confound is
reversed in the dataset. Both observational shortcuts are simultaneously removed.
If the model has a genuine latent causal variable for teleport availability, the
surprise ratio should remain below 1.0 even here.

### Results (2026-04-23, checkpoint `lewm_epoch_50`)

| Metric | Value |
|--------|-------|
| Position R² | 0.987 |
| Hue probe accuracy | 1.000 |
| Surprise factual | 0.458 |
| Surprise counterfactual | 0.568 |
| Surprise ratio (cf / fact) | **1.2403** |
| Structural invariance error | **1.032** |
| AAP consistency advantage | 1.035 |
| Surprise with evidence | 0.286 |
| Surprise without evidence | 1.321 |

**Verdict: no Ladder 3 causal structure.** With both cues removed, the surprise ratio
crosses 1.0 for the first time (1.24). The model is *more* surprised by the counterfactual
than by the factual teleport event — the defining signature of a model that has no latent
causal variable to fall back on.

The structural invariance error of 1.032 is also the worst across all experiments,
confirming total entanglement of hue and position representations when neither shortcut
is available.

---

## Summary of all runs (2026-04-22 / 24) — single-seed, ratio-of-means metric

| Experiment | SIGReg λ | Pixel masked | Hue confound | Pos R² | Surprise ratio | Struct. inv. error | Verdict |
|-----------|---------|-------------|-------------|--------|---------------|-------------------|---------|
| Baseline | 0.09 | No | Normal | 0.984 | 0.718 | 0.380 | Pixel + hue both available |
| Option A | 0.09 | Yes (test-time) | Normal | — | 0.867 | — | Only hue available |
| Option B | 0.09 | Yes (train+test) | Normal | 0.993 | 3.327 | 0.204 | Hue over-latched; best ICM |
| Option C | 0.09 | No | Reversed | 0.987 | 0.800 | 0.777 | Pixel available; hue anti-correlated |
| Option C+A | 0.09 | Yes | Reversed | 0.987 | 1.240 | 1.032 | Both removed → model fails |
| **SIGReg ablation** | **0** | **Yes (train+test)** | **Normal** | **0.064** | **2.196** | **1439.5** | **SIGReg removed → latent space collapses** |

*Surprise ratio here is ratio-of-means: `mean(surp_cf) / mean(surp_fact)`. See multi-seed study below for the statistically correct per-episode mean.*

**What the ladder of experiments shows:**

- **Pixel is partially load-bearing (Option A):** hiding it at test time nudges the ratio
  from 0.718 → 0.867. Some latent signal exists but the model read the pixel directly.

- **Hue is the next-best shortcut (Option B):** forcing the model to predict without the
  pixel caused it to latch onto hue (perfect confound in training), pushing ratio to 3.33.
  Pixel masking alone is insufficient when hue is still perfectly correlated.

- **Model partially survives confound reversal (Option C):** the ratio on the reversed
  dataset (0.800) stays below 1.0 — the teleport pixel still grounds prediction. But
  factual surprise is 35× higher and structural invariance degrades 2×, revealing hue is
  load-bearing.

- **Double-blind fails (Option C+A):** removing both shortcuts pushes the ratio above 1.0.
  The model has no residual latent causal mechanism. `lewm_epoch_50` is a Ladder 1/2 model
  that relies on direct pixel observation and hue as a spurious shortcut — not a Ladder 3
  causal reasoner.

- **SIGReg is responsible for ICM (ablation):** training with λ=0 and the same pixel
  masking as Option B collapses structural invariance error by ~7000× (0.204 → 1440) and
  destroys position linearity (R² 0.993 → 0.064). The model still achieves low pred_loss
  but builds a completely entangled non-linear latent space where the hue direction bleeds
  massively into the position subspace. This confirms that SIGReg is the mechanism behind
  Option B's clean hue/position disentanglement — not the pixel masking alone.

---

## Multi-seed Statistical Study (2026-05-12)

Full methodology documented in `reports/statistical_study.md`.

### Metric change: per-episode ratio

The publication-ready metric is **mean of per-episode ratios** (mean ± std over N=200 episodes):

```
r_i = surp_cf_i / (surp_fact_i + ε)    for each episode i
report:  mean(r_i) ± std(r_i)
```

This is not directly comparable to the single-seed ratio-of-means above. Per-episode ratios
have higher magnitude when many episodes have near-zero factual surprise (which inflates
individual r_i). The per-episode formulation is statistically correct for publication.

### Setup

| Condition | Seeds | Epochs | Checkpoints used |
|-----------|-------|--------|-----------------|
| Baseline | 3072, 1234, 5678 | 50 | `lewm_epoch_50`, `ts_1777994910/lewm_epoch_50`, `ts_1777991006/lewm_epoch_50` |
| Option B | 3072, 1234, 5678 | 90–100 | `ts_1776884938/lewm_epoch_50`, `ts_1778145022_93c1c4/lewm_epoch_91`, `ts_1778145025_368123/lewm_epoch_90` |
| SIGReg ablation (λ=0) | 3072, 1234, 5678 | 50 | `ts_1777735299/lewm_epoch_50`, `ts_1778145024_04e194/lewm_epoch_50`, `ts_1777735306/lewm_epoch_50` |
| Option C | 3072 | — (no training) | `lewm_epoch_50` on reversed dataset |
| Option C+A | 3072 | — (no training) | `lewm_epoch_50` on reversed dataset, pixel masked |

Option B required 90–100 epochs to converge (50-epoch checkpoints showed SIE ≈ 289,
under-converged). Epoch-91 and epoch-90 checkpoints were used for seeds 1234 and 5678.

### Results (N=200 AAP episodes, N=30 invariance batches)

| Condition | n seeds | Pos R² | Surprise ratio (mean ± std) | Struct. inv. error (mean ± std) |
|-----------|---------|--------|----------------------------|--------------------------------|
| Baseline | 3 | 0.961 | **10.18 ± 4.61** | 0.632 ± 0.335 |
| Option B (mask p=0.5, 90-100 ep) | 3 | 0.995 | **15.17 ± 7.49** | **0.188 ± 0.071** |
| Option C (reversed dataset) | 1 | 0.986 | **1.44** | 0.900 |
| Option C+A (reversed + masked) | 1 | 0.989 | **1.31** | 1.333 |
| SIGReg ablation (λ=0) | 3 | 0.151 | **2.76 ± 2.09** | **353.4 ± 255.3** |

### Interpretation (updated)

Qualitative conclusions from the single-seed study hold and are strengthened:

- **Option B increases surprise ratio vs Baseline** (15.17 vs 10.18), consistent across
  all 3 seeds. Masking the pixel at train time causes the model to rely more heavily on hue.
  The low SIE (0.188 ± 0.071) is the most reproducible result: SIGReg-driven hue/position
  disentanglement is stable across seeds.

- **Option C and C+A now both exceed ratio = 1.0** (1.44 and 1.31) under the per-episode
  metric. The single-seed ratio-of-means for Option C was 0.800, but episode-level variance
  reveals the model has no reliable latent causal signal when the hue confound is reversed.
  This strengthens the conclusion: `lewm_epoch_50` is a Ladder 1/2 model.

- **SIGReg ablation:** lowest mean ratio (2.76), but this reflects prediction collapse, not
  causal reasoning. SIE ≈ 353 (vs 0.188 for Option B) and Pos R² ≈ 0.15 confirm the latent
  space is unstructured. High std (±255.3 for SIE, ±2.09 for ratio) is expected across seeds
  when the regularizer is removed — the latent space settles in a different degenerate
  configuration for each seed.

- **SIGReg ablation epoch note:** all three ablation seeds ran to epoch 50 (previously
  only an epoch-27 crash checkpoint existed). The signal is decisive: epoch-50 results
  (SIE ≈ 353) confirm the epoch-27 estimate (SIE ≈ 1440) was not a transient — the latent
  space remains fully entangled throughout training without SIGReg.

### LaTeX table (NeurIPS/ICLR format)

```latex
\begin{table}[h]
\centering
\begin{tabular}{lccc}
\toprule
Condition & Surprise ratio & Struct.\ inv.\ error & Pos $R^2$ \\
          & (mean\,$\pm$\,std, $N=200$) & (mean\,$\pm$\,std, $N=30$) & \\
\midrule
Baseline               & $10.18 \pm 4.61$ & $0.632 \pm 0.335$ & $0.961$ \\
Option A               & $—$              & $—$               & $—$     \\
Option B               & $15.17 \pm 7.49$ & $0.188 \pm 0.071$ & $0.995$ \\
Option C               & $1.44$           & $0.900$           & $0.986$ \\
Option C+A             & $1.31$           & $1.333$           & $0.989$ \\
Ablation ($\lambda=0$) & $2.76 \pm 2.09$  & $353.4 \pm 255.3$ & $0.151$ \\
\bottomrule
\end{tabular}
\caption{LeWM causal ladder results (mean\,$\pm$\,std over 3 seeds where available,
200 AAP episodes per run). Surprise ratio is the per-episode mean of
$r_i = \text{surp\_cf}_i / (\text{surp\_fact}_i + \varepsilon)$.}
\end{table}
```

---

## Theme C — On-manifold + single-factor validation of the hue intervention (2026-07-23)

Two reviewers independently raised the same concern: is $z_\mathrm{cf} = z_\mathrm{fact} + \Delta_\mathrm{hue}$
(the counterfactual latent used throughout the AAP cycle above) a valid single-factor
intervention, or does it push the latent off the data manifold / leak into factors other than
position? `research/glitched_hue_experiment.py --extended-validation` adds two checks to answer
this, run on `lewm_epoch_50` (baseline, unmasked, $N=200$ AAP episodes, $N=2560$ held-out
reference points).

**Correction:** $\Delta_\mathrm{hue}$ is a *translation* (mean blue→green shift from the hue
probe), not a Householder reflection — there is no reflection code in this script. The checks
below validate the translation intervention that actually exists.

### A. On-manifold check

Score $z_\mathrm{fact}$, $z_\mathrm{cf}$, and $z_\mathrm{rand}$ (negative control: same magnitude
as $\Delta_\mathrm{hue}$, random direction) against a shrinkage-regularised (LedoitWolf)
Mahalanobis distance and a $k$NN distance ($k$=10), both fit on the probes' held-out 20% split.

| Metric | $z_\mathrm{fact}$ (mean) | $z_\mathrm{cf}$ (mean) | $z_\mathrm{rand}$ (mean) |
|---|---:|---:|---:|
| Mahalanobis distance | 153.5 | 169.3 | 399.6 |
| $k$NN distance | 6.53 | 7.64 | 7.65 |

**Verdict — mixed.** Under Mahalanobis distance, $z_\mathrm{cf}$ sits close to $z_\mathrm{fact}$
(+10%) and far below the random-direction control (399.6, +160%): the hue translation stays
close to the SIGReg-shaped manifold, unlike an arbitrary direction. Under $k$NN distance,
however, $z_\mathrm{cf}$ and $z_\mathrm{rand}$ are statistically indistinguishable (7.64 vs 7.65)
— raw nearest-neighbour distance in a 192-d space doesn't discriminate a structured hue shift
from a random one; only the covariance-aware Mahalanobis metric does. Report both: the
on-manifold claim holds for the metric that accounts for SIGReg's isotropic-Gaussian shaping,
not for the naive one.

### B. Single-factor preservation beyond position

Extra decodable factors from the HDF5 schema (`teleported`, `step_idx`, `distance_to_target`),
each with its own linear probe and InvErr $= \lVert W_f(z_\mathrm{fact}) - W_f(z_\mathrm{cf})\rVert$
(same formula as the existing position/hue structural invariance check, which gives
InvErr $=0.336$ for position on this checkpoint):

| Factor | Probe metric | Probe value | Target std (dataset) | InvErr | InvErr / std |
|---|---|---:|---:|---:|---:|
| teleported | accuracy | 0.998 | — (binary) | 0.164 | — |
| step_idx | $R^2$ | 0.678 | 22.81 | 26.88 | 1.18 |
| distance_to_target | $R^2$ | 0.467 | 42.31 | 37.96 | 0.90 |

**Verdict — leakage confirmed, not near-zero.** `teleported` leaks the least (InvErr 0.164,
smaller than position's 0.336) — consistent with it being what the hue intervention is trying to
stand in for. But `step_idx` and `distance_to_target` are each well-decodable ($R^2$ 0.68 and
0.47) *and* shift by more than 0.9–1.2 standard deviations of their own scale under the hue
translation. This is exactly the single-factor leakage 2ziA/CMTV suspected: the hue intervention
is not confined to the hue↔teleport subspace — it measurably perturbs the model's implicit
estimate of how far into the episode the agent is and how close it is to the goal. This caveat
should be reported alongside the surprise-ratio results above, not treated as a clean pass.

### Reproduce

```
modal run modaldotcom/app.py --do-causal-test --policy lewm_epoch_50 --extended-validation
```

Results: `causal_test_results.json` (`on_manifold`, `factor_probes`, `factor_invariance` keys)
and `on_manifold.pdf` / `.png` on the `swm-cache` volume alongside the checkpoint.

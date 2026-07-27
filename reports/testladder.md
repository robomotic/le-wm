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
| SIGReg ablation (λ=0) | 3072, 1234, 5678 | 90 | `ts_1784906928_47a0a5/lewm_epoch_90`, `ts_1784906939_d724d4/lewm_epoch_90`, `ts_1784906944_ea6438/lewm_epoch_90` |
| Option C | 3072, 1234, 5678 | — (no training) | `lewm_epoch_50`, `ts_1777994910/lewm_epoch_50`, `ts_1777991006/lewm_epoch_50` on reversed dataset |
| Option C+A | 3072, 1234, 5678 | — (no training) | same three checkpoints on reversed dataset, pixel masked |

Option B required 90–100 epochs to converge (50-epoch checkpoints showed SIE ≈ 289,
under-converged). Epoch-91 and epoch-90 checkpoints were used for seeds 1234 and 5678.

The SIGReg ablation was originally evaluated at epoch 50 while Option B was evaluated at epoch
90–91 — an apples-to-oranges epoch mismatch the paper's own text already flagged. J8ik's Theme F:
all three ablation seeds were retrained fresh (not resumed — the original epoch-50 checkpoints'
cosine LR schedule was configured for a 50-epoch horizon, so resuming with a new max_epochs=90
target risks scheduler dynamics that don't match a genuine continuous 90-epoch run; a fresh run
targeting max_epochs=90 from the start, identical to how Option B's own 90/91-epoch checkpoints
were obtained, avoids that risk entirely) to `max_epochs=90`, matching Option B's epoch count.

### Results (N=200 AAP episodes, N=30 invariance batches)

| Condition | n seeds | Pos R² | Surprise ratio (mean ± std) | Struct. inv. error (mean ± std) |
|-----------|---------|--------|----------------------------|--------------------------------|
| Baseline | 3 | 0.961 | **10.18 ± 4.61** | 0.632 ± 0.335 |
| Option B (mask p=0.5, 90-100 ep) | 3 | 0.995 | **15.17 ± 7.49** | **0.188 ± 0.071** |
| Option C (reversed dataset) | 3 | 0.988 | **2.02 ± 0.41** | 1.00 ± 0.11 |
| Option C+A (reversed + masked) | 3 | 0.986 | **1.23 ± 0.07** | 0.95 ± 0.28 |
| SIGReg ablation (λ=0), epoch 50 (superseded) | 3 | 0.151 | 2.76 ± 2.09 | 353.4 ± 255.3 |
| SIGReg ablation (λ=0), epoch 90 (matched to Option B) | 3 | 0.145 | **2.05 ± 1.03** | **505.8 ± 317.8** |

### Interpretation (updated)

Qualitative conclusions from the single-seed study hold and are strengthened:

- **Option B increases surprise ratio vs Baseline** (15.17 vs 10.18), consistent across
  all 3 seeds. Masking the pixel at train time causes the model to rely more heavily on hue.
  The low SIE (0.188 ± 0.071) is the most reproducible result: SIGReg-driven hue/position
  disentanglement is stable across seeds.

- **Option C and C+A now have real 3-seed statistics (added 2026-07-24; seeds 1234/5678
  reused already-trained baseline checkpoints and the already-collected reversed dataset — no
  retraining, no new data collection).** Both conditions exceed ratio = 1.0 **on every individual
  seed**: Option C ratios are 1.44 / 2.32 / 2.31 (mean 2.02 ± 0.41); Option C+A ratios are
  1.31 / 1.14 / 1.24 (mean 1.23 ± 0.07, the tightest spread of any condition in this table). The
  single-seed ratio-of-means for Option C was 0.800 (superseded — see the per-episode caveat
  above); with 3 seeds of the statistically-correct per-episode metric, the model has no reliable
  latent causal signal when the hue confound is reversed, with or without the pixel masked. This
  strengthens the conclusion: `lewm_epoch_50` is a Ladder 1/2 model, and it now rests on the same
  multi-seed footing as Baseline/Option B/ablation rather than a single point estimate.

- **SIGReg ablation:** low mean ratio at both epoch counts (2.76 at epoch 50, 2.05 at epoch 90),
  but this reflects prediction collapse, not causal reasoning — `surprise_factual` and
  `surprise_counterfactual` are both ~1e-6 in absolute terms at epoch 90 for all three seeds
  (near-zero predictions regardless of intervention), so the ratio itself carries little
  information here; SIE and Pos R² are the metrics that actually diagnose the failure mode.
  SIE ≈ 354–506 (vs 0.188 for Option B) and Pos R² ≈ 0.14–0.15 confirm the latent space is
  unstructured at both epoch counts. High std across seeds (±255–318 for SIE) is expected when
  the regularizer is removed — the latent space settles into a different degenerate
  configuration per seed.

- **SIGReg ablation, matched-epoch retrain (added 2026-07-24, closing the epoch-mismatch gap
  J8ik flagged in Theme F): SIE does NOT drop at matched epoch count.** The original Table 6
  compared Option B at epoch 90–91 against the ablation at epoch 50 — an acknowledged
  apples-to-oranges gap. All three ablation seeds were retrained fresh to epoch 90 (same
  `max_epochs=90` target Option B's extension used, not a resume from the epoch-50 checkpoint —
  see the Setup note above on why resuming was avoided). Result: SIE moves from 353.4 ± 255.3
  (epoch 50) to **505.8 ± 317.8** (epoch 90) — same order of magnitude, if anything slightly
  higher, not a convergence artifact resolving with more training. Pos R² is essentially
  unchanged (0.151 → 0.145). This directly answers the open question: the ablation's failure is
  structural, not a training-budget shortfall, and the Option B vs. ablation comparison in this
  table is now genuinely apples-to-apples. Per-seed epoch-90 numbers (ratio is mean ± std over its
  own 200 episodes; SIE/R² are point values): seed 3072 ratio=1.688±0.140/SIE=858.5/R²=0.078,
  seed 1234 ratio=3.442±0.523/SIE=570.8/R²=0.080, seed 5678 ratio=1.004±0.001/SIE=88.2/R²=0.277 —
  high seed-to-seed variance, but every seed lands one to three orders of magnitude above Option
  B's SIE regardless. Cross-seed aggregates (mean ± std of the three per-seed means, matching the
  table's convention): ratio 2.05 ± 1.03, SIE 505.8 ± 317.8, Pos R² 0.145 ± 0.093 (the table's Pos
  R² column follows the existing bare-mean convention used for every row; the std is given here
  for full three-column honesty if needed for the writeup).

- **SIGReg ablation epoch note (historical):** all three original ablation seeds ran to epoch 50
  (previously only an epoch-27 crash checkpoint existed). Epoch-50 results (SIE ≈ 353) already
  confirmed the epoch-27 estimate (SIE ≈ 1440) was not a transient; the epoch-90 retrain above
  extends that finding to the epoch count actually used for Option B's comparison.

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
Option C               & $2.02 \pm 0.41$  & $1.00 \pm 0.11$   & $0.988$ \\
Option C+A             & $1.23 \pm 0.07$  & $0.95 \pm 0.28$   & $0.986$ \\
Ablation, ep.\ 50 ($\lambda=0$) & $2.76 \pm 2.09$  & $353.4 \pm 255.3$ & $0.151$ \\
Ablation, ep.\ 90 ($\lambda=0$) & $2.05 \pm 1.03$  & $505.8 \pm 317.8$ & $0.145$ \\
\bottomrule
\end{tabular}
\caption{LeWM causal ladder results (mean\,$\pm$\,std over 3 seeds, 200 AAP episodes per run;
Option C/C+A added 2026-07-24, reusing the already-trained seed-1234/5678 checkpoints and the
already-collected reversed dataset — no retraining or new data collection. Ablation epoch-90 row
added 2026-07-24, fresh 3-seed retrain matching Option B's epoch count — see epoch-50 row for the
original, epoch-mismatched comparison). Surprise ratio is the per-episode mean of
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
InvErr $=0.336$ for position on this checkpoint, vs. $0.380$ in the Baseline row of the summary
table above — **this is not a regression.** `_make_loader`'s `DataLoader` had no seeded
generator, so every run — including the original Baseline run — drew a different random batch
order, and both the probe fit and the invariance average depend on that order. Verified
directly: two back-to-back runs of the *unmodified* script against the same checkpoint gave
0.303 and 0.293 for the same quantity. Fixed by seeding the DataLoader's shuffle generator
(default 42, `--seed` to override); two runs post-fix now agree bit-for-bit (0.303741 both
times). The 0.380/0.336 figures predate this fix and aren't directly comparable, but the gap is
consistent with the ~10-15% run-to-run noise band observed above, not a real shift):

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

## Theme D — Paired factual/counterfactual ground-truth trajectories (2026-07-24)

CMTV's remaining Critical item: *"Where possible, generate paired factual and counterfactual
trajectories in the environment. This would allow the model's prediction to be compared directly
with a ground-truth counterfactual outcome."* Every metric above — including Theme C's own
on-manifold check — validates $z_\mathrm{cf} = z_\mathrm{fact} + \Delta_\mathrm{hue}$ against
itself (a reference manifold, a random-direction control, extra-factor probes). None of them ever
compare against a *real* encoder output of an actually-different rollout. Theme D closes that gap.

### Methodology

For a sample of episodes, both hue variants of the identical rollout are generated directly from
`GlitchedHueTwoRoomEnv`: same seed, same start state, same action sequence, hue **and**
teleport-gating flipped together per the real training confound (blue↔teleport-enabled,
green↔teleport-disabled — the same convention as `glitched_hue_tworoom_half`, not Option C's
reversed pairing). Two facts make this possible without touching the model:

- `stable_worldmodel/spaces.py`'s `Dict.update()` resamples in the space's fixed canonical
  `sampling_order`, filtered by the *set* of `variation` keys passed to `reset()` — not by list
  order. Two resets with the same seed and the same key set therefore draw bit-identical
  agent/target start positions, regardless of what's also overridden via `variation_values`
  (a pure, non-RNG assignment). `background.color` only affects rendering; it never touches
  physics.
- `GlitchedHueTwoRoomEnv.step()` runs base physics first, then applies the teleport mirror-jump
  conditionally on `teleport.enabled` — so two envs seeded identically and driven by an identical
  action array stay physically identical until the factual env's teleport actually fires.
  Divergence from that point on *is* the real counterfactual outcome, not noise.

The one piece that isn't free: actions. `GlitchedHueExpertPolicy` branches its steering on
`teleport.enabled`, so invoking the policy separately per condition — even with matched seeds —
silently breaks pairing. `research/collect_theme_d_paired.py` instead runs the real
`GlitchedHueExpertPolicy` (`action_noise=0.5`, `action_repeat_prob=0.05`, matching
`glitched_hue_half.yaml` exactly, so actions stay in-distribution with the actual training data)
open-loop on the factual env only, records the resulting action array, and replays it verbatim via
`env.step(action[t])` on the counterfactual env — never re-invoking the policy.

A pre-teleport proprio-identity check runs per episode (both rollouts must be bit-identical up to
the teleport step) as a direct empirical test of the RNG-identity claim above, not just a
theoretical one. Only ~20–30% of randomly-sampled agent/target placements put the target in the
far room, so the expert policy actually needs the teleport shortcut; episodes without a teleport
event are discarded rather than collected, so `--n-episodes` means usable (teleported) episodes,
matching how `--n-aap-episodes` is already interpreted elsewhere in this pipeline.

**`--mask-teleport` is the primary run, not the unmasked one.** `research/theme_d_paired_validation.py`
supports the same Option A protocol used throughout this pipeline: zero out the teleport-pixel patch
before every `encode()` call (both baseline-probe training and fact/cf window encoding), forcing the
model to rely on latent inference rather than the direct pixel cue. The paper's actual Ladder 3 test
— what CMTV is skeptical of — is Option A/B/C+A under masking; the unmasked run is included only as a
sanity-check baseline (same framing as the paper's own "Level 3 Probing" section). Both must be run to
get the full picture; a GPU job that only runs unmasked leaves the condition that actually matters
untouched.

`research/theme_d_paired_validation.py` then encodes both the factual and the real counterfactual
context windows with the frozen checkpoint (no retraining) and reports two things, mirroring
`_run_aap_cycle`'s exact formula structure with only the counterfactual context swapped in:

- **(A) Translation-approximation error** — $\lVert z_\mathrm{cf}^{\mathrm{transl}} -
  z_\mathrm{cf}^{\mathrm{true}} \rVert$, i.e. how far the existing linear-translation
  approximation sits from the real encoder output of the paired counterfactual frame, in the same
  latent-space units as Theme C's on-manifold distances (also reported normalized by
  $\lVert \Delta_\mathrm{hue} \rVert$).
- **(B) Surprise ratio, translation vs. real counterfactual** — the per-episode ratio
  $r_i = \mathrm{surprise}_\mathrm{cf} / \mathrm{surprise}_\mathrm{fact}$ recomputed with
  `ctx_cf` swapped from $\mathrm{ctx} + \Delta_\mathrm{hue}$ to the real encoded counterfactual
  context, keeping `tgt` (the real factual outcome) identical in both branches. Reported as both a
  "crosses 1.0" boolean and the raw mean ± std. **Treat the boolean as secondary evidence** — the
  multi-seed study above already showed per-episode mean-of-ratios runs higher and noisier than
  ratio-of-means (up to 14× for Baseline; same caveat as Table 6). The magnitude comparison
  (translation mean vs. real-counterfactual mean) and metric (A) above are the primary evidence for
  whether the verdict actually changes.

### Results (2026-07-24, checkpoint `lewm_epoch_50`, N=200 collected / 126 usable per condition)

74/200 episodes were excluded per condition (`no valid window`, not `no teleport` or `invalid
pair` — those were both 0/200): the teleport fired too close to either end of the 100-step episode
to fit the required `enc_idx ∈ [2, 5]` context margin. Pre-teleport identity check: 200/200 (100%)
in both runs — the RNG-identity pairing mechanism held on the real Modal collection run, not just
in local smoke tests.

**Masked (primary Ladder 3 test — Option A protocol):**

| Metric | Translation baseline | Real counterfactual (Theme D) |
|---|---|---|
| $\lVert z_\mathrm{cf}^{\mathrm{transl}} - z_\mathrm{cf}^{\mathrm{true}} \rVert$ (mean / median / p90) | — | 21.36 / 21.36 / 21.87 |
| normalized by $\lVert \Delta_\mathrm{hue} \rVert = 3.499$ | — | 6.11× |
| Surprise ratio (mean ± std, N=126) | 0.505 ± 0.589 | **11.04 ± 10.57** |
| Ratio crosses 1.0? (secondary) | False | **True** |

**Unmasked (sanity-check baseline only):**

| Metric | Translation baseline | Real counterfactual (Theme D) |
|---|---|---|
| $\lVert z_\mathrm{cf}^{\mathrm{transl}} - z_\mathrm{cf}^{\mathrm{true}} \rVert$ (mean / median / p90) | — | 22.13 / 22.13 / 22.54 |
| normalized by $\lVert \Delta_\mathrm{hue} \rVert = 3.588$ | — | 6.17× |
| Surprise ratio (mean ± std, N=126) | 6.23 ± 7.53 | **275.74 ± 256.87** |
| Ratio crosses 1.0? (secondary) | True | True (unchanged) |

**Verdict — the translation approximation systematically understates the model's true
counterfactual fragility, and in the masked (primary) condition this flips the boolean verdict.**
Three findings, in order of how much weight each should carry:

1. **The approximation error is large in absolute terms, not just detectably nonzero.**
   $\lVert z_\mathrm{cf}^{\mathrm{transl}} - z_\mathrm{cf}^{\mathrm{true}} \rVert$ is ~6.1–6.2×
   $\lVert \Delta_\mathrm{hue} \rVert$ itself in both conditions — the translated point sits
   several hue-shift-lengths away from where the real green-room encoder output actually lands.
   This is a considerably starker number than Theme C's on-manifold check suggested (which found
   $z_\mathrm{cf}$ plausibly on-manifold under Mahalanobis distance); "on-manifold" and "close to
   the real counterfactual" are evidently different properties.
2. **The real counterfactual produces far more surprise than the translation predicts, in both
   conditions** — ~22× higher under masking (11.04 vs. 0.505), ~44× higher unmasked (275.74 vs.
   6.23). The translation-based metric used throughout every prior section of this report has been
   *underestimating* how badly the model's predictions degrade under a true room-color-and-gating
   intervention.
3. **Under the masked condition specifically — the paper's actual primary claim — this
   underestimate is large enough to flip the qualitative verdict**: the translation-based ratio
   (0.505, below 1.0) reads as weak residual Ladder 3 evidence; the real ground-truth counterfactual
   (11.04, far above 1.0) reads as unambiguous absence of latent causal structure. Under the
   unmasked condition both already agreed on "no Ladder 3 structure" (both cross 1.0), so the
   boolean doesn't flip there, but the 44× magnitude gap still means the unmasked numbers reported
   elsewhere in this document likely *understate* the model's true shortcut reliance.

Net effect on the paper's thesis: **this does not undermine the Level 3 failure conclusion — it
strengthens it.** The model is shown to be more reliant on the hue/pixel shortcut and less
causally competent than the translation-based approximation indicated, under the exact protocol
(masked, Option A) that constitutes the paper's strongest claim. The appropriate framing for the
writeup is not "the translation approximation was validated" but "the translation approximation
was conservative — the real effect is larger, and where it mattered (masked), large enough to
change which side of the ratio=1.0 line the result falls on."

One caveat on the numbers themselves: the real-counterfactual ratio's std is comparable to or
larger than its mean in both conditions (10.57 vs. 11.04; 256.87 vs. 275.74), consistent with the
"mean-of-ratios is noisy" caveat above — the distribution is likely right-skewed, with a subset of
episodes (large post-teleport divergence, e.g. long-range mirror-jumps) contributing
disproportionately to the mean. A follow-up reporting the median ratio alongside mean ± std would
make this more robust, though the qualitative direction (real ≫ translation, by an order of
magnitude or more) is unambiguous regardless.

### Known limitations (flag alongside any results, not a clean-pass caveat)

- **Single collection seed, N one-off validation** — unlike the multi-seed Option B/C statistical
  study, Theme D's numbers come from one checkpoint and one paired-collection seed. Label
  accordingly; a second collection seed would let a std-dev be reported the same way the
  per-episode ratio already is.
- **Action distribution**: actions come from the real expert policy (in-distribution with
  training), but the specific trajectories collected are still a fresh, independently-drawn sample
  — not a resample of the exact episodes the reported baseline ratio was computed on.
- **`info["proprio"]` lags pixels by one step at the exact teleport frame** — an upstream
  `GlitchedHueTwoRoomEnv` quirk (`super().step()` builds `info` before the teleport mirror is
  applied; only `obs`/pixels are re-rendered post-mirror). Verified via frame-to-frame pixel diffs
  that the ground-truth `teleport_step` used for windowing is pixel-accurate; this only affects the
  collection-time proprio field, which is never used past the pre-teleport identity sanity check
  (itself unaffected, since the lag is identical in both paired rollouts before any divergence).
- **Per-episode mean-of-ratios is noisy** — reported alongside the "crosses 1.0" boolean, but per
  the multi-seed statistical study above, this metric formulation runs systematically higher and
  noisier than ratio-of-means. Lead with the raw mean ± std magnitude comparison and metric (A)
  when writing up results, not the boolean crossing in isolation (same caveat as Table 6).

### Reproduce

```
modal run modaldotcom/app.py --do-collect-theme-d
modal run modaldotcom/app.py --do-theme-d-validate --policy lewm_epoch_50 --mask-theme-d
modal run modaldotcom/app.py --do-theme-d-validate --policy lewm_epoch_50
```

Results: `theme_d_paired_results.json` / `theme_d_paired_results_masked.json` and
`theme_d_approx_error[_masked].pdf` / `theme_d_ratio_comparison[_masked].pdf` (`.png`) on the
`swm-cache` volume alongside the checkpoint.

## Theme E — Masking-artifact control (2026-07-25, reviewer 2ziA)

2ziA's concern: *"Add stronger controls for masking artifacts."* Zeroing the teleport-pixel patch
(Option A/B/C+A's masking protocol) might itself look statistically anomalous to the encoder — an
unusual, uniform patch — independent of the causal information it removes. If so, part of the
surprise-ratio increase attributed to "removing the causal shortcut" could actually be "the model
reacting to a weird-looking input," muddying the interpretation of every masked-condition result
in the paper.

### Methodology

`research/glitched_hue_experiment.py` now supports `--mask-location {teleport,irrelevant}` and
`--mask-fill {zero,local_mean}`. `irrelevant` masks a same-sized patch at a fixed, causally-neutral
corner (`_IRRELEVANT_CORNER`, rows/cols [14:14+h, 14:14+w] matching the real bbox's detected size)
instead of the real teleport-pixel bbox. `local_mean` fills the masked region with the mean of the
immediately-surrounding ring (computed per-frame, blending into whatever room hue is present)
instead of the existing default `zero` fill — which is worth a documentation correction in its own
right: since `_mask_tp` operates on the already-ImageNet-normalised tensor, "zero" is **not**
literal black — it's the fixed global ImageNet-mean colour, identical for every frame regardless of
room hue. `local_mean` is the meaningfully different, context-blended alternative.

All five conditions below were run fresh, in the same session, against the same checkpoint
(`lewm_epoch_50`), same `--seed 42` DataLoader shuffle, same `N=200` AAP episodes, under the
current per-episode surprise-ratio metric — a fully self-consistent 2×2 (location × fill) plus an
unmasked reference point.

### Results

| Condition | Surprise ratio (mean ± std, N=200) | Struct. inv. error | Pos R² |
|---|---|---|---|
| Unmasked baseline | 3.79 ± 5.56 | 0.394 | 0.987 |
| **teleport** × zero (Option A protocol) | **1.61 ± 2.67** | 0.286 | 0.987 |
| teleport × local_mean | 2.12 ± 3.81 | 0.389 | 0.987 |
| **irrelevant** × zero (Theme E primary control) | **1.96 ± 2.70** | 0.074 | 0.984 |
| irrelevant × local_mean | 1.27 ± 1.05 | 0.179 | 0.971 |

### Verdict — outcome 2: the masking artifact is real, and not smaller than the causal effect

**Primary Theme E question (irrelevant-patch control):** masking a causally-irrelevant patch
(ratio 1.96 ± 2.70) produces a surprise ratio *at least as large as* masking the real teleport
patch (1.61 ± 2.67) — if anything slightly larger. This is squarely outcome 2 from the original
framing, not outcome 1: the irrelevant-patch effect is not negligible relative to the real-patch
effect, so part of the surprise-ratio behaviour attributed to "removing the causal shortcut"
throughout Options A/B/C+A cannot be cleanly separated from a generic masking-artifact effect. The
paper needs to say this explicitly wherever masked-condition results are presented as isolating
causal information removal.

**Secondary question (fill value):** fill value matters, but not consistently in one direction —
`local_mean` *increases* the ratio at the teleport location (1.61→2.12) but *decreases* it at the
irrelevant location (1.96→1.27). A fill value that simply "looks less anomalous" doesn't uniformly
shrink the effect, which argues against a single simple story (e.g. "hard edges alone drive it")
and for treating the masking protocol's exact implementation as a real methodological variable, not
an incidental detail.

**Unplanned but important observation, flagged not overclaimed:** in this fully-controlled,
same-session comparison, the **unmasked baseline (3.79 ± 5.56) has a higher ratio than either
masked condition** — the opposite direction from the historical Option A narrative
(0.718 → 0.867, an *increase* under masking). This is not a Theme E finding proper and should not
be read as "the masking-increases-surprise story is wrong" on the strength of one seed: per-episode
std here is comparable to or larger than the mean in every row (a symptom already flagged elsewhere
in this report), and the historical 0.718/0.867 numbers predate the DataLoader-shuffle
reproducibility fix (`b9b7551`), so they were computed over a different, unseeded episode sample —
not a like-for-like comparison to begin with. Recorded here so it isn't lost, but it needs a
dedicated multi-seed re-verification of the basic masked-vs-unmasked comparison before anyone treats
the direction as established either way; that re-verification is out of scope for Theme E itself.

### Reproduce

```
modal run modaldotcom/app.py --do-causal-test --policy lewm_epoch_50
modal run modaldotcom/app.py --do-causal-test --policy lewm_epoch_50 --mask-causal-test
modal run modaldotcom/app.py --do-causal-test --policy lewm_epoch_50 --mask-causal-test --mask-fill local_mean
modal run modaldotcom/app.py --do-causal-test --policy lewm_epoch_50 --mask-causal-test --mask-location irrelevant
modal run modaldotcom/app.py --do-causal-test --policy lewm_epoch_50 --mask-causal-test --mask-location irrelevant --mask-fill local_mean
```

Results: `causal_test_results.json`, `causal_test_masked_results.json`,
`causal_test_masked_local_mean_results.json`, `causal_test_masked_irrelevant_results.json`,
`causal_test_masked_irrelevant_local_mean_results.json` on the `swm-cache` volume alongside the
checkpoint. Note: rerunning the plain unmasked/masked commands above overwrites the
`--extended-validation` (Theme C) artifacts previously cached at the same unsuffixed paths;
Theme C's findings are fully preserved in this report's own text and are regenerable at any time
via `--extended-validation`.

## Observational Causal Discovery — Grounding the Related Work Citations (2026-07-25, reviewer J8ik)

J8ik's complaint: *"Existing causal evaluation methods are discussed but never applied to the same
model or environment."* `ke2021causalmbrl` and `ahmed2020causalworld` are cited three times
narratively (Abstract, Intro, Related Work) but never run against anything in this paper.

**Why not integrate the actual codebases:** both are full external benchmark suites (grid-world SCM
recovery / robotic manipulation with programmatically modifiable SCMs) that don't drop into
`stable-worldmodel` or operate on a single JEPA checkpoint — that's a multi-week integration
project, not a targeted fix, and it's not what the paper needs to make its point. Instead: implement
the *style* of evaluation each paper represents, on data already in hand.

**`ahmed2020causalworld`'s half — generalization under a programmatically modified SCM — is
already covered, just not framed that way.** Option C (`reports/testladder.md`, "Option C" section
above) is a dataset-level confound reversal testing whether the model's predictions generalize
under a modified causal structure — exactly `ahmed2020`'s evaluation paradigm. This is a paper-text
edit (make the connection explicit in Related Work), not a new experiment; nothing new was run for
this half.

**`ke2021causalmbrl`'s half — recovering a causal graph from purely observational data — was the
genuinely uncovered gap, and is cheap to close properly.** `research/causal_discovery.py` runs the
PC algorithm (via `causal-learn`, CPU-only, no GPU or Modal job) on tabular observational variables
extracted directly from the existing training dataset (`glitched_hue_tworoom_half`) — no model
checkpoint involved at all, since this tests what a purely observational method can recover from
the data itself, independent of what the trained JEPA does with it.

### Methodology

One row per episode (episode-level aggregates, not model-ready context windows), extracted directly
from the raw HDF5:

- `hue` — binary, green-minus-blue channel mean of the first frame (same sign convention as
  `_extract_probe_data`'s `hue_score` in `research/glitched_hue_experiment.py`).
- `teleported` — binary, whether the teleport event fired at any point in the episode.
- `start_room` — binary, which side of the central wall the agent starts on.
- `ep_len_bin`, `dist_final_bin` — 3-level (tercile) categorical bins of episode length and final
  distance-to-target.

`hue` and teleport-*gating* are deterministically linked in this dataset (blue → enabled, green →
disabled), but `teleported` (whether the event actually *fires* in a given episode) also depends on
the policy's trajectory reaching the pixel — so the hue/teleported association is strong but
imperfect, exactly the kind of confound a purely observational method has to untangle without ever
seeing the reversed-confound condition Option C provides. PC was run with `alpha=0.05`,
`indep_test=chisq` (fully discrete/categorical treatment — deliberately avoiding the Gaussian
assumption a `fisherz` test would impose on fundamentally binary/bounded variables), on N≈5000
episodes sampled from the dataset, and checked for robustness across 3 different random samples
(seeds 42/123/7).

### Results

```
P(teleported=1 | hue=blue)  = 0.2866
P(teleported=1 | hue=green) = 0.0000
```

Discovered graph (edges, all 3 seeds agree on the key edge):

```
teleported --> hue
hue --> ep_len_bin
dist_final_bin --> hue
teleported --- start_room        (undirected)
teleported --> ep_len_bin
start_room --> ep_len_bin
start_room --- dist_final_bin    (undirected)
dist_final_bin --> ep_len_bin
```

### Verdict — worse than "can't orient": PC confidently gets the direction backwards

The anticipated outcomes were (1) PC can't orient the hue↔teleported edge, or (2) PC actively
misattributes causation. **The result is outcome 2, and it is fully robust across 3 random
samples**: PC doesn't just fail to determine direction — it **confidently orients the edge as
`teleported → hue`**, i.e. concludes that whether the agent got teleported *causes* the room's
colour. This is backwards: the true generative process sets room hue first (via
`variation.teleport.enabled`, itself set before the episode starts), and *that* determines whether
the teleport event can fire during rollout — not the reverse. A purely observational method applied
to this dataset produces a confident, wrong causal claim, not merely an uninformative one.

This is the direct empirical counterpart to the interventional diagnostics this paper actually
relies on: the AAP surprise ratio and Theme D's ground-truth paired counterfactual both resolve the
hue/teleport relationship *by intervention* (forcing the hue value and observing what the model
predicts, or literally rolling out both conditions from the same start state) — which is precisely
what a purely observational method, no matter how standard or well-established, structurally cannot
do. Reporting these side by side is the point: `ke2021causalmbrl`'s paradigm is now actually run
against this paper's own data, and it fails in exactly the way the paper's broader argument predicts
observational methods should fail on a hidden-confound environment like this one.

### Caveats

- **This tests the *data*, not the *model*.** No JEPA checkpoint is involved — this is a property of
  what's recoverable from the raw dataset's observational variables, independent of what the trained
  encoder does with pixels. It is complementary to, not a replacement for, the model-facing
  diagnostics (AAP ratio, Theme D) elsewhere in this report.
- **Discretization choice is one reasonable option among several.** Tercile-binning `ep_len`/
  `dist_final` and using `chisq` was chosen to avoid imposing a Gaussian assumption on bounded/binary
  variables, but sensitivity to bin count or a continuous (`fisherz`) treatment was not explored.
- **Algorithm choice**: PC's specific orientation rules (Meek's rules applied to the detected
  v-structures) are one standard causal-discovery approach; a different algorithm (GES, LiNGAM,
  etc.) might behave differently on this data. That robustness dimension is out of scope here — the
  point is that *a* standard, well-established method, run properly, produces this result; it is not
  a claim that *no* observational method could ever succeed.

### Reproduce

```
python research/causal_discovery.py --n-episodes 5000 --seed 42
```

No Modal job, no GPU. Requires `causal-learn` (added to `pyproject.toml`). Results:
`causal_discovery_results.json` in the working directory.

## Theme G — Decorrelated-Confound Positive Control (2026-07-27)

The biggest remaining conceptual gap: every prior result on this checkpoint lineage measures
*failure* to disentangle a confound. A positive control asks the opposite question — give the
model training data with **no confound to latch onto at all**, and check whether the same AAP
battery reports a clean result. If it does, that's evidence the metric itself is sound and the
failures documented throughout this report are a property of the confounded training data, not an
artifact of the evaluation pipeline.

### Pipeline

`research/collect_decorrelated.py` reuses `collect_option_c.py`'s confound-override mechanism
(`variation_values` setting `background.color` and `teleport.enabled` together) but draws both
independently per chunk of 20 episodes, rather than Option C's deterministic reversed pairing. A
local smoke test caught a real bug before the expensive full run: the initial draft copied
`collect_option_c.py`'s use of `RandomPolicy` verbatim, which gives ~3× longer episodes than the
actual training recipe (measured: mean `ep_len` ≈88 vs. `glitched_hue_tworoom_half`'s ≈32) —
conflating "confound removed" with "action/behavior distribution also changed." Switched to
`GlitchedHueExpertPolicy` matching `scripts/data/config/glitched_hue_half.yaml` exactly
(`action_noise=0.5, action_repeat_prob=0.05`), confirmed via rerun to restore baseline-scale episode
length (mean ≈22).

Full collection: 20,000 episodes, 420,139 steps, seed 3072. Decorrelation confirmed on a held-out
2,000-episode sample: **corr(hue, teleported) = 0.0425** (vs. training's near-perfect confound).
Training: 100 epochs, identical hyperparameters to every other checkpoint in this report (ViT-Tiny,
SIGReg λ=0.09, batch 64, AdamW lr=5e-5/wd=1e-3, bf16, history=3, 6-layer/16-head predictor).
`validate/sigreg_loss ≈ 1.48` at completion — healthy, not collapsed (contrast with the SIGReg
ablation's SIE in the hundreds).

### Results — not the clean pass the plan anticipated

| Condition | Surprise ratio (mean ± std, N=200) | Struct. inv. error | Pos R² |
|---|---|---|---|
| Unmasked | **154.85 ± 236.24** | 0.236 | 0.996 |
| Masked (teleport patch) | 0.92 ± 0.11 | 0.598 | 0.921 |

The unmasked ratio is an order of magnitude beyond anything else in this entire report (Baseline
≈10, Option B ≈15, everything else single digits) — the opposite of "expected result if the
pipeline is sound: ratio well below 1.0 in both conditions." The masked ratio, by contrast, lands
almost exactly where the plan predicted.

### Diagnosis: the translation-based `z_cf` is severely off-manifold for this checkpoint

Before treating 154.85 as a finding about the model, `--extended-validation`'s on-manifold check
(Theme C's machinery) was run against both conditions:

| Condition | Mahalanobis: z_fact / z_cf / z_rand | kNN: z_fact / z_cf / z_rand |
|---|---|---|
| Unmasked | 57.66 / **587.51** / 2317.20 | 2.24 / 6.02 / 6.01 |
| Masked | 22.44 / **1472.74** / 2440.16 | 0.20 / 2.18 / 2.18 |

For comparison, the *original* confounded-training checkpoint's Theme C check found `z_cf` at
~1.1× `z_fact` under Mahalanobis — clearly on-manifold. Here, `z_cf` sits **10×** (unmasked) to
**65×** (masked) farther from the reference manifold than `z_fact`, and under kNN distance is
statistically indistinguishable from a literally-random direction in *both* conditions (matching,
but far exceeding in severity, Theme C's original "mixed" verdict on kNN).

**Interpretation:** `z_cf = z_fact + Δ_hue` was never a fully trustworthy construction (Theme C
already showed leakage; Theme D already showed the approximation error is large even for the
original model). For *this* checkpoint specifically, it appears to have broken down far more
severely. A plausible mechanism, offered as a hypothesis rather than a confirmed explanation: under
the original confounded training data, hue and teleport-availability were perfectly correlated, so
the model could (and evidently did) encode both along a single shared latent direction — meaning
`Δ_hue`, estimated purely from a hue probe, incidentally captured a joint "hue-and-availability"
shift that stayed roughly on-manifold. Under the decorrelated data, the model can no longer use a
shared direction (the two factors vary independently), so it must encode them separately — and a
translation along the now-hue-only direction no longer corresponds to any real, jointly-consistent
data point. Note this hypothesis does *not* fully explain why the *masked* condition is even more
off-manifold (65×) than *unmasked* (10×) yet reports a ratio far closer to 1.0 — off-manifold
distance and downstream prediction error are evidently not simply monotonically related for this
model, and no strong quantitative claim is made about that specific relationship here.

**Bottom line: the raw surprise-ratio numbers above should not be read as a clean pass or fail for
Theme G's positive-control question.** The pipeline itself worked exactly as intended (decorrelation
confirmed, training healthy) — what broke down is the *evaluation metric's own validity* for this
specific checkpoint, not evidence about the checkpoint's causal competence one way or the other.
This is itself a genuine, reportable finding: the translation-based counterfactual construction used
throughout Options A/B/C+A does not transfer cleanly across checkpoints trained under materially
different data regimes, which is a real limitation of that construction as a general-purpose tool,
independent of anything it says about this particular model.

### Recommended next step (not yet run)

`research/collect_theme_d_paired.py` / `theme_d_paired_validation.py` collect *real* paired
factual/counterfactual rollouts directly from the environment and evaluate a given checkpoint
against them — sidestepping the on-manifold question entirely, since no latent translation is
involved. Both scripts already work against any checkpoint with zero modification; running them
against this decorrelated checkpoint would give a trustworthy, ground-truth read on whether it
actually shows reduced hue-reliance, resolving what the AAP metric alone cannot answer here.

### Caveats

- **Single seed (3072).** The plan explicitly called for starting with one seed before committing to
  a 3-seed extension; given the AAP-metric validity concern above, a 3-seed extension of *this*
  metric would not obviously resolve anything until the Theme D cross-check is run first.
- **This is the single most expensive item in the review cycle** (full 20k-episode collection +
  100-epoch training, ~run overnight). The infrastructure (script, Hydra config, Modal wiring) is
  reusable for any future seed or re-run without further engineering.

### Reproduce

```
modal run modaldotcom/app.py --do-collect-decorrelated --decorrelated-seed 3072
modal run --detach modaldotcom/app.py::train --data glitched_hue_decorrelated --max-epochs 100 --seed 3072 --no-wandb-enabled
modal run modaldotcom/app.py --do-causal-test --policy <new_ckpt> --dataset-name glitched_hue_decorrelated
modal run modaldotcom/app.py --do-causal-test --policy <new_ckpt> --dataset-name glitched_hue_decorrelated --mask-causal-test
modal run modaldotcom/app.py --do-causal-test --policy <new_ckpt> --dataset-name glitched_hue_decorrelated --extended-validation
modal run modaldotcom/app.py --do-causal-test --policy <new_ckpt> --dataset-name glitched_hue_decorrelated --mask-causal-test --extended-validation
```

Checkpoint used: `ts_1785091494_1a91d8/lewm_epoch_100`. Results: `causal_test_glitched_hue_decorrelated_results.json`,
`causal_test_masked_glitched_hue_decorrelated_results.json` (each with on-manifold data when run with
`--extended-validation`) on the `swm-cache` volume alongside the checkpoint.

### Resolution — Theme D ground-truth cross-check confirms the artifact hypothesis

Ran `theme_d_paired_validation.py` directly against `ts_1785091494_1a91d8/lewm_epoch_100`, reusing
the *existing* Theme D paired dataset (`glitched_hue_theme_d_fact.h5`/`_cf.h5`, 200 episodes, seed
42) with zero modification — the paired rollouts are generated from the environment directly and
are not tied to any specific checkpoint's training data, so no recollection was needed.

| Condition | Translation ratio | REAL ground-truth ratio | Approx. error (normalized) |
|---|---|---|---|
| Masked | 0.969 ± 0.036 | **0.916 ± 0.054** | 1.83× |
| Unmasked | 1.115 ± 0.333 | **1.604 ± 1.026** | 4.36× |

**The real ground-truth ratios (0.92 masked, 1.60 unmasked) are modest in both conditions** — nothing
resembling the 154.85 the translation-based `causal_test` AAP cycle reported. This confirms the
artifact hypothesis via a second, independent line of evidence that doesn't depend on any linear
translation at all: `ctx_cf_true` here is the literal real encoder output of the real paired
counterfactual rollout, not `z + Δ_hue`.

**This was originally incomplete: the ratio was the only metric cross-checked against ground truth.**
The paper insists on three metrics before calling anything a verdict elsewhere in this report
(surprise ratio, structural invariance error, AAP consistency advantage) — reporting only the ratio
here was a gap. `theme_d_paired_validation.py` was extended to also compute a ground-truth SIE
(reusing `pos_dirs`, already extracted by Stage 1 but previously unused): `sie_true = mean
|pos_dirs·z_fact_anchor − pos_dirs·z_cf_true_anchor|`, the same formula `_multi_factor_invariance`
uses, with the real counterfactual encoding in place of `z + Δ_hue`. AAP consistency advantage did
**not** need a ground-truth version — verified directly in `_aap_consistency_advantage`'s code that
it never constructs `z_cf` or touches `Δ_hue` at all (it only compares a factual context against a
blind/mean-embedding prior), so the existing `causal_test` numbers for it are already trustworthy.

| Condition | Metric | Translation-based | Ground-truth | Notes |
|---|---|---|---|---|
| Masked | Surprise ratio | 0.969 ± 0.036 | **0.916 ± 0.054** | both close to 1.0 |
| Masked | SIE | 7.600 ± 0.0000017 | **0.310 ± 0.091** | see variance note below |
| Masked | Consistency advantage | 0.086 | *(no ground-truth version needed)* | from `causal_test`, not `z_cf`-dependent |
| Unmasked | Surprise ratio | 1.115 ± 0.333 | **1.604 ± 1.026** | both modest, no blowup |
| Unmasked | SIE | 0.684 ± 0.0000004 | **0.447 ± 0.267** | see variance note below |
| Unmasked | Consistency advantage | 1.519 | *(no ground-truth version needed)* | from `causal_test`, not `z_cf`-dependent |

**A previously-unnoticed discovery, found while building this table, that applies retroactively to
every "Structural invariance error X ± Y" reported anywhere in this document (Baseline, Option
A/B/C/C+A, the SIGReg ablation, Theme G) — the reported std has always been ≈0 by mathematical
construction, not because of genuine batch-to-batch consistency.** Since
`z_cf = z + Δ_hue` with a single fixed `Δ_hue` per run, `_multi_factor_invariance`'s formula
`mean|z·wᵀ − z_cf·wᵀ|` algebraically reduces to `|Δ_hue·wᵀ|` — a constant that cancels the
per-sample `z` dependence entirely, regardless of which batch or episode it's evaluated on. Directly
confirmed here: the translation-SIE std above is `1.6e-6` (masked) and `4.1e-7` (unmasked) —
numerically zero, not "coincidentally small." This does not invalidate the SIE *point estimates*
already reported throughout this report (`|Δ_hue·wᵀ|` is still a real, meaningful orthogonality
measurement between the hue-shift and position-probe directions), but the "± std" attached to every
one of them has never carried the information a reader would reasonably assume it does. The
ground-truth SIE computed here (0.310 masked, 0.447 unmasked) is the first SIE number in this entire
document with genuine, non-trivial variance, since `z_cf_true` is a real, independently-varying
encoding per episode rather than a fixed offset.

**One honest caveat on this table's own "Translation ratio" column**: `theme_d_paired_validation.py`
hardcodes its `Δ_hue` probe-fitting to the *original* baseline dataset (`glitched_hue_tworoom_half`),
not `glitched_hue_decorrelated` — a pre-existing design choice from when Theme D was built
exclusively for the original checkpoint, not parameterized for arbitrary datasets. This table's
"Translation ratio" is therefore a *different* `Δ_hue` estimate than the one `causal_test`'s own AAP
cycle used to produce 154.85 (which fit probes on `glitched_hue_decorrelated` itself, matching the
checkpoint's actual training distribution) — the two "translation" numbers are not directly
comparable, and this table's translation column should not be read as a replication or explanation
of the 154.85 figure. What *does* explain 154.85 is the on-manifold check above (fit and evaluated
entirely within `glitched_hue_decorrelated`, no cross-dataset mismatch), which independently showed
`z_cf` sitting 10-65× off-manifold specifically for this checkpoint's own latent geometry.

**Combined verdict, now on the complete multi-metric picture: two independent, artifact-free
measurements (the on-manifold check's distance-to-manifold diagnosis, and Theme D's real
paired-rollout ratio + SIE) both point away from 154.85 being a genuine property of the model, and
the ground-truth SIE reinforces the same conclusion.** Ground-truth SIE is low in both conditions
(0.310 masked, 0.447 unmasked) — consistent with a model whose position encoding is not badly
entangled with the real counterfactual, matching the modest ground-truth ratios (0.92, 1.60).
Consistency advantage (1.519 unmasked, 0.086 masked, from `causal_test`, unaffected by the
translation artifact since it never constructs `z_cf`) adds a third, independently-computed metric
pointing the same direction — positive in both conditions, meaning factual evidence still reduces
predictive uncertainty relative to a blind prior, as expected of a functioning (not collapsed) world
model. Taken together, all three metrics — surprise ratio, SIE, and consistency advantage — support
the same reading: this positive-control checkpoint shows no dramatic shortcut reliance, closer to
"no confound to latch onto" than the raw unmasked `causal_test` ratio alone suggested. The
translation-based AAP surprise ratio *and* SIE, across this entire report, should be treated as
unreliable specifically for checkpoints whose training data structurally differs from the checkpoint
the metric was originally validated against — and the SIE's reported "± std" in particular should
never have been read as evidence of stability in the first place, for any condition in this document,
given it is mathematically forced to ≈0 regardless of what the model actually does.

### Reproduce (cross-check)

```
modal run modaldotcom/app.py --do-theme-d-validate --policy ts_1785091494_1a91d8/lewm_epoch_100 --mask-theme-d
modal run modaldotcom/app.py --do-theme-d-validate --policy ts_1785091494_1a91d8/lewm_epoch_100
```

No new data collection — reuses the existing `glitched_hue_theme_d_fact.h5`/`_cf.h5` paired dataset.
Results: `theme_d_paired_results_masked.json`, `theme_d_paired_results.json` in
`ts_1785091494_1a91d8/` on the `swm-cache` volume.

## Oracle Calibration — Provable ceiling/floor reference points for the AAP Surprise Ratio (2026-07-27)

Every surprise-ratio number in this report so far comes from a trained JEPA checkpoint, so there is
no independent way to know what a "perfect" or "maximally shortcut-reliant" model *should* score on
this exact metric. `research/oracle_calibration.py` answers that with no model at all: two
hand-coded predictors, built directly on ground-truth simulator features (`dist_to_pad`,
`teleport_enabled`, and a hue-channel score derived from the raw pixel patch), scored through the
same per-episode ratio formula (`eq:per-episode-ratio`/`eq:aap-ratio`) used everywhere else in this
document. No training, no GPU, no new data — this is a pure calibration check on the metric itself,
run against 200 teleported episodes sampled from the existing `glitched_hue_tworoom_half` dataset (2
episodes skipped for unreadable HDF5 chunks, a pre-existing local-copy issue unrelated to this
script).

- **`true_cause_predict`** reads only `dist_to_pad` and `teleport_enabled` — the actual physical
  cause of the outcome — and outputs `sigmoid(K·(radius − dist_to_pad))`, gated to 0 when
  teleport is disabled. It never reads hue.
- **`hue_shortcut_predict`** reads only the pixel-derived `hue_score` and outputs
  `sigmoid(−K_hue·hue_score)` (negative sign because `hue_score` is negative for blue, the
  training-confound color that must predict high teleport-probability). It never reads
  `dist_to_pad` or `teleport_enabled`.

Both predictors are deliberately smooth (sigmoid, not a hard 0/1 threshold) — an earlier hard-coded
draft of `hue_shortcut_predict` (`1.0 if blue else 0.0`) was caught before running anything, since on
teleported-only episodes it forces `surp_fact = 0` exactly, producing a ratio dominated entirely by
the epsilon term rather than anything real. `make_counterfactual` performs a single `do(hue)`
intervention directly on the ground-truth feature dict — flips only `hue`/`hue_score`, leaves
`dist_to_pad`/`teleport_enabled` untouched — guaranteeing zero cross-factor leakage, unlike the
latent-space `z_cf = z + Δ_hue` translation used everywhere else in this report.

### Results

| Oracle | Ratio-of-means | Per-episode mean ± std | N |
|---|---|---|---|
| `true_cause` (ceiling) | 1.0000 | 1.0000 ± 0.0000 | 200 |
| `hue_shortcut` (floor, K_hue calibrated at target_prob=0.95) | 335.26 | 335.27 ± 0.94 | 200 |

**`true_cause`'s ratio ≈ 1.0 is forced by construction, not an empirical finding about robustness.**
Since `make_counterfactual` never touches `dist_to_pad` or `teleport_enabled` — the only two
quantities `true_cause_predict` reads — `pred_fact` and `pred_cf` are identical for any predictor
steepness `K`, so `surp_fact == surp_cf` exactly (confirmed bit-for-bit locally on synthetic data,
and numerically to ~1e-12 on the real 200-episode sample). A 3-point K-sweep (0.5×/1×/2× the
calibrated `K = 0.2944`) confirms this: the ratio stays at 1.000000 at every K, because it is an
algebraic identity being re-verified, not a genuinely free parameter being tested.

**`hue_shortcut`'s "floor" has no equally clean single number, and none is reported here.** The same
K-sweep applied to `K_hue` shows the ratio is extremely sensitive to a hyperparameter that has no
physical anchor (unlike `true_cause_predict`'s pad radius, `K_hue`'s calibration target
`target_prob=0.95` is a modeling choice, not a measurement):

| K_hue (relative to calibrated) | Ratio-of-means | Per-episode mean ± std | surp_fact (mean) | surp_cf (mean) |
|---|---|---|---|---|
| 0.5× (K_hue=0.00659) | 16.87 | 16.87 ± 0.02 | 0.0348 | 0.587 |
| 1.0× (K_hue=0.01319, calibrated) | 335.26 | 335.27 ± 0.94 | 0.00250 | 0.837 |
| 2.0× (K_hue=0.02638) | 129,215.97 | 129,220.47 ± 766.40 | 0.0000076 | 0.983 |

The ratio swings **~8,000×** across a 4× range in `K_hue` — going from 16.87 to 129,216 — while the
absolute surprise values behind it move smoothly and stay well-behaved throughout: `surp_fact` shrinks
toward zero as `K_hue` grows (0.0348 → 0.0025 → 0.0000076) while `surp_cf` stays bounded near its
ceiling (0.587 → 0.837 → 0.983). **This is the same near-zero-denominator mechanism already
documented in `sec:ratio-discrepancy` (Jensen's inequality: as the factual surprise shrinks toward
zero, the ratio blows up even though nothing about the counterfactual surprise itself is changing) —
reproduced here in a fully controlled setting where the blowup can be watched happening directly as
`K_hue` increases and `surp_fact → 0`.** It is the same phenomenon already known to affect the
per-episode mean-of-ratios numbers throughout Themes D/E/G, not a new problem introduced by this
oracle.

Per the reasoning behind this decision: reporting a single number (e.g. 335.26) for the hue-shortcut
floor would be the least defensible figure in this entire report if pressed on where it came from —
`target_prob=0.95` is not measured from anything, so a table entry citing "the" floor without its
footnote invites exactly the citation-without-context failure this report has otherwise been careful
to avoid. Instead: **the hue-shortcut floor is reported as a range (~17 to ~129,000 across the
observed sweep, explicitly unanchored to any principled `K_hue`)**, with the absolute `surp_fact`/
`surp_cf` values above serving as the stable, concrete quantities a reader can actually reason about.

### Interpretation

The ceiling (`true_cause`, ratio ≈ 1.0) is a mathematical guarantee about any predictor that
literally cannot see the intervened factor — it establishes what "no shortcut reliance whatsoever"
looks like on this exact metric, and every checkpoint's ratio in this report should be read relative
to that anchor, not relative to 0. The floor is real in direction (a predictor that reads *only* the
shortcut feature does produce ratios far above 1, confirming the metric responds to shortcut reliance
the way it's supposed to) but its magnitude is not a number this report is prepared to defend as
precise — only the qualitative fact that it is large and grows without bound as the predictor's
confidence in the shortcut increases.

### Caveats

- CPU-only, local script — no GPU, no Modal run, no new HDF5 collection. Results are fully
  reproducible from the existing `glitched_hue_tworoom_half.h5` dataset already on disk.
- `teleport_enabled` is not stored as its own column in `glitched_hue_tworoom_half.h5` (only the
  `teleported` outcome is present); the script falls back to `teleport_enabled = (hue == blue)`,
  valid only because this dataset was built with that exact confound rule — not a general-purpose
  substitution.
- 2 of 200 sampled episodes were skipped for unreadable HDF5 chunks in the local dataset copy, a
  pre-existing corruption issue also encountered in `research/causal_discovery.py`, unrelated to
  anything in this script.
- This calibrates the *metric*, not any specific checkpoint. It says nothing new about
  `lewm_epoch_100` or any other trained model's actual behavior — it establishes what the numbers
  in every other section of this report should be compared against.

### Reproduce

```
source .venv/bin/activate
export HDF5_PLUGIN_PATH=.venv/lib/python3.10/site-packages/hdf5plugin/plugins
python3 research/oracle_calibration.py --n-episodes 200 --seed 42 --k-sweep
```

Results: `oracle_calibration_results.json` (written to the current working directory; not committed
to git, regenerable directly from the command above in under a minute on CPU).

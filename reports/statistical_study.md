# Statistical Study — LeWM Causal Ladder

## Motivation

The initial results table reports single point estimates from a single training seed (3072).
For NeurIPS/ICLR publication, three issues must be addressed:

1. **Ratio-of-means**: `mean(cf) / mean(fact)` is fragile; per-episode ratios with ± std are required.
2. **N = 50 AAP episodes**: too small for stable estimates; 200 episodes removes most variance at zero retraining cost.
3. **Single seed**: reviewers expect ≥3 seeds for training-dependent metrics.

---

## Reuse map — no retraining needed

| Checkpoint path on volume       | Condition  | Seed |
|---------------------------------|------------|------|
| `lewm_epoch_50`                 | Baseline   | 3072 |
| `ts_1776884938/lewm_epoch_50`   | Option B   | 3072 |

Option C and C+A reuse the baseline checkpoint on the `glitched_hue_optionc` dataset — dataset-level test only.

The SIGReg ablation (`ts_1776992707`) crashed at epoch 27. Fresh epoch-50 runs are needed for all three seeds.

---

## New training runs (7 total, all launched in parallel)

| Job | Condition        | Seed | Epochs | Flags                          |
|-----|------------------|------|--------|--------------------------------|
| 1   | Baseline         | 1234 | 50     | —                              |
| 2   | Baseline         | 5678 | 50     | —                              |
| 3   | Option B         | 1234 | 50     | `mask_prob=0.5`                |
| 4   | Option B         | 5678 | 50     | `mask_prob=0.5`                |
| 5   | Ablation (λ=0)   | 3072 | 50     | `mask_prob=0.5, sigreg=0`      |
| 6   | Ablation (λ=0)   | 1234 | 50     | `mask_prob=0.5, sigreg=0`      |
| 7   | Ablation (λ=0)   | 5678 | 50     | `mask_prob=0.5, sigreg=0`      |

Each job runs on a separate A10G GPU via `train.spawn()`.

---

## Parallel execution timeline

```
t=0 h   Phase 1: spawn 7 training runs simultaneously
        [Baseline-1234] [Baseline-5678]
        [OptionB-1234]  [OptionB-5678]
        [Ablation-3072] [Ablation-1234] [Ablation-5678]

t≈7.5 h Phase 2: spawn all causal tests simultaneously (≤15 min each)
        [Baseline×3] [OptionB×3] [Ablation×3] [OptionC] [OptionC+A]

t≈8 h   Phase 3: aggregate across seeds (<1 min)
        → statistical_study_results.json
```

Total wall-clock: ~8 h (vs ~54 h sequential).

---

## Statistical methodology

### Surprise ratio
- **Old**: `mean(surp_cf per episode) / mean(surp_fact per episode)` — ratio of means, no variance estimate.
- **New**: per-episode ratio `r_i = surp_cf_i / (surp_fact_i + ε)`, then report `mean(r_i) ± std(r_i)` over N=200 episodes.

### Structural invariance error
- N=30 loader batches; now returns `mean ± std` across batches.

### Cross-seed aggregation
Each metric key `k` in the results JSON gets `k_mean` and `k_std` over the `n_seeds` runs.

### Probe metrics (position R², hue accuracy)
Linear probes are deterministic given the trained encoder; variance comes from the encoder only. These are included in the aggregation but are expected to show small std for stable conditions.

---

## How to run

```bash
# Full study (7 parallel training + 11 parallel causal tests)
modal run modaldotcom/app.py --do-statistical-study

# Custom seeds or episode count
modal run modaldotcom/app.py --do-statistical-study \
    --study-seeds 3072,1234,5678 --study-epochs 50 --study-n-aap 200

# Re-run only causal tests (no retraining)
modal run modaldotcom/app.py --do-causal-test \
    --policy lewm_epoch_50 --n-aap-episodes 200

# Download results
modal volume get swm-cache statistical_study_results.json
```

---

## Expected output format

```json
{
  "seeds": [3072, 1234, 5678],
  "n_aap_episodes": 200,
  "conditions": {
    "baseline": {
      "surprise_ratio_mean":                0.718,
      "surprise_ratio_std":                 0.031,
      "structural_invariance_error_mean":   0.204,
      "structural_invariance_error_std":    0.012,
      "n_seeds": 3
    },
    "option_b": { ... },
    "ablation": { ... },
    "option_c": { ... },
    "option_ca": { ... }
  }
}
```

---

## LaTeX table template

```latex
\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
Condition & Surprise ratio & Struct. inv. err & Pos $R^2$ & Hue acc \\
          & (mean $\pm$ std, $N=200$) & (mean $\pm$ std, $N=30$) & & \\
\midrule
Baseline  & $0.718 \pm 0.031$ & $0.204 \pm 0.012$ & $0.993$ & $0.997$ \\
Option A  & — & — & — & — \\
Option B  & $3.327 \pm ??$ & $0.197 \pm ??$ & $??$ & $??$ \\
Option C  & $0.800 \pm ??$ & $0.207 \pm ??$ & $??$ & $??$ \\
Option C+A & — & — & — & — \\
Ablation ($\lambda=0$) & $2.196 \pm ??$ & $1439.5 \pm ??$ & $0.064$ & $??$ \\
\bottomrule
\end{tabular}
\caption{LeWM causal ladder results (mean $\pm$ std over 3 seeds, 200 AAP episodes).}
\end{table}
```

Fill in `??` cells after `--do-statistical-study` completes.

---

## Verification checklist

After the study completes:

1. Confirm `n_seeds: 3` for each condition in the JSON.
2. `surprise_ratio_mean` for Baseline ≈ 0.718 (matches single-seed result); std < 0.05.
3. `structural_invariance_error_mean` for Ablation >> 1 (existing: 1439); std is large (expected, since the latent space is unstructured without SIGReg).
4. Check `per_episode_ratios` array in any single-run JSON to confirm N=200 entries.
5. Ablation `position_probe_r2` ≈ 0.064 (latent space collapses without SIGReg).

#!/usr/bin/env python3
"""
Theme D validation: does a REAL ground-truth counterfactual change the verdict?

Every existing causal-test metric in research/glitched_hue_experiment.py
compares the model's prediction against a translated latent,
z_cf = z_fact + delta_hue -- never against a real encoder output of an
actually-different rollout. This script loads the paired dataset produced by
research/collect_theme_d_paired.py (same seed/start/actions, hue + teleport-
gating flipped per the training confound) and reports:

  (a) ||z_cf_translation - z_cf_true|| -- how good is the translation
      approximation, in the same latent-space units used throughout.
  (b) The AAP surprise ratio recomputed with the translation-based ctx_cf
      swapped for the REAL encoded counterfactual context (ctx_cf_true),
      keeping _run_aap_cycle's formula structure otherwise identical (same
      `tgt` = the real factual next-frame embedding in both branches) --
      and whether the "ratio crosses 1.0" verdict changes.

No retraining: reuses the existing lewm_epoch_50 checkpoint, and reuses
_load_model/_make_loader/_train_probes from glitched_hue_experiment.py so the
delta_hue translation baseline is identical to what's already reported.

Usage:
    python research/theme_d_paired_validation.py <ckpt_path>
    python research/theme_d_paired_validation.py <ckpt_path> --paired-dataset-name glitched_hue_theme_d
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.glitched_hue_experiment import (  # noqa: E402
    _load_model, _make_loader, _extract_probe_data, _train_probes,
    _setup_pub_style, _despine, _save_fig, _C,
    _DEVICE, _HISTORY_SIZE, _FRAMESKIP, _NUM_STEPS, _DATASET_NAME,
)

_PAIRED_DATASET_NAME = "glitched_hue_theme_d"

# _run_aap_cycle's boundary convention: with its `t_tp` = index of the last
# PRE-teleport context frame, it requires 1 <= t_tp <= _NUM_STEPS-3. Our
# `enc_idx` = index of the POST-teleport (target) frame, so t_tp = enc_idx-1,
# giving enc_idx in [2, _NUM_STEPS-2]. Mirrored here so N is directly
# comparable to the already-reported translation-based ratio.
_ENC_IDX_CANDIDATES = [3, 2, 4, _NUM_STEPS - 2]


def _find_window(t_tp_raw: int, ep_len: int, span: int):
    """Return (start, enc_idx) for a raw-step window containing the teleport
    event at a valid encoded anchor index, or None if no candidate fits."""
    for enc_idx in _ENC_IDX_CANDIDATES:
        start = t_tp_raw - enc_idx * _FRAMESKIP
        if start >= 0 and start + span <= ep_len:
            return start, enc_idx
    return None


@torch.no_grad()
def _encode_batch(jepa, ds, ep_idx, start, span):
    chunk = ds.load_chunk(np.array(ep_idx), np.array(start), np.array(start) + span)
    pixels = torch.stack([c["pixels"] for c in chunk]).to(_DEVICE)
    action = torch.stack([c["action"] for c in chunk]).to(_DEVICE)
    out = jepa.encode({"pixels": pixels, "action": action})
    return out["emb"], out["act_emb"]


def main():
    parser = argparse.ArgumentParser(description="Theme D: real vs. translated counterfactual")
    parser.add_argument("ckpt_path", help="Path to *_object.ckpt checkpoint file")
    parser.add_argument("--paired-dataset-name", default=_PAIRED_DATASET_NAME,
                         help="Prefix for the paired dataset (default: glitched_hue_theme_d); "
                              "expects <name>_fact.h5 and <name>_cf.h5 in STABLEWM_HOME")
    parser.add_argument("--n-probe-batches", type=int, default=200,
                         help="DataLoader batches for training the baseline hue/delta probes (default: 200)")
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Episodes per encode() call (default: 16)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    import stable_worldmodel as swm
    import h5py

    out_dir = Path(args.ckpt_path).parent
    print(f"Device   : {_DEVICE}")
    print(f"Output   : {out_dir}")

    # -----------------------------------------------------------------
    # Stage 1 -- load model + baseline hue/delta probes (same data/code
    # path as the already-reported translation baseline)
    # -----------------------------------------------------------------
    print("\n[1/4] Loading checkpoint + baseline probes ...")
    jepa = _load_model(args.ckpt_path)
    baseline_loader = _make_loader(dataset_name=_DATASET_NAME, seed=42)
    z_all, hue_all, pos_all, max_deltas = _extract_probe_data(jepa, baseline_loader, args.n_probe_batches)
    (probe_pos, probe_hue, scaler, pos_r2, hue_acc, hue_dir, delta_hue, pos_dirs) = _train_probes(z_all, hue_all, pos_all)
    print(f"      Position R^2 = {pos_r2:.4f}   Hue accuracy = {hue_acc:.4f}")
    baseline_transform = baseline_loader.dataset.transform

    # -----------------------------------------------------------------
    # Stage 2 -- open paired datasets; pick a valid teleport-anchored
    # window per episode using GROUND-TRUTH teleport_step (no proprio-
    # delta heuristic needed here, unlike _run_aap_cycle)
    # -----------------------------------------------------------------
    print("\n[2/4] Scanning paired dataset for valid teleport-anchored windows ...")
    cache = swm.data.utils.get_cache_dir()
    fact_name = f"{args.paired_dataset_name}_fact"
    cf_name = f"{args.paired_dataset_name}_cf"
    fact_path = cache / f"{fact_name}.h5"
    cf_path = cache / f"{cf_name}.h5"

    with h5py.File(fact_path, "r") as f_fact, h5py.File(cf_path, "r") as f_cf:
        ep_len_fact = f_fact["ep_len"][:]
        ep_len_cf = f_cf["ep_len"][:]
        teleport_step = f_fact["teleport_step"][:]
        pair_valid = f_fact["pair_valid"][:]
        n_episodes = len(ep_len_fact)
        if not np.array_equal(ep_len_fact, ep_len_cf):
            raise RuntimeError("fact/cf episode-length mismatch -- datasets are not paired correctly")

    span = _NUM_STEPS * _FRAMESKIP
    windows = []  # (ep_idx, start, enc_idx)
    n_no_teleport = n_invalid_pair = n_no_window = 0
    for ep in range(n_episodes):
        if teleport_step[ep] < 0:
            n_no_teleport += 1
            continue
        if not bool(pair_valid[ep]):
            n_invalid_pair += 1
            continue
        w = _find_window(int(teleport_step[ep]), int(ep_len_fact[ep]), span)
        if w is None:
            n_no_window += 1
            continue
        start, enc_idx = w
        windows.append((ep, start, enc_idx))

    print(f"      {n_episodes} episodes collected")
    print(f"      excluded: no teleport={n_no_teleport}  invalid pair={n_invalid_pair}  no valid window={n_no_window}")
    print(f"      usable episodes: {len(windows)}")
    if not windows:
        print("      ERROR: no usable episodes -- check collection episode_len / teleport rate")
        return {}

    ds_fact = swm.data.HDF5Dataset(name=fact_name, frameskip=_FRAMESKIP, num_steps=_NUM_STEPS,
                                    transform=baseline_transform, keys_to_load=["pixels", "action", "proprio"])
    ds_cf = swm.data.HDF5Dataset(name=cf_name, frameskip=_FRAMESKIP, num_steps=_NUM_STEPS,
                                  transform=baseline_transform, keys_to_load=["pixels", "action", "proprio"])

    # -----------------------------------------------------------------
    # Stage 3 -- encode fact/cf windows in lockstep; compute both metrics
    # -----------------------------------------------------------------
    print("\n[3/4] Encoding paired windows and computing metrics ...")
    approx_err, approx_err_norm = [], []
    surp_fact, surp_cf_translation, surp_cf_true = [], [], []
    delta_hue_norm = float(delta_hue.norm())

    bs = args.batch_size
    for i in range(0, len(windows), bs):
        batch = windows[i:i + bs]
        eps = [w[0] for w in batch]
        starts = [w[1] for w in batch]
        enc_idxs = [w[2] for w in batch]

        emb_fact, act_emb_fact = _encode_batch(jepa, ds_fact, eps, starts, span)
        emb_cf, _ = _encode_batch(jepa, ds_cf, eps, starts, span)

        for j, enc_idx in enumerate(enc_idxs):
            ctx_start = max(0, enc_idx - _HISTORY_SIZE)

            z_fact_anchor = emb_fact[j, enc_idx - 1]      # last pre-teleport frame
            z_cf_true_anchor = emb_cf[j, enc_idx - 1]     # same frame, real cf encoding
            z_cf_translation = z_fact_anchor + delta_hue

            err = float((z_cf_translation - z_cf_true_anchor).norm())
            approx_err.append(err)
            approx_err_norm.append(err / (delta_hue_norm + 1e-8))

            ctx = emb_fact[j, ctx_start:enc_idx].unsqueeze(0)
            act_ctx = act_emb_fact[j, ctx_start:enc_idx].unsqueeze(0)
            tgt = emb_fact[j, enc_idx].unsqueeze(0)        # real FACTUAL outcome (ground truth)

            pred_f = jepa.predict(ctx, act_ctx)[:, -1]
            s_fact = F.mse_loss(pred_f, tgt).item()

            ctx_cf_translation = ctx + delta_hue
            pred_cf_translation = jepa.predict(ctx_cf_translation, act_ctx)[:, -1]
            s_cf_translation = F.mse_loss(pred_cf_translation, tgt).item()

            ctx_cf_true = emb_cf[j, ctx_start:enc_idx].unsqueeze(0)
            pred_cf_true = jepa.predict(ctx_cf_true, act_ctx)[:, -1]
            s_cf_true = F.mse_loss(pred_cf_true, tgt).item()

            surp_fact.append(s_fact)
            surp_cf_translation.append(s_cf_translation)
            surp_cf_true.append(s_cf_true)

        if (i + bs) % (bs * 5) == 0 or i + bs >= len(windows):
            print(f"      encoded {min(i + bs, len(windows))}/{len(windows)}")

    eps_ = 1e-12
    ratio_translation = [cf / (f + eps_) for cf, f in zip(surp_cf_translation, surp_fact)]
    ratio_true = [cf / (f + eps_) for cf, f in zip(surp_cf_true, surp_fact)]

    def _summ(arr):
        return {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90))}

    def _ratio_summ(arr):
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": len(arr)}

    metrics = {
        "n_episodes_collected": int(n_episodes),
        "n_episodes_usable": len(windows),
        "n_excluded_no_teleport": int(n_no_teleport),
        "n_excluded_invalid_pair": int(n_invalid_pair),
        "n_excluded_no_window": int(n_no_window),
        "translation_approx_error": _summ(approx_err),
        "translation_approx_error_normalized": _summ(approx_err_norm),
        "surprise_ratio_translation_baseline": _ratio_summ(ratio_translation),
        "surprise_ratio_true_counterfactual": _ratio_summ(ratio_true),
        "ratio_crosses_one_translation": bool(np.mean(ratio_translation) >= 1.0),
        "ratio_crosses_one_true": bool(np.mean(ratio_true) >= 1.0),
    }

    print("\n" + "=" * 66)
    print("  THEME D -- PAIRED GROUND-TRUTH COUNTERFACTUAL VALIDATION")
    print("=" * 66)
    print(f"  Usable episodes (of {n_episodes} collected): {len(windows)}")
    print(f"  ||z_cf_translation - z_cf_true||   "
          f"{metrics['translation_approx_error']['mean']:.4f} "
          f"(median {metrics['translation_approx_error']['median']:.4f}, "
          f"p90 {metrics['translation_approx_error']['p90']:.4f})")
    print(f"    normalized by ||delta_hue||={delta_hue_norm:.4f}: "
          f"{metrics['translation_approx_error_normalized']['mean']:.4f}")
    rt = metrics["surprise_ratio_translation_baseline"]
    ru = metrics["surprise_ratio_true_counterfactual"]
    print(f"  Surprise ratio (translation ctx_cf):  {rt['mean']:.4f} ± {rt['std']:.4f}  (N={rt['n']})"
          f"  crosses 1.0: {metrics['ratio_crosses_one_translation']}")
    print(f"  Surprise ratio (REAL ctx_cf_true):     {ru['mean']:.4f} ± {ru['std']:.4f}  (N={ru['n']})"
          f"  crosses 1.0: {metrics['ratio_crosses_one_true']}")
    verdict_changed = metrics["ratio_crosses_one_translation"] != metrics["ratio_crosses_one_true"]
    print(f"  Verdict changed by using the real counterfactual: {verdict_changed}")
    print("=" * 66)

    results_path = out_dir / "theme_d_paired_results.json"
    with open(results_path, "w") as f:
        json.dump({"checkpoint": str(args.ckpt_path), "paired_dataset": args.paired_dataset_name,
                   "metrics": metrics}, f, indent=2)
    print(f"\nResults  -> {results_path}")

    print("\n[4/4] Saving plots ...")
    _plot_approx_error(approx_err, approx_err_norm, out_dir)
    _plot_ratio_comparison(ratio_translation, ratio_true, out_dir)
    print(f"Plots    -> {out_dir}/theme_d_approx_error.pdf / .png")
    print(f"         -> {out_dir}/theme_d_ratio_comparison.pdf / .png")

    if not args.no_wandb:
        _log_to_wandb(metrics, args.ckpt_path, out_dir)

    return metrics


def _plot_approx_error(approx_err, approx_err_norm, out_dir: Path):
    _setup_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7))
    for ax, data, title in [
        (axes[0], approx_err, r"$\|z_\mathrm{cf}^{\mathrm{transl}} - z_\mathrm{cf}^{\mathrm{true}}\|$"),
        (axes[1], approx_err_norm, r"normalized by $\|\Delta_\mathrm{hue}\|$"),
    ]:
        bp = ax.boxplot([data], tick_labels=[""], showfliers=False, patch_artist=True)
        bp["boxes"][0].set_facecolor(_C["purple"])
        bp["boxes"][0].set_alpha(0.35)
        ax.set_title(title, fontsize=8)
        _despine(ax)
    fig.tight_layout()
    _save_fig(fig, out_dir, "theme_d_approx_error")


def _plot_ratio_comparison(ratio_translation, ratio_true, out_dir: Path):
    _setup_pub_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    labels = ["Translation\n(existing)", "Real ground-truth\n(Theme D)"]
    data = [ratio_translation, ratio_true]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, colour in zip(bp["boxes"], [_C["orange"], _C["green"]]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.35)
    ax.axhline(1.0, color=_C["tp"], linestyle=":", lw=1.0)
    ax.set_ylabel("Per-episode surprise ratio (cf / fact)")
    _despine(ax)
    fig.tight_layout()
    _save_fig(fig, out_dir, "theme_d_ratio_comparison")


def _log_to_wandb(metrics, ckpt_path, out_dir: Path):
    import wandb

    run_tag = Path(ckpt_path).parent.name
    run = wandb.init(
        project="lewm-causality",
        entity="paoloai-robomotic",
        name=f"theme_d_{run_tag}",
        tags=["causal_test", "aap", "theme_d", "ground_truth_counterfactual"],
        config={"checkpoint": str(ckpt_path)},
    )
    payload = {f"theme_d/{k}": v for k, v in metrics.items() if isinstance(v, (int, float, bool))}
    for stem in ("theme_d_approx_error", "theme_d_ratio_comparison"):
        p = out_dir / f"{stem}.png"
        if p.exists():
            payload[f"theme_d/{stem}"] = wandb.Image(str(p))
    wandb.log(payload)
    print(f"\nW&B run  -> {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()

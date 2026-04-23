#!/usr/bin/env python3
"""
Causal disentanglement test for LeWM on GlitchedHueTwoRoom.

Tests whether the trained world model learned the true causal mechanism
(teleport pixel) or the spurious correlation (background hue), using
Pearl's Abduction-Action-Prediction (AAP) framework.

Three metrics (research/measures.md):
  1. Surprise ratio          — counterfactual / factual MSE at the teleport step
  2. Structural invariance   — does hue intervention leak into position dims?
  3. AAP consistency advantage — does factual evidence improve predictions vs blind?

Results: JSON + two plots saved next to the checkpoint and logged to W&B.

Usage:
    python research/glitched_hue_experiment.py <ckpt_path>
    python research/glitched_hue_experiment.py <ckpt_path> --no-wandb
    python research/glitched_hue_experiment.py <ckpt_path> --n-probe-batches 100

    # Option A — Ladder 3 test: mask the teleport pixel patch before every encode
    python research/glitched_hue_experiment.py <ckpt_path> --mask-teleport
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

# Make local modules (jepa.py, module.py, utils.py) importable when the script
# is invoked from /workspace or from the research/ subdirectory.
sys.path.insert(0, str(Path(__file__).parent.parent))

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_IMG_SIZE = 224
_NUM_STEPS = 4          # history_size(3) + num_preds(1), matches training config
_FRAMESKIP = 5
_DATASET_NAME = "glitched_hue_tworoom_half"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Causal disentanglement test for LeWM (AAP pipeline)"
    )
    parser.add_argument("ckpt_path", help="Path to *_object.ckpt checkpoint file")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    parser.add_argument(
        "--n-probe-batches", type=int, default=200,
        help="DataLoader batches used to train the linear probes (default: 200)",
    )
    parser.add_argument(
        "--n-aap-episodes", type=int, default=20,
        help="Teleport episodes to average the AAP cycle over (default: 20)",
    )
    parser.add_argument(
        "--mask-teleport", action="store_true",
        help=(
            "Zero out the teleport-pixel patch(es) before every encode call. "
            "Removes the direct causal cue so the model must rely on latent "
            "inference alone — Option A Ladder 3 test (reports/testladder.md)."
        ),
    )
    parser.add_argument(
        "--dataset-name", default=_DATASET_NAME,
        help="HDF5 dataset name in STABLEWM_HOME (default: glitched_hue_tworoom_half)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    ds_suffix = f"_{dataset_name}" if dataset_name != _DATASET_NAME else ""
    suffix  = "_masked" if args.mask_teleport else ""
    suffix  = suffix + ds_suffix
    out_dir = Path(args.ckpt_path).parent
    print(f"Device   : {_DEVICE}")
    print(f"Output   : {out_dir}")
    print(f"Mask TP  : {args.mask_teleport}")
    print(f"Dataset  : {dataset_name}")

    # -------------------------------------------------------------------
    # Stage 0 — Detect teleport patch bbox (only when masking is requested)
    # -------------------------------------------------------------------
    tp_bbox = None
    if args.mask_teleport:
        import stable_worldmodel as swm
        dataset_path = str(swm.data.utils.get_cache_dir() / f"{dataset_name}.h5")
        print(f"\n[0/5] Detecting teleport patch bbox from {dataset_path} ...")
        tp_bbox = _detect_teleport_bbox(dataset_path)
        r0, r1, c0, c1 = tp_bbox
        print(
            f"      pixel bbox  rows [{r0}:{r1}], cols [{c0}:{c1}]  "
            f"→ patches row [{r0//14}:{r1//14}], col [{c0//14}:{c1//14}]"
        )

    # -------------------------------------------------------------------
    # Stage 1 — Load model
    # -------------------------------------------------------------------
    print("\n[1/5] Loading checkpoint ...")
    jepa = _load_model(args.ckpt_path)
    print(f"      {args.ckpt_path}")

    # -------------------------------------------------------------------
    # Stage 2 — Extract latents; train linear probes
    # -------------------------------------------------------------------
    print(f"\n[2/5] Extracting latents ({args.n_probe_batches} batches) ...")
    loader = _make_loader(dataset_name=dataset_name)
    z_all, hue_all, pos_all, max_deltas = _extract_probe_data(
        jepa, loader, args.n_probe_batches,
        mask_teleport=args.mask_teleport, tp_bbox=tp_bbox,
    )
    print(f"      {len(z_all)} samples, embed_dim={z_all.shape[1]}")

    print("      Training probes ...")
    (
        probe_pos, probe_hue, scaler,
        pos_r2, hue_acc,
        hue_dir, delta_hue, pos_dirs,
    ) = _train_probes(z_all, hue_all, pos_all)
    print(f"      Position R² = {pos_r2:.4f}   Hue accuracy = {hue_acc:.4f}")

    # -------------------------------------------------------------------
    # Stage 3 — AAP cycle
    # -------------------------------------------------------------------
    print(f"\n[3/5] AAP cycle ({args.n_aap_episodes} teleport episodes) ...")
    aap_results = _run_aap_cycle(
        jepa, loader, hue_dir, delta_hue, args.n_aap_episodes,
        mask_teleport=args.mask_teleport, tp_bbox=tp_bbox,
    )
    if not aap_results:
        print("      WARNING: no teleport episodes found — check dataset or threshold")
        return {}

    surp_fact  = float(np.mean([r["surprise_factual"]       for r in aap_results]))
    surp_cf    = float(np.mean([r["surprise_counterfactual"] for r in aap_results]))
    surp_ratio = surp_cf / (surp_fact + 1e-12)
    print(f"      Factual surprise:         {surp_fact:.6f}")
    print(f"      Counterfactual surprise:  {surp_cf:.6f}")
    print(f"      Surprise ratio (cf/fact): {surp_ratio:.4f}")

    # -------------------------------------------------------------------
    # Stage 4a — Structural invariance
    # -------------------------------------------------------------------
    print("\n[4a/5] Structural invariance ...")
    inv_error = _structural_invariance(
        jepa, loader, delta_hue, pos_dirs,
        mask_teleport=args.mask_teleport, tp_bbox=tp_bbox,
    )
    print(f"       Invariance error: {inv_error:.6f}")

    # -------------------------------------------------------------------
    # Stage 4b — AAP consistency advantage
    # -------------------------------------------------------------------
    print("\n[4b/5] AAP consistency advantage ...")
    aap_adv, surp_with, surp_without = _aap_consistency_advantage(
        jepa, loader,
        mask_teleport=args.mask_teleport, tp_bbox=tp_bbox,
    )
    print(f"       Surprise with evidence:    {surp_with:.6f}")
    print(f"       Surprise without evidence: {surp_without:.6f}")
    print(f"       Advantage:                 {aap_adv:.6f}")

    # -------------------------------------------------------------------
    # Stage 5 — Report, save, visualise
    # -------------------------------------------------------------------
    metrics = {
        "position_probe_r2":             pos_r2,
        "hue_probe_accuracy":            hue_acc,
        "surprise_factual":              surp_fact,
        "surprise_counterfactual":       surp_cf,
        "surprise_ratio":                surp_ratio,
        "structural_invariance_error":   inv_error,
        "aap_consistency_advantage":     aap_adv,
        "aap_surprise_with_evidence":    surp_with,
        "aap_surprise_without_evidence": surp_without,
    }

    _print_report(metrics)

    results_path = out_dir / f"causal_test{suffix}_results.json"
    with open(results_path, "w") as f:
        json.dump({"checkpoint": str(args.ckpt_path), "metrics": metrics}, f, indent=2)
    print(f"\nResults  → {results_path}")

    print("\n[5/5] Saving plots ...")
    _save_plots(aap_results, z_all, hue_all, max_deltas, delta_hue, out_dir, suffix)
    print(f"Plots    → {out_dir}/surprise_over_time{suffix}.pdf / .png")
    print(f"         → {out_dir}/latent_pca{suffix}.pdf / .png")

    if not args.no_wandb:
        _log_to_wandb(metrics, args.ckpt_path, out_dir, suffix)

    return metrics


# ---------------------------------------------------------------------------
# Stage 1 — Model loading
# ---------------------------------------------------------------------------

def _load_model(ckpt_path):
    """Load the JEPA object saved by ModelObjectCallBack (torch.save)."""
    jepa = torch.load(ckpt_path, map_location=_DEVICE, weights_only=False)
    jepa.eval()
    return jepa


# ---------------------------------------------------------------------------
# Teleport patch detection and masking helpers (Option A — Ladder 3 test)
# ---------------------------------------------------------------------------

# Shared with train.py (Option B augmentation) — canonical implementation in utils.py
from utils import detect_teleport_bbox as _detect_teleport_bbox


def _mask_tp(pixels: torch.Tensor, tp_bbox: tuple) -> torch.Tensor:
    """Return a copy of `pixels` with the teleport patch region zeroed out.

    Args:
        pixels: (..., C, H, W) float tensor (already ImageNet-normalised).
        tp_bbox: (row_min, row_max, col_min, col_max) in pixel space.
    """
    r0, r1, c0, c1 = tp_bbox
    p = pixels.clone()
    p[..., r0:r1, c0:c1] = 0.0
    return p


# ---------------------------------------------------------------------------
# Stage 2 — Data loading and probe training
# ---------------------------------------------------------------------------

def _make_loader(batch_size=64, shuffle=True, dataset_name=_DATASET_NAME):
    """Build a DataLoader over the given HDF5 dataset with the training pipeline."""
    import stable_worldmodel as swm
    import stable_pretraining as spt
    from utils import get_img_preprocessor, get_column_normalizer

    dataset = swm.data.HDF5Dataset(
        num_steps=_NUM_STEPS,
        frameskip=_FRAMESKIP,
        name=dataset_name,
        keys_to_load=["pixels", "action", "proprio"],
        keys_to_cache=["action", "proprio"],
        transform=None,
    )
    transforms = [get_img_preprocessor("pixels", "pixels", _IMG_SIZE)]
    for col in ("action", "proprio"):
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, drop_last=True,
    )


@torch.no_grad()
def _extract_probe_data(jepa, loader, n_batches, mask_teleport=False, tp_bbox=None):
    """Return (z, hue_scores, positions, max_deltas) arrays for probe training.

    hue_scores: G_norm - B_norm per window (positive = green, negative = blue)
    positions:  first 2 dims of proprio at the first timestep
    max_deltas: maximum inter-frame positional delta within each window (proxy
                for teleport events — large values indicate a teleport fired)
    """
    zs, hues, poss, deltas = [], [], [], []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        pixels = batch["pixels"].to(_DEVICE)  # (B, T, C, H, W)
        action = batch["action"].to(_DEVICE)
        if mask_teleport:
            pixels = _mask_tp(pixels, tp_bbox)

        out = jepa.encode({"pixels": pixels, "action": action})
        zs.append(out["emb"][:, 0].cpu().float().numpy())  # first-frame latent (B, D)

        # After ImageNet normalisation: G_norm-B_norm is positive for green rooms
        # (green channel normalised mean ≈ 0.456/0.224, blue ≈ 0.406/0.225)
        hues.append(
            (pixels[:, 0, 1] - pixels[:, 0, 2])
            .mean(dim=(-2, -1)).cpu().float().numpy()
        )
        poss.append(batch["proprio"][:, 0, :2].float().numpy())  # (B, 2)

        delta = (
            batch["proprio"][:, 1:, :2] - batch["proprio"][:, :-1, :2]
        ).norm(dim=-1)  # (B, T-1)
        deltas.append(delta.max(dim=-1).values.float().numpy())

    return (
        np.concatenate(zs),
        np.concatenate(hues),
        np.concatenate(poss),
        np.concatenate(deltas),
    )


def _train_probes(z, hue_scores, pos):
    """Train position (Ridge) and hue (LogisticRegression) linear probes.

    Returns probes, metrics, and pre-computed intervention vectors in the
    original (unscaled) latent space:
      hue_dir  — unit vector pointing from blue-like to green-like embeddings
      delta_hue — mean blue→green translation (hue intervention vector)
      pos_dirs  — (2, D) position probe weight directions (unscaled)
    """
    rng = np.random.default_rng(42)
    scaler = StandardScaler()
    z_s = scaler.fit_transform(z)

    hue_labels = (hue_scores > 0.0).astype(int)  # green=1, blue=0

    idx   = rng.permutation(len(z))
    n_tr  = int(0.8 * len(z))
    tr, te = idx[:n_tr], idx[n_tr:]

    probe_pos = Ridge(alpha=1.0)
    probe_pos.fit(z_s[tr], pos[tr])
    pos_r2 = float(r2_score(pos[te], probe_pos.predict(z_s[te])))

    probe_hue = LogisticRegression(max_iter=1000, C=1.0)
    probe_hue.fit(z_s[tr], hue_labels[tr])
    hue_acc = float(accuracy_score(hue_labels[te], probe_hue.predict(z_s[te])))

    # Convert probe coefficients from scaled space to original latent space
    # StandardScaler: z_s = (z - mean) / scale  →  coef_orig = coef_scaled / scale
    inv_scale = 1.0 / (scaler.scale_ + 1e-8)

    coef_hue_orig  = probe_hue.coef_[0] * inv_scale              # (D,)
    hue_norm       = np.linalg.norm(coef_hue_orig) + 1e-8
    hue_dir_np     = coef_hue_orig / hue_norm
    hue_dir = torch.tensor(hue_dir_np, dtype=torch.float32, device=_DEVICE)

    # Mean blue→green translation in the original latent space
    blue_proj  = z[hue_labels == 0] @ hue_dir_np  # (N_blue,)
    green_proj = z[hue_labels == 1] @ hue_dir_np  # (N_green,)
    shift      = float(green_proj.mean() - blue_proj.mean())
    delta_hue  = torch.tensor(shift * hue_dir_np, dtype=torch.float32, device=_DEVICE)

    # Position probe weight directions (unscaled)
    pos_coef_orig = probe_pos.coef_ * inv_scale[None]  # (2, D)
    pos_dirs = torch.tensor(pos_coef_orig, dtype=torch.float32, device=_DEVICE)

    return probe_pos, probe_hue, scaler, pos_r2, hue_acc, hue_dir, delta_hue, pos_dirs


# ---------------------------------------------------------------------------
# Stage 3 — AAP cycle
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_aap_cycle(jepa, loader, hue_dir, delta_hue, n_episodes,
                   mask_teleport=False, tp_bbox=None):
    """Encode factual (blue+teleport) windows; intervene on hue; measure per-step surprise.

    For each teleport window found, computes surprise at every prediction step
    (growing context: ctx=[f0] → predict f1, ctx=[f0,f1] → predict f2, …).
    This reveals whether the hue intervention specifically disrupts the
    prediction at the teleport step vs later steps.

    Intervention: translate embedding by delta_hue (mean blue→green shift)
    rather than a reflection, so the counterfactual latent sits in the green
    room cluster rather than at a mirrored blue position.
    """
    results = []

    for batch in loader:
        if len(results) >= n_episodes:
            break

        proprio   = batch["proprio"]                        # (B, T, proprio_dim)
        delta     = (proprio[:, 1:, :2] - proprio[:, :-1, :2]).norm(dim=-1)  # (B, T-1)
        max_delta, t_idx = delta.max(dim=-1)                # (B,)

        # Teleport threshold: 80th percentile of per-sample max position change
        threshold = float(torch.quantile(max_delta, 0.80))

        pixels  = batch["pixels"].to(_DEVICE)
        action  = batch["action"].to(_DEVICE)
        if mask_teleport:
            pixels = _mask_tp(pixels, tp_bbox)
        out     = jepa.encode({"pixels": pixels, "action": action})
        emb     = out["emb"]      # (B, T, D)
        act_emb = out["act_emb"]
        T       = emb.size(1)

        for b in range(emb.size(0)):
            if len(results) >= n_episodes:
                break
            if max_delta[b].item() < threshold:
                continue

            t_tp = min(t_idx[b].item(), T - 2)  # frame index of the teleport
            z = emb[b]      # (T, D)
            a = act_emb[b]  # (T, A)

            surp_f_steps, surp_cf_steps = [], []
            for t in range(T - 1):
                ctx     = z[:t + 1].unsqueeze(0)    # (1, t+1, D)
                act_ctx = a[:t + 1].unsqueeze(0)
                tgt     = z[t + 1].unsqueeze(0)     # (1, D)

                pred_f  = jepa.predict(ctx, act_ctx)[:, -1]
                surp_f_steps.append(F.mse_loss(pred_f, tgt).item())

                # Counterfactual: translate full context toward green-room cluster
                ctx_cf  = ctx + delta_hue
                pred_cf = jepa.predict(ctx_cf, act_ctx)[:, -1]
                surp_cf_steps.append(F.mse_loss(pred_cf, tgt).item())

            results.append({
                "teleport_step":           t_tp,
                "surprise_factual":        surp_f_steps[t_tp],
                "surprise_counterfactual": surp_cf_steps[t_tp],
                "surp_fact_steps":         surp_f_steps,
                "surp_cf_steps":           surp_cf_steps,
                # First-frame latent for PCA visualisation
                "z_fact": z[0].cpu().float().numpy(),
                "z_cf":   (z[0] + delta_hue).cpu().float().numpy(),
            })

    print(f"      {len(results)} teleport episodes found")
    return results


# ---------------------------------------------------------------------------
# Stage 4a — Structural invariance
# ---------------------------------------------------------------------------

@torch.no_grad()
def _structural_invariance(jepa, loader, delta_hue, pos_dirs, n_batches=30,
                            mask_teleport=False, tp_bbox=None):
    """Mean absolute change in the position subspace after the hue intervention.

    Invariance Error = mean |z @ pos_dirs.T - z_cf @ pos_dirs.T|

    Near zero means the position dimensions are orthogonal to the hue
    intervention vector (Independent Causal Mechanisms).
    """
    errors = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        pixels = batch["pixels"].to(_DEVICE)
        action = batch["action"].to(_DEVICE)
        if mask_teleport:
            pixels = _mask_tp(pixels, tp_bbox)
        out    = jepa.encode({"pixels": pixels, "action": action})
        B, T, D = out["emb"].shape
        z    = out["emb"].reshape(B * T, D)   # (BT, D)
        z_cf = z + delta_hue                   # (BT, D)

        # Project both onto position directions and measure drift
        pos_fact = z    @ pos_dirs.T           # (BT, 2)
        pos_cf   = z_cf @ pos_dirs.T
        errors.append((pos_fact - pos_cf).abs().mean().item())

    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Stage 4b — AAP consistency advantage
# ---------------------------------------------------------------------------

@torch.no_grad()
def _aap_consistency_advantage(jepa, loader, n_warmup=20, n_eval=50,
                                mask_teleport=False, tp_bbox=None):
    """Measure how much factual evidence improves over a blind (mean) context.

    Advantage = mean_surprise(blind) - mean_surprise(factual)
    Positive advantage signals Ladder 3 behaviour: the world model uses
    real observations to constrain its predictions rather than relying on
    priors alone.
    """
    # Build the "blind" prior: mean embedding over the first n_warmup batches
    emb_buf = []
    for i, batch in enumerate(loader):
        if i >= n_warmup:
            break
        pixels = batch["pixels"].to(_DEVICE)
        action = batch["action"].to(_DEVICE)
        if mask_teleport:
            pixels = _mask_tp(pixels, tp_bbox)
        out    = jepa.encode({"pixels": pixels, "action": action})
        emb_buf.append(out["emb"].cpu())
    mean_ctx = torch.cat(emb_buf, 0).mean(0, keepdim=True).to(_DEVICE)  # (1, T, D)

    s_with, s_without = [], []
    count = 0
    for batch in loader:
        if count >= n_eval:
            break
        pixels  = batch["pixels"].to(_DEVICE)
        action  = batch["action"].to(_DEVICE)
        if mask_teleport:
            pixels = _mask_tp(pixels, tp_bbox)
        out     = jepa.encode({"pixels": pixels, "action": action})
        emb     = out["emb"]      # (B, T, D)
        act_emb = out["act_emb"]
        B       = emb.size(0)

        ctx     = emb[:, :-1]                            # (B, T-1, D)
        act_ctx = act_emb[:, :-1]
        tgt     = emb[:, -1]                             # (B, D)

        # With factual context
        pred_with    = jepa.predict(ctx, act_ctx)[:, -1]
        # With blind (mean) context — actions are still factual
        blind_ctx    = mean_ctx[:, :-1].expand(B, -1, -1)
        pred_without = jepa.predict(blind_ctx, act_ctx)[:, -1]

        s_with.append(F.mse_loss(pred_with,    tgt, reduction="none").mean(-1).cpu())
        s_without.append(F.mse_loss(pred_without, tgt, reduction="none").mean(-1).cpu())
        count += B

    mean_with    = float(torch.cat(s_with).mean())
    mean_without = float(torch.cat(s_without).mean())
    return mean_without - mean_with, mean_with, mean_without


# ---------------------------------------------------------------------------
# Stage 5 — Visualisation
# ---------------------------------------------------------------------------

# Publication-quality colour palette (ColorBrewer-safe, print-friendly)
_C = {
    "blue":   "#2166ac",
    "green":  "#4dac26",
    "red":    "#d6604d",
    "purple": "#762a83",
    "orange": "#e08214",
    "tp":     "#b2182b",
}


def _setup_pub_style():
    """Apply IEEE/NeurIPS-compatible rcParams: serif font, 9 pt, 300 dpi."""
    plt.rcParams.update({
        "font.family":         "serif",
        "font.size":           9,
        "axes.titlesize":      9,
        "axes.labelsize":      9,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "legend.fontsize":     7.5,
        "legend.framealpha":   0.85,
        "legend.edgecolor":    "0.75",
        "legend.handlelength": 1.8,
        "lines.linewidth":     1.4,
        "axes.linewidth":      0.7,
        "xtick.major.width":   0.7,
        "ytick.major.width":   0.7,
        "xtick.major.size":    3.0,
        "ytick.major.size":    3.0,
        "figure.dpi":          150,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.03,
    })


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_fig(fig, out_dir, stem):
    """Save figure as both PDF (for LaTeX inclusion) and PNG (preview)."""
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)


def _save_plots(aap_results, z_all, hue_all, max_deltas, delta_hue, out_dir, suffix=""):
    _setup_pub_style()
    _plot_surprise_over_time(aap_results, out_dir, suffix)
    _plot_latent_pca(aap_results, z_all, hue_all, max_deltas, delta_hue, out_dir, suffix)


def _plot_surprise_over_time(aap_results, out_dir, suffix=""):
    """Per-step factual vs counterfactual surprise, averaged over AAP episodes."""
    if not aap_results:
        return

    n_steps  = len(aap_results[0]["surp_fact_steps"])
    fact_mat = np.array([r["surp_fact_steps"] for r in aap_results])
    cf_mat   = np.array([r["surp_cf_steps"]   for r in aap_results])

    mean_f,  std_f  = fact_mat.mean(0), fact_mat.std(0)
    mean_cf, std_cf = cf_mat.mean(0),   cf_mat.std(0)
    xs = np.arange(n_steps)
    t_tp = int(np.median([r["teleport_step"] for r in aap_results]))

    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    ax.plot(xs, mean_f,  color=_C["blue"],   lw=1.4,
            label="Factual (blue room)")
    ax.fill_between(xs, mean_f - std_f,  mean_f + std_f,
                    alpha=0.18, color=_C["blue"], linewidth=0)

    ax.plot(xs, mean_cf, color=_C["orange"], lw=1.4, linestyle="--",
            label="Counterfactual (hue $\\to$ green)")
    ax.fill_between(xs, mean_cf - std_cf, mean_cf + std_cf,
                    alpha=0.18, color=_C["orange"], linewidth=0)

    ax.axvline(t_tp, color=_C["tp"], linestyle=":", lw=1.0,
               label=f"Teleport ($t={t_tp}$)")

    ax.set_xlabel("Prediction step")
    ax.set_ylabel("MSE (surprise)")
    ax.set_xticks(xs)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=3, fontsize=6.5,
        frameon=False, handletextpad=0.3, columnspacing=0.8,
    )
    _despine(ax)

    _save_fig(fig, out_dir, f"surprise_over_time{suffix}")


def _plot_latent_pca(aap_results, z_all, hue_all, max_deltas, delta_hue, out_dir, suffix=""):
    """2D PCA of latent embeddings with hue-intervention arrows."""
    n_bg = min(3000, len(z_all))
    rng  = np.random.default_rng(0)
    idx  = rng.choice(len(z_all), n_bg, replace=False)
    z_bg, h_bg, d_bg = z_all[idx], hue_all[idx], max_deltas[idx]

    pca = PCA(n_components=2, random_state=42)
    pca.fit(z_bg)
    z2d = pca.transform(z_bg)

    hue_label = (h_bg > 0).astype(int)
    tp_flag   = d_bg > np.percentile(d_bg, 90)

    if aap_results:
        z_fact_np = np.stack([r["z_fact"] for r in aap_results])
        z_cf_np   = np.stack([r["z_cf"]   for r in aap_results])
        zf2d      = pca.transform(z_fact_np)
        zc2d      = pca.transform(z_cf_np)
    else:
        zf2d = zc2d = np.empty((0, 2))

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    for label, colour, name in [
        (0, _C["blue"],  "Blue room"),
        (1, _C["green"], "Green room"),
    ]:
        m = (hue_label == label) & ~tp_flag
        ax.scatter(z2d[m, 0], z2d[m, 1], c=colour, s=4, alpha=0.18,
                   linewidths=0, label=name)

    if tp_flag.any():
        ax.scatter(z2d[tp_flag, 0], z2d[tp_flag, 1],
                   marker="*", s=40, c=_C["tp"], zorder=5,
                   linewidths=0, label="Teleport frame")

    for i in range(len(zf2d)):
        ax.annotate(
            "", xy=(zc2d[i, 0], zc2d[i, 1]), xytext=(zf2d[i, 0], zf2d[i, 1]),
            arrowprops=dict(arrowstyle="-|>", color=_C["purple"],
                            lw=0.8, mutation_scale=5),
        )
    if len(zf2d):
        ax.scatter(zf2d[:, 0], zf2d[:, 1], c=_C["purple"], s=18,
                   zorder=6, label=r"$z_\mathrm{fact}$")
        ax.scatter(zc2d[:, 0], zc2d[:, 1], c=_C["orange"], s=18,
                   marker="D", zorder=6, label=r"$z_\mathrm{cf}$ (hue intervention)")

    var0 = pca.explained_variance_ratio_[0] * 100
    var1 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var0:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({var1:.1f}% var.)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=2, fontsize=6.5, markerscale=1.5,
        frameon=False, handletextpad=0.3, columnspacing=0.8, labelspacing=0.3,
    )
    _despine(ax)

    _save_fig(fig, out_dir, f"latent_pca{suffix}")


# ---------------------------------------------------------------------------
# Reporting and W&B logging
# ---------------------------------------------------------------------------

def _print_report(metrics):
    print("\n" + "=" * 62)
    print("  CAUSAL DISENTANGLEMENT TEST — RESULTS")
    print("=" * 62)
    for k, v in metrics.items():
        print(f"  {k:<44s}: {v:.6f}")
    print()
    r = metrics["surprise_ratio"]
    a = metrics["aap_consistency_advantage"]
    e = metrics["structural_invariance_error"]
    print(f"  Surprise ratio              {r:.4f}")
    print(f"    log only — values near 1.0 suggest Ladder 3 behaviour")
    print(f"  AAP consistency advantage   {a:.4f}")
    print(f"    positive → factual evidence reduces uncertainty (Ladder 3)")
    print(f"  Structural invariance error {e:.4f}")
    print(f"    near 0   → position dims orthogonal to hue (ICM)")
    print("=" * 62)


def _log_to_wandb(metrics, ckpt_path, out_dir, suffix=""):
    import wandb

    run_tag = Path(ckpt_path).parent.name
    run = wandb.init(
        project="lewm-causality",
        entity="paoloai-robomotic",
        name=f"causal_test_{run_tag}{suffix}",
        tags=["causal_test", "aap", "disentanglement"]
              + (["masked_teleport"] if suffix else []),
        config={"checkpoint": str(ckpt_path), "mask_teleport": bool(suffix)},
    )
    payload = {f"causal/{k}": v for k, v in metrics.items()}
    for stem in ("surprise_over_time", "latent_pca"):
        p = out_dir / f"{stem}{suffix}.png"
        if p.exists():
            payload[f"causal/{stem}"] = wandb.Image(str(p))
    wandb.log(payload)
    print(f"\nW&B run  → {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Oracle calibration: provable ceiling/floor reference points for the AAP
Surprise Ratio, using hand-coded predictors on ground-truth features instead
of a JEPA checkpoint. No training, no GPU, no new data collection -- pure
tabular analysis on data already in the training dataset, same cost class as
research/causal_discovery.py.

Two predictors, both reading ground-truth environment features rather than
pixels:

  true_cause_predict(dist_to_pad, teleport_enabled, radius, K)
      Smooth sigmoid decision on the TRUE mechanism only. Never reads hue.

  hue_shortcut_predict(hue_score, K_hue)
      Smooth sigmoid decision on ONLY a continuous hue proxy (green-minus-blue
      channel mean of the first frame), matching the training confound's
      direction exactly. Never reads dist_to_pad or teleport_enabled.

Both are deliberately smooth (sigmoid), not hard 0/1 thresholds. A perfectly
deterministic predictor gives EXACTLY zero squared error whenever it matches
the (binary) target, which makes the surprise ratio degenerate --
surp_fact=0 exactly, so ratio = surp_cf / (0 + eps) is dominated entirely by
the epsilon rather than measuring anything real. This is not a hypothetical
concern: a hard-threshold hue_shortcut_predict(hue) = 1.0 if hue==BLUE else
0.0, scored on episodes filtered to teleported=True (which in the confounded
training dataset are always hue=blue), gives surp_fact=0 EXACTLY, producing a
~1e12 "ratio" that reflects the epsilon, not shortcut reliance. Both
predictors are kept in the same continuous (0,1) regime the real MSE-based
ratio operates in, sidestepping this without inventing an arbitrary
noise-injection hyperparameter.

IMPORTANT, worth stating plainly rather than presenting as a surprising
empirical result: true_cause_predict's ratio is invariant to K BY
CONSTRUCTION, not because of genuine empirical robustness. make_counterfactual
flips ONLY hue -- dist_to_pad and teleport_enabled (the only two things
true_cause_predict reads) are held fixed -- so pred_fact and pred_cf are
IDENTICAL for any K, making surp_fact == surp_cf exactly and ratio ~= 1.0
regardless of steepness. This is the mathematically correct property for a
ceiling reference (a truly hue-blind predictor cannot be surprised by a
hue-only intervention, by definition), but the --k-sweep flag's "robustness
check" is really confirming an algebraic identity, not testing a free
parameter -- unlike hue_shortcut_predict's K_hue, which genuinely changes the
ratio since hue_score DOES change between factual and counterfactual.

Data availability note: glitched_hue_tworoom_half.h5 (the default dataset)
does not store variation.teleport.enabled or variation.background.color
directly -- only the per-step `teleported` (fired) column. For THIS dataset
specifically, teleport_enabled is recoverable exactly because the training
confound is deterministic by construction (blue -> enabled, green ->
disabled), so teleport_enabled == (hue == blue) exactly. This substitution is
dataset-specific: for datasets that store variation.teleport.enabled directly
(e.g. glitched_hue_optionc, glitched_hue_decorrelated), the true stored value
is read instead and does NOT reduce to a function of hue.

Usage:
    python research/oracle_calibration.py --n-episodes 200 --seed 42
    python research/oracle_calibration.py --n-episodes 200 --k-sweep
"""

import argparse
import json
import math
from pathlib import Path

try:
    import hdf5plugin  # noqa: F401 -- registers LZ4/blosc/zstd filters used by swm datasets
except ImportError:
    pass  # Modal image has it; local envs may need `pip install hdf5plugin`

import h5py
import numpy as np

_DATASET_NAME = "glitched_hue_tworoom_half"
_BLUE = np.array([0, 0, 255], dtype=np.uint8)
_GREEN = np.array([0, 180, 0], dtype=np.uint8)
_TP_POS = np.array([56.0, 112.0], dtype=np.float32)   # matches GlitchedHueTwoRoomEnv's init_value
_TP_RADIUS = 10.0                                      # matches GlitchedHueTwoRoomEnv's init_value
_EPS = 1e-12

# Theoretical hue_score for each pure background colour, computed directly
# from the fixed RGB constants used throughout this project (not estimated
# from noisy pixel data) -- the "ground truth" value for the counterfactual
# intervention. Blue and green are NOT symmetric in this colour scheme
# (-255 vs +180), so the counterfactual is not a simple sign flip.
_HUE_SCORE_BLUE = float(_BLUE[1]) - float(_BLUE[2])     # 0 - 255   = -255.0
_HUE_SCORE_GREEN = float(_GREEN[1]) - float(_GREEN[2])  # 180 - 0   = +180.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_K(radius: float, target_prob: float = 0.95) -> float:
    """K such that sigmoid(K * radius) == target_prob.

    sigmoid(K * (radius - dist_to_pad)) always crosses exactly 0.5 at
    dist_to_pad == radius (the boundary), for ANY K -- that part needs no
    calibration. What target_prob calibrates is how SHARP that crossing is:
    at dist_to_pad=0 (right at the pad), the prediction is already
    target_prob; at dist_to_pad=2*radius, it's (1 - target_prob). That's a
    transition spanning about one pad-radius on each side of the boundary,
    matching "crosses 0.5 within one pad-radius of the boundary."
    """
    return math.log(target_prob / (1 - target_prob)) / radius


def _calibrate_K_hue(hue_scores: np.ndarray, target_prob: float = 0.95) -> float:
    """K_hue such that sigmoid(K_hue * median(|hue_score|)) == target_prob.

    The hue-analog of _calibrate_K, using the empirical scale of hue_score
    (no fixed physical radius exists for a colour-channel difference) rather
    than a hardcoded reference distance.
    """
    scale = float(np.median(np.abs(hue_scores)))
    return math.log(target_prob / (1 - target_prob)) / scale


def true_cause_predict(dist_to_pad: float, teleport_enabled: int, radius: float, K: float) -> float:
    """Smooth, continuous decision on the TRUE mechanism only. Never reads hue."""
    if not teleport_enabled:
        return 0.0
    return float(sigmoid(K * (radius - dist_to_pad)))


def hue_shortcut_predict(hue_score: float, K_hue: float) -> float:
    """Reads ONLY a continuous hue proxy -- same sigmoid family as
    true_cause_predict, so both oracles live in the same non-degenerate
    continuous regime. Never reads dist_to_pad or teleport_enabled.

    NEGATIVE sign is deliberate: hue_score is green-minus-blue channel mean,
    so it is NEGATIVE for blue and POSITIVE for green. The shortcut must
    predict HIGH probability for blue (matching the training confound's
    direction: blue -> enabled -> teleport more likely), so a negative
    hue_score needs to map to a HIGH sigmoid output -- hence -K_hue * hue_score,
    not +K_hue * hue_score (which would backwards predict high for green).
    """
    return float(sigmoid(-K_hue * hue_score))


def make_counterfactual(features: dict) -> dict:
    """Pure single-factor do(hue): flip ONLY the observed hue/hue_score,
    holding dist_to_pad and teleport_enabled -- the TRUE causal variables --
    fixed at their factual values.

    This is the guarantee Theme C's latent translation could never make
    (zero leakage into other factors, confirmed there via on-manifold and
    extra-factor-probe checks): it's only possible here because the
    intervention happens directly in interpretable, ground-truth feature
    space rather than in an opaque learned embedding where "hue direction"
    and "everything else" are not perfectly separable.
    """
    cf = dict(features)
    cf["hue"] = 1 - features["hue"]
    cf["hue_score"] = _HUE_SCORE_GREEN if cf["hue"] == 1 else _HUE_SCORE_BLUE
    # dist_to_pad, teleport_enabled: UNCHANGED -- deliberately.
    return cf


def _extract_episode_features(h5_path: str, n_episodes: int, seed: int) -> list:
    """Per usable (teleported) episode, extract ground-truth features at the
    anchor step -- the last pre-teleport frame, matching Theme D's
    `enc_idx - 1` convention (research/theme_d_paired_validation.py), here
    expressed directly in raw step indices since no frameskip windowing or
    model encoding is involved at all.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        ep_len_all = f["ep_len"][:]
        ep_offset_all = f["ep_offset"][:]
        has_tp_enabled = "variation.teleport.enabled" in f
        candidates = np.where(ep_len_all >= 2)[0]
        scan_order = rng.permutation(candidates)

        rows = []
        n_skipped_corrupt = n_no_teleport = 0
        for i in scan_order:
            if len(rows) >= n_episodes:
                break
            off, ln = int(ep_offset_all[i]), int(ep_len_all[i])

            try:
                tp_col = f["teleported"][off: off + ln]
            except OSError:
                n_skipped_corrupt += 1
                continue

            hit = np.where(tp_col)[0]
            if len(hit) == 0:
                n_no_teleport += 1
                continue  # not a teleported episode -- filtered per AAP convention
            t_hit = int(hit[0])
            if t_hit < 1:
                continue  # no pre-teleport frame available

            anchor = t_hit - 1  # Theme D's enc_idx-1 convention, raw-step terms

            try:
                first_frame = f["pixels"][off].astype(np.float32)
                proprio_anchor = f["proprio"][off + anchor]
                tp_enabled_raw = int(f["variation.teleport.enabled"][off]) if has_tp_enabled else None
            except OSError:
                n_skipped_corrupt += 1
                continue

            hue_score = float(first_frame[..., 1].mean() - first_frame[..., 2].mean())
            hue = int(hue_score > 0)  # 1 = green, 0 = blue

            if has_tp_enabled:
                teleport_enabled = tp_enabled_raw
            else:
                # glitched_hue_tworoom_half doesn't store the field directly, but
                # the training confound makes it deterministic: blue -> enabled.
                # Dataset-specific substitution -- see module docstring.
                teleport_enabled = int(hue == 0)

            dist_to_pad = float(np.linalg.norm(proprio_anchor[:2] - _TP_POS))

            rows.append({
                "episode_idx": int(i),
                "hue": hue,
                "hue_score": hue_score,
                "teleport_enabled": teleport_enabled,
                "dist_to_pad": dist_to_pad,
                "teleported": 1.0,  # always 1 -- filtered to teleported episodes
            })

        if n_skipped_corrupt:
            print(f"      (skipped {n_skipped_corrupt} episodes with unreadable HDF5 chunks)")
        print(f"      (skipped {n_no_teleport} non-teleported episodes during scan)")

    return rows


def score(rows: list, K: float, K_hue: float, radius: float) -> dict:
    results = {
        "true_cause": {"ratio": [], "surp_fact": [], "surp_cf": []},
        "hue_shortcut": {"ratio": [], "surp_fact": [], "surp_cf": []},
    }

    for r in rows:
        target = r["teleported"]
        cf = make_counterfactual(r)

        pred_fact_tc = true_cause_predict(r["dist_to_pad"], r["teleport_enabled"], radius, K)
        pred_cf_tc = true_cause_predict(cf["dist_to_pad"], cf["teleport_enabled"], radius, K)
        surp_fact_tc = (pred_fact_tc - target) ** 2
        surp_cf_tc = (pred_cf_tc - target) ** 2
        results["true_cause"]["surp_fact"].append(surp_fact_tc)
        results["true_cause"]["surp_cf"].append(surp_cf_tc)
        results["true_cause"]["ratio"].append(surp_cf_tc / (surp_fact_tc + _EPS))

        pred_fact_hs = hue_shortcut_predict(r["hue_score"], K_hue)
        pred_cf_hs = hue_shortcut_predict(cf["hue_score"], K_hue)
        surp_fact_hs = (pred_fact_hs - target) ** 2
        surp_cf_hs = (pred_cf_hs - target) ** 2
        results["hue_shortcut"]["surp_fact"].append(surp_fact_hs)
        results["hue_shortcut"]["surp_cf"].append(surp_cf_hs)
        results["hue_shortcut"]["ratio"].append(surp_cf_hs / (surp_fact_hs + _EPS))

    return results


def summarize(vals: dict) -> dict:
    surp_fact = np.array(vals["surp_fact"])
    surp_cf = np.array(vals["surp_cf"])
    ratio_per_episode = np.array(vals["ratio"])
    return {
        # eq:aap-ratio convention
        "ratio_of_means": float(surp_cf.mean() / (surp_fact.mean() + _EPS)),
        # eq:per-episode-ratio convention
        "per_episode_mean": float(ratio_per_episode.mean()),
        "per_episode_std": float(ratio_per_episode.std()),
        "n_episodes": len(ratio_per_episode),
        # Absolute surprise values -- these stay well-behaved (surp_fact -> 0,
        # surp_cf bounded near its max) even where the RATIO diverges as K
        # grows, since the ratio is a near-zero-denominator effect, not a
        # change in the counterfactual surprise itself. Report alongside the
        # ratio so a diverging ratio has a stable, concrete quantity next to
        # it rather than looking like an unexplained blowup.
        "surp_fact_mean": float(surp_fact.mean()),
        "surp_cf_mean": float(surp_cf.mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Oracle calibration: ceiling/floor reference points for the AAP surprise ratio"
    )
    parser.add_argument("--dataset-name", default=_DATASET_NAME)
    parser.add_argument("--n-episodes", type=int, default=200,
                         help="Target number of usable (teleported) episodes (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius", type=float, default=_TP_RADIUS,
                         help=f"Teleport pad radius (default: {_TP_RADIUS}, matches the env)")
    parser.add_argument("--radius-k", type=float, default=None,
                         help="Steepness for true_cause_predict's sigmoid. Default: "
                              "auto-calibrated so sigmoid(K*radius)=0.95 (see _calibrate_K).")
    parser.add_argument("--hue-k", type=float, default=None,
                         help="Steepness for hue_shortcut_predict's sigmoid. Default: "
                              "auto-calibrated from the empirical hue_score scale.")
    parser.add_argument("--target-prob", type=float, default=0.95,
                         help="Target probability used by both auto-calibrations (default: 0.95)")
    parser.add_argument("--k-sweep", action="store_true",
                         help="Run both oracles at 0.5x/1x/2x the calibrated K and report "
                              "ratio sensitivity (true_cause is expected to be invariant BY "
                              "CONSTRUCTION -- see module docstring; hue_shortcut is not).")
    parser.add_argument("--out-dir", default=None,
                         help="Directory to write oracle_calibration_results.json (default: cwd)")
    args = parser.parse_args()

    import stable_worldmodel as swm

    h5_path = str(swm.data.utils.get_cache_dir() / f"{args.dataset_name}.h5")
    print(f"Dataset : {h5_path}")
    print(f"[1/3] Extracting {args.n_episodes} teleported-episode ground-truth records ...")
    rows = _extract_episode_features(h5_path, args.n_episodes, args.seed)
    print(f"      {len(rows)} usable (teleported) episodes")
    if not rows:
        print("      ERROR: no usable episodes found")
        return {}

    hue_scores = np.array([r["hue_score"] for r in rows])
    K = args.radius_k if args.radius_k is not None else _calibrate_K(args.radius, args.target_prob)
    K_hue = args.hue_k if args.hue_k is not None else _calibrate_K_hue(hue_scores, args.target_prob)
    print(f"      radius={args.radius}  K={K:.4f} (calibrated at target_prob={args.target_prob})")
    print(f"      K_hue={K_hue:.6f} (calibrated from empirical hue_score scale, "
          f"median|hue_score|={np.median(np.abs(hue_scores)):.2f})")

    print("\n[2/3] Scoring both oracles ...")
    results = score(rows, K, K_hue, args.radius)
    summary = {name: summarize(vals) for name, vals in results.items()}

    print("\n[3/3] Results:")
    print(f"  {'Oracle':<16} {'Ratio-of-means':<16} {'Per-episode mean +/- std':<26} N")
    for name in ["true_cause", "hue_shortcut"]:
        s = summary[name]
        print(f"  {name:<16} {s['ratio_of_means']:<16.4f} "
              f"{s['per_episode_mean']:.4f} +/- {s['per_episode_std']:.4f}"
              f"{'':<10} {s['n_episodes']}")

    k_sweep = None
    if args.k_sweep:
        print("\n[K-sweep] true_cause_predict and hue_shortcut_predict at 0.5x / 1x / 2x calibrated K ...")
        k_sweep = {"true_cause": {}, "hue_shortcut": {}}
        for mult in [0.5, 1.0, 2.0]:
            res_tc = score(rows, K * mult, K_hue, args.radius)
            s_tc = summarize(res_tc["true_cause"])
            k_sweep["true_cause"][f"{mult}x"] = s_tc
            print(f"    true_cause    K={K * mult:.4f} ({mult}x): "
                  f"ratio-of-means={s_tc['ratio_of_means']:.6f}  "
                  f"per-episode={s_tc['per_episode_mean']:.6f} +/- {s_tc['per_episode_std']:.6f}")

            res_hs = score(rows, K, K_hue * mult, args.radius)
            s_hs = summarize(res_hs["hue_shortcut"])
            k_sweep["hue_shortcut"][f"{mult}x"] = s_hs
            print(f"    hue_shortcut  K_hue={K_hue * mult:.6f} ({mult}x): "
                  f"ratio-of-means={s_hs['ratio_of_means']:.6f}  "
                  f"per-episode={s_hs['per_episode_mean']:.6f} +/- {s_hs['per_episode_std']:.6f}  "
                  f"[surp_fact={s_hs['surp_fact_mean']:.6f}  surp_cf={s_hs['surp_cf_mean']:.6f}]")
        print("\n    hue_shortcut's ratio diverges as K_hue grows because surp_fact -> 0 while")
        print("    surp_cf stays bounded near its max -- a near-zero-denominator effect on the")
        print("    RATIO, not a change in how surprised the shortcut actually is by the")
        print("    counterfactual. The absolute surp_fact/surp_cf values above are the stable,")
        print("    K_hue-independent-in-spirit quantities; the ratio itself has no principled")
        print("    K_hue to anchor it (unlike true_cause_predict's physical pad radius), so no")
        print("    single point estimate should be reported as THE hue-shortcut floor.")
        print("\n    NOTE: true_cause_predict's ratio is invariant to K BY CONSTRUCTION, not")
        print("    empirical robustness -- make_counterfactual never changes dist_to_pad or")
        print("    teleport_enabled (the only two things true_cause_predict reads), so")
        print("    pred_fact == pred_cf identically for any K, and surp_fact == surp_cf")
        print("    exactly. This sweep confirms that algebraic identity numerically; unlike")
        print("    hue_shortcut_predict's K_hue sweep, it is not testing a genuinely free")
        print("    parameter, since hue_score DOES change between factual and counterfactual.")

    out = {
        "dataset": args.dataset_name,
        "n_episodes": len(rows),
        "radius": args.radius,
        "K": K,
        "K_hue": K_hue,
        "target_prob": args.target_prob,
        "true_cause": summary["true_cause"],
        "hue_shortcut": summary["hue_shortcut"],
        "k_sweep": k_sweep,
    }
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    out_path = out_dir / "oracle_calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults -> {out_path}")

    return out


if __name__ == "__main__":
    main()

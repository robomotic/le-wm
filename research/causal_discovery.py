#!/usr/bin/env python3
"""
Observational causal discovery baseline (J8ik's uncovered half of the related-work gap).

J8ik's complaint: ke2021causalmbrl and ahmed2020causalworld are cited narratively
(Abstract, Intro, Related Work) but never actually run against anything in this
paper. Integrating either codebase directly is a multi-week project (both are
full external benchmark suites that don't drop into stable-worldmodel or operate
on a single JEPA checkpoint) -- not what the paper needs to make its point.

What this script does instead: implements the STYLE of evaluation ahmed2020's
causal_mbrl represents -- recovering a causal graph from purely observational
data -- using a standard, well-established algorithm (PC, via causal-learn) on
tabular variables already sitting in the existing training dataset
(glitched_hue_tworoom_half). No GPU, no Modal, no new environment rollouts.

(ahmed2020causalworld's OTHER half -- testing generalization under a
programmatically modified SCM -- is what Option C / Option C+A already do,
just not framed that way in Related Work; see reports/testladder.md's Option C
section. That connection is a paper-text edit, not a new experiment.)

Variables (one row per episode, extracted directly from the raw HDF5 -- not via
swm.data.HDF5Dataset windowing, since we want whole-episode aggregates, not
model-ready context windows):

  hue          -- binary, green-minus-blue channel mean of the first frame
                  (same sign convention as _extract_probe_data's hue_score in
                  research/glitched_hue_experiment.py: positive = green).
  teleported   -- binary, whether the teleport event fired at any point in
                  the episode (any() over the per-step `teleported` column).
  start_room   -- binary, which side of the central wall the agent starts on
                  (proprio[0] >= WALL_CENTER=112).
  ep_len_bin   -- 3-level categorical, tercile-binned episode length.
  dist_final   -- 3-level categorical, tercile-binned final distance_to_target.

hue and teleport-gating are deterministically linked in this dataset (blue ->
enabled, green -> disabled) but "teleported" (whether the event actually FIRES
in a given episode) also depends on the policy's trajectory -- so the
hue/teleported statistical association is strong but imperfect, exactly the
kind of confound a purely observational method has to untangle without ever
seeing the intervened (reversed-confound) condition.

Usage:
    python research/causal_discovery.py
    python research/causal_discovery.py --n-episodes 5000 --seed 42
"""

import argparse
import json
from pathlib import Path

try:
    import hdf5plugin  # noqa: F401 -- registers LZ4/blosc/zstd filters used by swm datasets
except ImportError:
    pass  # Modal image has it; local envs may need `pip install hdf5plugin`

import h5py
import numpy as np
import pandas as pd

_DATASET_NAME = "glitched_hue_tworoom_half"
_WALL_CENTER = 112.0


def _extract_episode_table(h5_path: str, n_episodes: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        ep_len_all = f["ep_len"][:]
        ep_offset_all = f["ep_offset"][:]
        n_total = len(ep_len_all)

        candidates = np.where(ep_len_all >= 2)[0]
        n_sample = min(n_episodes, len(candidates))
        idx = rng.choice(candidates, size=n_sample, replace=False)
        idx.sort()  # sequential-ish h5py access is faster than scattered fancy indexing

        rows = []
        n_skipped = 0
        for i in idx:
            off, ln = int(ep_offset_all[i]), int(ep_len_all[i])

            try:
                first_frame = f["pixels"][off].astype(np.float32)  # (H, W, 3) uint8 -> float
                teleported_col = f["teleported"][off: off + ln]
                proprio0 = f["proprio"][off]
                dist_final = float(f["distance_to_target"][off + ln - 1])
            except OSError:
                # A small number of chunks in this local HDF5 copy are unreadable
                # (a pre-existing, unrelated storage/decompression issue -- not
                # specific to this script). Skip rather than crash the whole run.
                n_skipped += 1
                continue

            hue_score = float(first_frame[..., 1].mean() - first_frame[..., 2].mean())
            hue = int(hue_score > 0)  # 1 = green, 0 = blue
            teleported = bool(teleported_col.any())
            start_room = int(float(proprio0[0]) >= _WALL_CENTER)

            rows.append({
                "episode_idx": int(i),
                "ep_len": ln,
                "hue": hue,
                "teleported": int(teleported),
                "start_room": start_room,
                "dist_final": dist_final,
            })

        if n_skipped:
            print(f"      (skipped {n_skipped} episodes with unreadable HDF5 chunks)")

    df = pd.DataFrame(rows)
    df["ep_len_bin"] = pd.qcut(df["ep_len"], q=3, labels=False, duplicates="drop")
    df["dist_final_bin"] = pd.qcut(df["dist_final"], q=3, labels=False, duplicates="drop")
    return df


def _edge_info(cg, name_a: str, name_b: str) -> dict:
    """Return structured info about the edge (if any) between two named nodes.

    Returns a dict with keys:
      status: "none" | "undirected" | "directed" | "ambiguous"
      cause, effect: node names when status == "directed", else None
      description: human-readable summary line
    """
    from causallearn.graph.Endpoint import Endpoint

    nodes = {n.get_name(): n for n in cg.G.get_nodes()}
    a, b = nodes[name_a], nodes[name_b]
    edge = cg.G.get_edge(a, b)
    if edge is None:
        return {
            "status": "none", "cause": None, "effect": None,
            "description": f"{name_a} -- {name_b}: NO EDGE (independence test found no significant association)",
        }

    e1, e2 = edge.get_endpoint1(), edge.get_endpoint2()
    n1, n2 = edge.get_node1().get_name(), edge.get_node2().get_name()

    def _mark(e):
        if e == Endpoint.TAIL:
            return "-"
        if e == Endpoint.ARROW:
            return ">"
        if e == Endpoint.CIRCLE:
            return "o"
        return "?"

    left_mark, right_mark = _mark(e1), _mark(e2)
    arrow = f"{n1} {'<' if left_mark == '>' else '-'}--{right_mark} {n2}"

    if e1 == Endpoint.TAIL and e2 == Endpoint.TAIL:
        status, cause, effect = "undirected", None, None
        verdict = "UNDIRECTED -- association detected, but PC cannot determine which variable causes which"
    elif e1 == Endpoint.ARROW and e2 == Endpoint.TAIL:
        status, cause, effect = "directed", n2, n1
        verdict = f"DIRECTED: {n2} -> {n1}"
    elif e1 == Endpoint.TAIL and e2 == Endpoint.ARROW:
        status, cause, effect = "directed", n1, n2
        verdict = f"DIRECTED: {n1} -> {n2}"
    else:
        status, cause, effect = "ambiguous", None, None
        verdict = f"PARTIALLY ORIENTED / AMBIGUOUS ({left_mark}, {right_mark}) -- not a clean directed or undirected edge"

    return {
        "status": status, "cause": cause, "effect": effect,
        "description": f"{name_a} -- {name_b}: {arrow}  =>  {verdict}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Observational causal discovery (PC algorithm) on the training dataset"
    )
    parser.add_argument("--n-episodes", type=int, default=5000,
                         help="Episodes to sample for the discovery table (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05,
                         help="PC algorithm significance level (default: 0.05)")
    parser.add_argument("--dataset-name", default=_DATASET_NAME)
    parser.add_argument("--out-dir", default=None,
                         help="Directory to write causal_discovery_results.json (default: cwd)")
    args = parser.parse_args()

    import stable_worldmodel as swm
    from causallearn.search.ConstraintBased.PC import pc

    h5_path = str(swm.data.utils.get_cache_dir() / f"{args.dataset_name}.h5")
    print(f"Dataset : {h5_path}")
    print(f"[1/3] Extracting {args.n_episodes} episode-level observational records ...")
    df = _extract_episode_table(h5_path, args.n_episodes, args.seed)
    print(f"      {len(df)} episodes extracted")
    print(f"      hue distribution: {df['hue'].value_counts().to_dict()}")
    print(f"      teleported distribution: {df['teleported'].value_counts().to_dict()}")
    print(f"      P(teleported=1 | hue=blue=0) = {df.loc[df.hue == 0, 'teleported'].mean():.4f}")
    print(f"      P(teleported=1 | hue=green=1) = {df.loc[df.hue == 1, 'teleported'].mean():.4f}")

    cols = ["hue", "teleported", "start_room", "ep_len_bin", "dist_final_bin"]
    data = df[cols].to_numpy().astype(float)

    print(f"\n[2/3] Running PC algorithm (alpha={args.alpha}, indep_test=chisq) ...")
    cg = pc(data, alpha=args.alpha, indep_test="chisq", node_names=cols, show_progress=False)

    print("\n[3/3] Discovered graph:")
    print(cg.G)

    hue_teleported = _edge_info(cg, "hue", "teleported")
    print(f"\n  KEY EDGE: {hue_teleported['description']}")

    all_edges = []
    for name_a in cols:
        for name_b in cols:
            if name_a >= name_b:
                continue
            info = _edge_info(cg, name_a, name_b)
            if info["status"] != "none":
                all_edges.append(info["description"])
                print(f"    {info['description']}")

    results = {
        "dataset": args.dataset_name,
        "n_episodes": len(df),
        "alpha": args.alpha,
        "variables": cols,
        "p_teleported_given_blue": float(df.loc[df.hue == 0, "teleported"].mean()),
        "p_teleported_given_green": float(df.loc[df.hue == 1, "teleported"].mean()),
        "hue_teleported_edge": hue_teleported["description"],
        "hue_teleported_status": hue_teleported["status"],
        "hue_teleported_cause": hue_teleported["cause"],
        "hue_teleported_effect": hue_teleported["effect"],
        "all_edges": all_edges,
        "adjacency_matrix": cg.G.graph.tolist(),
    }

    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    out_path = out_dir / "causal_discovery_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out_path}")

    print("\n" + "=" * 66)
    print("  VERDICT")
    print("=" * 66)
    if hue_teleported["status"] == "undirected":
        print("  PC recovers the hue<->teleported ASSOCIATION but cannot ORIENT it --")
        print("  purely observational discovery cannot resolve which variable causes")
        print("  which, unlike the interventional AAP ratio / Theme D ground-truth")
        print("  counterfactual, which directly test the causal direction by intervention.")
    elif hue_teleported["status"] == "directed":
        print(f"  PC ORIENTS the edge: {hue_teleported['cause']} -> {hue_teleported['effect']}.")
        if hue_teleported["cause"] != "hue":
            print(f"  This gets it BACKWARDS: the true generative process sets room hue first")
            print(f"  (via teleport.enabled), then episode rollout determines whether the")
            print(f"  teleport event fires -- not the other way around. A confident, wrong")
            print(f"  causal claim from purely observational data is the worse of the two")
            print(f"  anticipated outcomes.")
        else:
            print("  PC attributes hue as the cause of teleportation, consistent with the")
            print("  training-time confound direction -- though hue is itself a proxy for")
            print("  the upstream teleport.enabled setting, not a true root cause either.")
    else:
        print(f"  {hue_teleported['description']}")
    print("=" * 66)

    return results


if __name__ == "__main__":
    main()

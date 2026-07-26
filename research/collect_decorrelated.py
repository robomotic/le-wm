#!/usr/bin/env python3
"""
Collect a decorrelated-confound dataset for GlitchedHueTwoRoom (Theme G positive control).

Training confound:  blue room <-> teleport enabled,  green room <-> teleport disabled
Option C reversal:  blue room <-> teleport disabled, green room <-> teleport enabled
This script:         hue and teleport.enabled drawn INDEPENDENTLY -- no confound at all.

Reuses research/collect_option_c.py's confound-override mechanism verbatim (same
`variation_values` trick collect_theme_d_paired.py's pairing relies on): a single
`options["variation_values"]` dict sets `background.color` and `teleport.enabled`
together for a `world.record_dataset()` call. Option C changes only WHICH pairing
is assigned to the two batches it collects (reversed vs. training); this script
instead assigns each pairing independently, per CHUNK of episodes rather than
per single episode.

Why chunks, not literal per-episode coin flips: `variation_values` is fixed for
an entire `record_dataset()` call, so genuine per-episode independence would mean
one `record_dataset()` call per episode (20,000 individual H5-append calls for
the full run). Chunking (default 20 episodes per independent coin flip) keeps the
call count practical while still driving the DATASET-level hue/teleport
correlation to ~0 -- what actually matters for a "no confound to latch onto"
positive control, since the model only ever sees shuffled minibatches across many
chunks during training, never a single chunk in isolation. Env-stepping compute
(the actual bottleneck) is identical either way; only per-call H5 bookkeeping
overhead scales with chunk count, and that's small relative to rollout time.

Usage (run via Modal -- local env not required):
    python research/collect_decorrelated.py --n-episodes 200 --seed 3072                # smoke test
    python research/collect_decorrelated.py --n-episodes 20000 --seed 3072              # full run
    python research/collect_decorrelated.py --only-check --n-check-episodes 2000        # re-check an existing dataset
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

_DATASET_NAME = "glitched_hue_decorrelated"
_BLUE  = np.array([0,   0, 255], dtype=np.uint8)
_GREEN = np.array([0, 180,   0], dtype=np.uint8)

_IO_CHUNK = 256   # frames per merge I/O chunk -- keeps peak RAM well under 1 GB
_DEFAULT_EPISODE_CHUNK = 20  # episodes per independent (hue, teleport_enabled) coin flip


def _collect_chunk(world, dataset_name, n_episodes, bg_color, tp_enabled, seed):
    options = {
        "variation": ["background.color", "teleport.enabled"],
        "variation_values": {
            "background.color": bg_color,
            "teleport.enabled": tp_enabled,
        },
    }
    world.record_dataset(dataset_name, episodes=n_episodes, seed=seed, options=options)


def _merge_many(paths: list, out: Path) -> None:
    """Concatenate N identically-structured HDF5 datasets into one, chunked I/O."""
    try:
        import hdf5plugin
        _ = hdf5plugin
    except ImportError:
        pass

    with h5py.File(out, "w") as fo:
        lens = []
        for p in paths:
            with h5py.File(p, "r") as f:
                lens.append(f["ep_len"][:])
        ep_lens = np.concatenate(lens)
        n_steps_per_file = [int(l.sum()) for l in lens]
        n_total = sum(n_steps_per_file)

        offsets = []
        running = 0
        for l in lens:
            offsets.append(running + np.concatenate([[0], l[:-1].cumsum()]).astype(np.int64))
            running += int(l.sum())
        ep_offsets = np.concatenate(offsets)

        fo.create_dataset("ep_len", data=ep_lens)
        fo.create_dataset("ep_offset", data=ep_offsets)

        with h5py.File(paths[0], "r") as f0:
            data_keys = [k for k in f0.keys() if k not in ("ep_len", "ep_offset")]

        for k in data_keys:
            with h5py.File(paths[0], "r") as f0:
                dtype = f0[k].dtype
                sample_shape = f0[k].shape[1:]

            if dtype == object:
                all_vals = []
                for p in paths:
                    with h5py.File(p, "r") as f:
                        all_vals.append(f[k][:])
                fo.create_dataset(k, data=np.concatenate(all_vals))
                continue

            out_shape = (n_total,) + sample_shape
            fo.create_dataset(k, shape=out_shape, dtype=dtype)

            global_ptr = 0
            for p, n_steps in zip(paths, n_steps_per_file):
                with h5py.File(p, "r") as f:
                    ds = f[k]
                    for start in range(0, n_steps, _IO_CHUNK):
                        end = min(start + _IO_CHUNK, n_steps)
                        fo[k][global_ptr + start: global_ptr + end] = ds[start:end]
                global_ptr += n_steps

            print(f"  merged {k}: {out_shape} {dtype}")

    print(f"Merged {len(paths)} chunks -> {out}  ({len(ep_lens)} episodes, {n_total} steps)")


def collect(n_episodes: int, seed: int = 42, chunk_size: int = _DEFAULT_EPISODE_CHUNK,
            dataset_name: str = _DATASET_NAME) -> Path:
    import stable_worldmodel as swm
    from stable_worldmodel.envs.glitched_hue_two_room import GlitchedHueExpertPolicy

    cache = swm.data.utils.get_cache_dir()
    world = swm.World("swm/GlitchedHueTwoRoom-v1", num_envs=4, image_shape=(224, 224))
    # NOT swm.policy.RandomPolicy (what collect_option_c.py uses) -- RandomPolicy gives
    # ~3x longer episodes than the actual training recipe (confirmed empirically: mean
    # ep_len ~88 vs. ~32 for glitched_hue_tworoom_half), which would conflate "confound
    # removed" with "action/behavior distribution also changed" in the eventual
    # baseline comparison. GlitchedHueExpertPolicy with these exact hyperparameters
    # matches scripts/data/config/glitched_hue_half.yaml -- the recipe that actually
    # built glitched_hue_tworoom_half -- keeping confound removal as the ONLY variable
    # that differs from baseline. It needs no adaptation for decorrelation: it already
    # reads whatever teleport.enabled value is active for the current episode and
    # reacts accordingly, independent of how that value was assigned.
    world.set_policy(GlitchedHueExpertPolicy(action_noise=0.5, action_repeat_prob=0.05, seed=seed))

    rng = np.random.default_rng(seed)
    n_chunks = -(-n_episodes // chunk_size)  # ceil division
    chunk_paths = []
    flips = []  # (hue_choice, tp_choice) per chunk, for a quick design-level summary

    collected = 0
    for c in range(n_chunks):
        this_n = min(chunk_size, n_episodes - collected)
        hue_choice = int(rng.integers(0, 2))       # 0 = blue, 1 = green
        tp_choice = int(rng.integers(0, 2))        # 0 = disabled, 1 = enabled -- INDEPENDENT draw
        bg = _BLUE if hue_choice == 0 else _GREEN
        chunk_name = f"{dataset_name}_chunk{c:05d}"
        chunk_seed = seed + c * 1_000_003  # large odd stride, well-separated per-chunk seeds

        print(f"  [{c + 1}/{n_chunks}] {this_n} episodes  "
              f"hue={'green' if hue_choice else 'blue'}  teleport={'ON' if tp_choice else 'OFF'}")
        _collect_chunk(world, chunk_name, this_n, bg, tp_choice, chunk_seed)
        chunk_paths.append(cache / f"{chunk_name}.h5")
        flips.append((hue_choice, tp_choice))
        collected += this_n

    world.close()

    n_hue_green = sum(h for h, _ in flips)
    n_tp_on = sum(t for _, t in flips)
    both = sum(1 for h, t in flips if h == 1 and t == 1)
    print(f"\n  Chunk-level design summary ({n_chunks} chunks):")
    print(f"    green chunks: {n_hue_green}/{n_chunks}   teleport-ON chunks: {n_tp_on}/{n_chunks}")
    print(f"    (green AND ON): {both}/{n_chunks}  "
          f"(expected ~{n_chunks * n_hue_green * n_tp_on / (n_chunks ** 2):.1f} under independence)")

    out = cache / f"{dataset_name}.h5"
    print(f"\nMerging {n_chunks} chunks into {out} ...")
    _merge_many(chunk_paths, out)
    return out


def check_decorrelation(dataset_name: str = _DATASET_NAME, n_episodes: int = 2000, seed: int = 0) -> dict:
    """Sanity-check marginals and hue/teleport correlation using the SAME per-episode
    extraction as research/causal_discovery.py, for a consistent, independently-derived
    measurement (not just the chunk-level design summary printed during collection)."""
    import stable_worldmodel as swm
    from research.causal_discovery import _extract_episode_table

    h5_path = str(swm.data.utils.get_cache_dir() / f"{dataset_name}.h5")
    df = _extract_episode_table(h5_path, n_episodes, seed)

    hue_mean = df["hue"].mean()
    tp_enabled_proxy = df["teleported"].mean()  # NOTE: "teleported" firing != "teleport.enabled";
    # see caveat below. This still gives a directionally useful correlation check.
    corr = df["hue"].corr(df["teleported"])

    print(f"\n  Decorrelation check (N={len(df)} episodes, independently re-extracted from data):")
    print(f"    P(hue=green)        = {hue_mean:.4f}  (expect ~0.5)")
    print(f"    P(teleported=True)  = {tp_enabled_proxy:.4f}")
    print(f"    corr(hue, teleported) = {corr:.4f}  (expect ~0, well below Option C's/training's "
          f"near-perfect correlation)")

    return {
        "n_episodes": len(df),
        "p_hue_green": float(hue_mean),
        "p_teleported": float(tp_enabled_proxy),
        "corr_hue_teleported": float(corr),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect the decorrelated-confound dataset (Theme G)")
    parser.add_argument("--n-episodes", type=int, default=20000,
                        help="Total episodes (default: 20000, matching glitched_hue_tworoom_half's scale)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=_DEFAULT_EPISODE_CHUNK,
                        help=f"Episodes per independent (hue, teleport_enabled) coin flip "
                             f"(default: {_DEFAULT_EPISODE_CHUNK})")
    parser.add_argument("--dataset-name", default=_DATASET_NAME)
    parser.add_argument("--only-check", action="store_true",
                        help="Skip collection; just run the decorrelation sanity check on an existing dataset")
    parser.add_argument("--n-check-episodes", type=int, default=2000,
                        help="Episodes to sample for the post-hoc decorrelation check (default: 2000)")
    args = parser.parse_args()

    if not args.only_check:
        out = collect(args.n_episodes, args.seed, args.chunk_size, args.dataset_name)
        print(f"\n✅ Dataset saved to: {out}")

    check_decorrelation(args.dataset_name, args.n_check_episodes, seed=0)


if __name__ == "__main__":
    main()

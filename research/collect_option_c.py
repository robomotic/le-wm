#!/usr/bin/env python3
"""
Collect Option C reversed-confound dataset for GlitchedHueTwoRoom.

Confound reversed vs training:
  Training:  blue room ↔ teleport enabled,   green room ↔ teleport disabled
  Option C:  blue room ↔ teleport disabled,  green room ↔ teleport enabled

Two fixed-option collection passes are merged into one HDF5 so the AAP
pipeline sees a 50/50 mix of both conditions in a single dataset.

Usage (run via Modal — local env not required):
    python research/collect_option_c.py --n-episodes 5000
    python research/collect_option_c.py --only-merge   # re-run merge if half-files exist
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

_DATASET_NAME = "glitched_hue_optionc"
_BLUE  = np.array([0,   0, 255], dtype=np.uint8)   # teleport DISABLED (reversed)
_GREEN = np.array([0, 180,   0], dtype=np.uint8)   # teleport ENABLED  (reversed)

_CHUNK = 256   # frames per I/O chunk — keeps peak RAM well under 1 GB


def _collect_half(
    world,
    dataset_name: str,
    n_episodes: int,
    bg_color: np.ndarray,
    tp_enabled: int,
    seed: int,
) -> None:
    options = {
        "variation": ["background.color", "teleport.enabled"],
        "variation_values": {
            "background.color": bg_color,
            "teleport.enabled": tp_enabled,
        },
    }
    world.record_dataset(dataset_name, episodes=n_episodes, seed=seed, options=options)


def _merge(path_a: Path, path_b: Path, out: Path) -> None:
    """Concatenate two identically-structured HDF5 datasets into one.

    Uses chunked I/O so that large datasets (e.g. pixels) are never fully
    loaded into RAM — only _CHUNK frames at a time are held in memory.
    """
    with h5py.File(path_a, "r") as fa, \
         h5py.File(path_b, "r") as fb, \
         h5py.File(out, "w") as fo:

        data_keys = [k for k in fa.keys() if k not in ("ep_len", "ep_offset")]
        len_a = fa["ep_len"][:]
        len_b = fb["ep_len"][:]
        off_a = fa["ep_offset"][:]
        n_steps_a = int(len_a.sum())
        n_steps_b = int(len_b.sum())
        n_total   = n_steps_a + n_steps_b

        ep_lens = np.concatenate([len_a, len_b])
        ep_offsets = np.concatenate([
            off_a,
            (n_steps_a + np.concatenate([[0], len_b[:-1].cumsum()])).astype(off_a.dtype),
        ])
        fo.create_dataset("ep_len",    data=ep_lens)
        fo.create_dataset("ep_offset", data=ep_offsets)

        for k in data_keys:
            ds_a = fa[k]
            ds_b = fb[k]

            # Object-dtype datasets (e.g. env_name strings): load all at once
            # since they're tiny and np.concatenate handles them fine.
            if ds_a.dtype == object:
                fo.create_dataset(k, data=np.concatenate([ds_a[:], ds_b[:]]))
                continue

            # Numeric datasets: create empty then fill in chunks.
            out_shape = (n_total,) + ds_a.shape[1:]
            fo.create_dataset(k, shape=out_shape, dtype=ds_a.dtype)

            for start in range(0, n_steps_a, _CHUNK):
                end = min(start + _CHUNK, n_steps_a)
                fo[k][start:end] = ds_a[start:end]

            for start in range(0, n_steps_b, _CHUNK):
                end = min(start + _CHUNK, n_steps_b)
                fo[k][n_steps_a + start:n_steps_a + end] = ds_b[start:end]

            print(f"  merged {k}: {out_shape} {ds_a.dtype}")

    n_ep = len(ep_lens)
    print(f"Merged → {out}  ({n_ep} episodes, {n_total} steps)")


def collect(n_episodes: int, seed: int = 42) -> Path:
    import stable_worldmodel as swm

    cache = swm.data.utils.get_cache_dir()
    world = swm.World("swm/GlitchedHueTwoRoom-v1", num_envs=4, image_shape=(224, 224))
    world.set_policy(swm.policy.RandomPolicy(seed=seed))

    half = n_episodes // 2

    name_blue  = f"{_DATASET_NAME}_blue"
    name_green = f"{_DATASET_NAME}_green"

    print(f"[1/3] Collecting {half} blue/disabled episodes  (teleport OFF, bg=blue) ...")
    _collect_half(world, name_blue,  half, _BLUE,  0, seed)

    print(f"[2/3] Collecting {half} green/enabled episodes  (teleport ON,  bg=green) ...")
    _collect_half(world, name_green, half, _GREEN, 1, seed + half)

    world.close()

    out = _merge_halves(cache)
    return out


def _merge_halves(cache: Path) -> Path:
    name_blue  = f"{_DATASET_NAME}_blue"
    name_green = f"{_DATASET_NAME}_green"
    out = cache / f"{_DATASET_NAME}.h5"
    print(f"[3/3] Merging into {_DATASET_NAME}.h5 ...")
    _merge(cache / f"{name_blue}.h5", cache / f"{name_green}.h5", out)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Collect Option C reversed-confound dataset"
    )
    parser.add_argument("--n-episodes", type=int, default=5000,
                        help="Total episodes (split 50/50 blue/green, default 5000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-merge", action="store_true",
                        help="Skip collection; re-run merge from existing half-files")
    args = parser.parse_args()

    if args.only_merge:
        import stable_worldmodel as swm
        cache = swm.data.utils.get_cache_dir()
        out = _merge_halves(cache)
    else:
        out = collect(args.n_episodes, args.seed)

    print(f"\n✅ Dataset saved to: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Collect the Theme D paired factual/counterfactual dataset for GlitchedHueTwoRoom.

CMTV's remaining Critical review item asks for paired factual/counterfactual
trajectories generated directly in the environment, so the model's prediction
can be compared against a genuine ground-truth counterfactual encoder output
instead of the linear-translation z_cf = z_fact + delta_hue used everywhere
else in research/glitched_hue_experiment.py (see reports/testladder.md,
"Theme D").

Confound (matches the TRAINING distribution, i.e. glitched_hue_tworoom_half --
NOT Option C's reversed pairing in collect_option_c.py):
  Factual:        blue room  <-> teleport ENABLED
  Counterfactual: green room <-> teleport DISABLED

Pairing mechanism (verified against stable_worldmodel/spaces.py):
  - Dict.update(keys) (invoked by env.reset(seed=..., options={'variation': keys})
    resamples in the space's fixed canonical `sampling_order`, filtered by the
    *set* of `keys` passed -- not by list order. So two reset() calls with the
    same seed and the same set of `variation` keys draw agent/target start
    positions in bit-identical order, regardless of what's also overridden via
    `variation_values` afterward (a pure, non-RNG assignment). background.color
    only affects rendering (TwoRoomEnv._render_frame); it never touches physics.
  - GlitchedHueTwoRoomEnv.step() runs base physics first, then applies the
    teleport mirror-jump conditionally on teleport.enabled -- so two envs
    seeded identically and driven by an identical action array are physically
    identical up to the step where the factual env's teleport actually fires.
  - Actions are NOT re-drawn per condition: they are pre-generated ONCE by
    running the real GlitchedHueExpertPolicy open-loop on the factual env only
    (in-distribution with the actual training data: action_noise=0.5,
    action_repeat_prob=0.05, per scripts/data/config/glitched_hue_half.yaml in
    the stable-worldmodel-causality fork), then replayed verbatim via
    env.step(action[t]) on the counterfactual env. Re-invoking the policy on
    the cf env would silently break pairing, since
    GlitchedHueExpertPolicy._compute_waypoint() branches its steering on
    teleport.enabled.

Output schema matches stable_worldmodel.World.record_dataset's HDF5 layout
exactly (RAW per-step resolution -- unstrided -- for pixels/action/proprio/
teleported/step_idx/distance_to_target, plus ep_offset/ep_len index arrays),
so the paired files can be loaded with swm.data.HDF5Dataset directly and fed
through the exact same frameskip-windowing/action-reshape/transform pipeline
research/glitched_hue_experiment.py already uses. Two Theme-D-only per-episode
arrays are added: `teleport_step` (ground truth, raw-step index of the first
post-teleport frame; -1 if teleport never fired) and `pair_valid` (bool,
result of the pre-teleport proprio-identity sanity check).

Usage (run via Modal -- local env not required):
    python research/collect_theme_d_paired.py --n-episodes 200
    python research/collect_theme_d_paired.py --n-episodes 5 --episode-len 100   # smoke test
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

_DATASET_NAME = "glitched_hue_theme_d"

# Training confound -- matches glitched_hue_tworoom_half, NOT Option C's reversal.
_BLUE  = np.array([0,   0, 255], dtype=np.uint8)   # teleport ENABLED
_GREEN = np.array([0, 180,   0], dtype=np.uint8)   # teleport DISABLED

# Teleport zone -- matches scripts/data/config/glitched_hue.yaml (inherited by
# the "half" variant used to build glitched_hue_tworoom_half) and
# GlitchedHueTwoRoomEnv's own init_value defaults, so paired episodes use the
# same teleport geometry as the baseline training/eval dataset.
_TP_POS    = np.array([56.0, 112.0], dtype=np.float32)
_TP_RADIUS = np.array([10.0], dtype=np.float32)
_TP_COLOR  = np.array([255, 255, 255], dtype=np.uint8)

# Expert-policy hyperparameters -- matches glitched_hue_half.yaml exactly, so
# actions stay in-distribution with the checkpoint's actual training data.
_ACTION_NOISE = 0.5
_ACTION_REPEAT_PROB = 0.05

_VARIATION_KEYS = ["agent.position", "target.position", "background.color", "teleport.enabled"]

_STEP_KEYS = ["pixels", "action", "proprio", "teleported", "step_idx", "distance_to_target"]


def _options(bg_color: np.ndarray, tp_enabled: int) -> dict:
    return {
        "variation": list(_VARIATION_KEYS),
        "variation_values": {
            "background.color": bg_color,
            "teleport.enabled": tp_enabled,
            "teleport.position": _TP_POS,
            "teleport.radius": _TP_RADIUS,
            "teleport.color": _TP_COLOR,
        },
    }


def _make_envs():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401 -- registers swm/* env ids

    env_fact = gym.make("swm/GlitchedHueTwoRoom-v1", render_mode="rgb_array")
    env_cf   = gym.make("swm/GlitchedHueTwoRoom-v1", render_mode="rgb_array")
    return env_fact, env_cf


def _make_expert_policy(seed: int):
    from stable_worldmodel.envs.glitched_hue_two_room import GlitchedHueExpertPolicy

    return GlitchedHueExpertPolicy(
        action_noise=_ACTION_NOISE,
        action_repeat_prob=_ACTION_REPEAT_PROB,
        seed=seed,
    )


def _generate_actions(env_fact, info_f, episode_len, seed, use_expert, policy=None):
    """Return a (episode_len, 2) float32 action array, generated ONCE.

    If use_expert, drives the real GlitchedHueExpertPolicy open-loop on
    env_fact (the only env it ever touches) so actions stay in-distribution
    with the checkpoint's actual training data. Otherwise draws i.i.d. uniform
    actions from an independent RNG stream (simpler fallback/ablation).
    """
    actions = np.zeros((episode_len, 2), dtype=np.float32)
    if not use_expert:
        rng = np.random.default_rng(seed + 1_000_000)  # independent of env RNG
        return rng.uniform(-1.0, 1.0, size=(episode_len, 2)).astype(np.float32)

    policy.set_env(env_fact)
    policy.set_seed(seed + 1_000_000)  # independent of env RNG, deterministic per episode
    info = info_f
    for t in range(episode_len):
        a = policy.get_action(info)
        actions[t] = a
        _, _, _, _, info = env_fact.step(a)
    # NOTE: this loop consumes env_fact's physics -- caller must reset env_fact
    # again before the real recorded rollout below (see _roll_one_pair).
    return actions


def _record(buf: dict, pixels: np.ndarray, info: dict, step_idx: int, action: np.ndarray) -> None:
    buf["pixels"].append(pixels)
    buf["action"].append(action.astype(np.float32))
    buf["proprio"].append(np.asarray(info["proprio"], dtype=np.float32))
    buf["teleported"].append(bool(info.get("teleported", False)))
    buf["step_idx"].append(step_idx)
    buf["distance_to_target"].append(float(info["distance_to_target"]))


def _roll_one_pair(env_fact, env_cf, seed: int, episode_len: int, use_expert: bool, policy=None) -> dict:
    """Roll a single paired factual/counterfactual episode.

    Two full resets of env_fact happen when use_expert=True: one throwaway
    pass to record the expert-policy action sequence (since the policy must
    actually observe env_fact's evolving state to compute waypoints), and one
    real pass -- with the SAME seed, so the RNG-stream-identity guarantee for
    agent/target start positions still holds -- to record the actual factual
    rollout. The action array from the throwaway pass is replayed verbatim in
    both the real factual and counterfactual rollouts.
    """
    if use_expert:
        obs_f0, info_f0 = env_fact.reset(seed=seed, options=_options(_BLUE, 1))
        action_seq = _generate_actions(env_fact, info_f0, episode_len, seed, True, policy)
    else:
        action_seq = _generate_actions(env_fact, None, episode_len, seed, False)

    # Real, recorded pass -- identical seed, so identical start state.
    obs_f, info_f = env_fact.reset(seed=seed, options=_options(_BLUE, 1))
    obs_c, info_c = env_cf.reset(seed=seed, options=_options(_GREEN, 0))

    agent_start_fact = np.asarray(info_f["proprio"], dtype=np.float32).copy()
    agent_start_cf   = np.asarray(info_c["proprio"], dtype=np.float32).copy()

    buf_f = {k: [] for k in _STEP_KEYS}
    buf_c = {k: [] for k in _STEP_KEYS}

    dummy_action = action_seq[0] if episode_len > 0 else np.zeros(2, dtype=np.float32)
    _record(buf_f, env_fact.render(), info_f, 0, dummy_action)
    _record(buf_c, env_cf.render(),   info_c, 0, dummy_action)

    teleport_step = -1  # ground truth: buffer index of the FIRST post-teleport frame
    for t in range(episode_len):
        a = action_seq[t]
        obs_f, _, _, _, info_f = env_fact.step(a)
        obs_c, _, _, _, info_c = env_cf.step(a)

        if info_f.get("teleported") and teleport_step == -1:
            # Buffer index of the frame recorded just below. Verified against
            # pixel frame-to-frame diffs: the visual jump lands exactly at this
            # index (env.render() reads self.agent_position AFTER step()'s
            # internal mutation). NOTE: info_f["proprio"] itself lags by one
            # step here -- GlitchedHueTwoRoomEnv.step() calls super().step()
            # (which builds `info` including proprio) BEFORE applying the
            # teleport mirror, so the proprio value in THIS info dict is still
            # pre-mirror; the mirrored proprio only appears in info at t+2's
            # buffer entry. This is an upstream env quirk, not a bug here --
            # it doesn't affect anything since only pixels get encoded, and
            # the pre-teleport identity check below only compares indices
            # strictly before teleport_step (unaffected either way).
            teleport_step = t + 1

        next_action = action_seq[t + 1] if t + 1 < episode_len else action_seq[-1]
        _record(buf_f, env_fact.render(), info_f, t + 1, next_action)
        _record(buf_c, env_cf.render(),   info_c, t + 1, next_action)

    proprio_f = np.stack(buf_f["proprio"])
    proprio_c = np.stack(buf_c["proprio"])
    end = teleport_step if teleport_step >= 0 else len(proprio_f)
    pair_valid = bool(np.allclose(proprio_f[:end], proprio_c[:end], atol=1e-4))
    if not pair_valid:
        diverge_at = int(np.argmax(np.any(np.abs(proprio_f[:end] - proprio_c[:end]) > 1e-4, axis=-1)))
        print(f"    WARNING: pre-teleport divergence at buffer index {diverge_at} (expected identity up to {end})")

    return {
        "ep_fact": buf_f,
        "ep_cf": buf_c,
        "teleport_step": teleport_step,
        "pair_valid": pair_valid,
        "agent_start_fact": agent_start_fact,
        "agent_start_cf": agent_start_cf,
    }


class _EpisodeWriter:
    """Minimal resizable-HDF5 writer mirroring World._init_h5_datasets/_write_episode."""

    def __init__(self, path: Path):
        try:
            import hdf5plugin
            self._compression = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        except ImportError:
            self._compression = None
        self.path = path
        self.f = h5py.File(str(path), "w", libver="latest")
        self._initialized = False
        self.f.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)
        self.f.create_dataset("ep_len",    shape=(0,), maxshape=(None,), dtype=np.int32)
        self._global_ptr = 0

    def _init_datasets(self, sample_ep: dict) -> None:
        for key, data_list in sample_ep.items():
            sample = np.array(data_list[0])
            shape = (0,) + sample.shape
            maxshape = (None,) + sample.shape
            if sample.ndim >= 2:  # pixels
                chunks = (100,) + sample.shape
                compression = self._compression
            else:
                chunks = (1000,) + sample.shape
                compression = None
            self.f.create_dataset(key, shape=shape, maxshape=maxshape, dtype=sample.dtype,
                                   chunks=chunks, compression=compression)
        self._initialized = True

    def write_episode(self, ep_buf: dict) -> int:
        if not self._initialized:
            self._init_datasets(ep_buf)
        ep_len = len(ep_buf["step_idx"])
        for key, data_list in ep_buf.items():
            ds = self.f[key]
            arr = np.array(data_list)
            curr = ds.shape[0]
            ds.resize(curr + ep_len, axis=0)
            ds[curr:] = arr
        idx = self.f["ep_offset"].shape[0]
        self.f["ep_offset"].resize(idx + 1, axis=0)
        self.f["ep_len"].resize(idx + 1, axis=0)
        self.f["ep_offset"][idx] = self._global_ptr
        self.f["ep_len"][idx] = ep_len
        self._global_ptr += ep_len
        self.f.flush()
        return ep_len

    def write_episode_meta(self, key: str, value, dtype) -> None:
        if key not in self.f:
            self.f.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype)
        ds = self.f[key]
        idx = ds.shape[0]
        ds.resize(idx + 1, axis=0)
        ds[idx] = value

    def close(self):
        self.f.close()


def collect(n_episodes: int, episode_len: int, seed: int, dataset_name: str,
            use_expert: bool = True, max_attempts_factor: int = 20) -> tuple[Path, Path]:
    """Collect `n_episodes` USABLE (teleported) paired episodes.

    Only ~20% of randomly-sampled agent/target placements put the target in
    the other room, so the expert policy actually needs the teleport shortcut
    (confirmed empirically: 6/30 at default settings). Attempting is cheap
    relative to writing, so episodes without a teleport event are discarded
    rather than written -- this keeps the paired dataset lean and lets
    --n-episodes mean "usable episodes for the AAP comparison," matching how
    research/glitched_hue_experiment.py's --n-aap-episodes is interpreted.
    """
    import stable_worldmodel as swm

    cache = swm.data.utils.get_cache_dir()
    fact_path = cache / f"{dataset_name}_fact.h5"
    cf_path   = cache / f"{dataset_name}_cf.h5"

    env_fact, env_cf = _make_envs()
    policy = _make_expert_policy(seed=seed) if use_expert else None

    w_fact = _EpisodeWriter(fact_path)
    w_cf   = _EpisodeWriter(cf_path)

    n_written = 0
    n_valid = 0
    n_attempted = 0
    max_attempts = n_episodes * max_attempts_factor
    while n_written < n_episodes and n_attempted < max_attempts:
        ep_seed = seed + n_attempted
        n_attempted += 1
        result = _roll_one_pair(env_fact, env_cf, ep_seed, episode_len, use_expert, policy)

        if result["teleport_step"] < 0:
            continue  # discard: no teleport event, nothing to compare downstream

        w_fact.write_episode(result["ep_fact"])
        w_cf.write_episode(result["ep_cf"])
        for w in (w_fact, w_cf):
            w.write_episode_meta("teleport_step", result["teleport_step"], np.int32)
            w.write_episode_meta("pair_valid", result["pair_valid"], bool)

        n_written += 1
        n_valid += int(result["pair_valid"])

        if n_written % 20 == 0 or n_written == n_episodes:
            print(f"  [{n_written}/{n_episodes} usable, {n_attempted} attempted] valid_pairs={n_valid}")

    w_fact.close()
    w_cf.close()
    env_fact.close()
    env_cf.close()

    if n_written < n_episodes:
        print(f"\n  WARNING: only found {n_written}/{n_episodes} teleported episodes "
              f"within {max_attempts} attempts. Consider raising --max-attempts-factor "
              f"or checking the teleport rate with a smaller --n-episodes run first.")

    pass_rate = n_valid / n_written if n_written else 0.0
    print(f"\nCollected {n_written} usable (teleported) episodes from {n_attempted} attempts "
          f"({n_written / n_attempted:.1%} hit rate).")
    print(f"Pre-teleport identity check: {n_valid}/{n_written} passed ({pass_rate:.1%})")
    if pass_rate < 0.95:
        print("  WARNING: pass rate below 95% -- inspect divergence warnings above before")
        print("  trusting downstream Theme D results. See reports/testladder.md Theme D risks.")

    return fact_path, cf_path


def main():
    parser = argparse.ArgumentParser(
        description="Collect the Theme D paired factual/counterfactual dataset"
    )
    parser.add_argument("--n-episodes", type=int, default=200,
                         help="Number of USABLE (teleported) paired episodes to collect "
                              "(default: 200) -- episodes without a teleport event are "
                              "discarded and don't count toward this total")
    parser.add_argument("--max-attempts-factor", type=int, default=20,
                         help="Give up after n-episodes * this many attempts (default: 20)")
    parser.add_argument("--episode-len", type=int, default=100,
                         help="Raw env steps per episode (default: 100, matches "
                              "world.max_episode_steps in glitched_hue.yaml)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Base seed; episode i uses seed+i (default: 42)")
    parser.add_argument("--dataset-name", default=_DATASET_NAME,
                         help="Output dataset name prefix (default: glitched_hue_theme_d)")
    parser.add_argument("--random-actions", action="store_true",
                         help="Use i.i.d. uniform actions instead of the real "
                              "GlitchedHueExpertPolicy (simpler fallback; actions "
                              "will be out-of-distribution vs. the training data)")
    args = parser.parse_args()

    fact_path, cf_path = collect(
        n_episodes=args.n_episodes,
        episode_len=args.episode_len,
        seed=args.seed,
        dataset_name=args.dataset_name,
        use_expert=not args.random_actions,
        max_attempts_factor=args.max_attempts_factor,
    )
    print(f"\n✅ Factual dataset:        {fact_path}")
    print(f"✅ Counterfactual dataset: {cf_path}")


if __name__ == "__main__":
    main()

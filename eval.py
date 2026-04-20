import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def _write_breakdown(f, label, episode_successes, mask):
    n = int(mask.sum())
    if n == 0:
        f.write(f"  {label}: n/a (0 samples)\n")
        return
    k = int(episode_successes[mask].sum())
    f.write(f"  {label}: {k}/{n} ({100. * k / n:.1f}%)\n")


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row's episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    # Patch world.step to accumulate per-episode teleport usage during the rollout.
    # episode_successes[i] aligns with random_episode_indices[i] (same env ordering).
    num_envs = len(eval_episodes)
    teleported_any = np.zeros(num_envs, dtype=bool)
    _orig_step = world.step

    def _tracking_step():
        _orig_step()
        tp = world.infos.get('teleported')
        if tp is not None:
            tp_arr = np.asarray(tp, dtype=bool).reshape(num_envs, -1)
            teleported_any[:] |= tp_arr.any(axis=1)

    world.step = _tracking_step

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    end_time = time.time()

    world.step = _orig_step

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")

    # --- breakdown file ---
    episode_successes = np.asarray(metrics['episode_successes'], dtype=bool)
    row_data = dataset.get_row_data(random_episode_indices)

    def _col(key):
        v = row_data.get(key)
        return np.asarray(v) if v is not None else None

    bg_colors   = _col('variation.background.color')   # (N,3) uint8
    teleport_en = _col('variation.teleport.enabled')   # (N,)  int
    agent_pos   = _col('variation.agent.position')     # (N,2) float
    target_pos  = _col('variation.target.position')    # (N,2) float
    wall_axes   = _col('variation.wall.axis')          # (N,)  int  (may be absent)

    stem = Path(cfg.output.filename).stem
    suffix = Path(cfg.output.filename).suffix
    breakdown_path = results_path.parent / f"{stem}_breakdown{suffix}"

    with breakdown_path.open("a") as f:
        f.write("\n")

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== BREAKDOWN ====\n")

        # overall
        _write_breakdown(f, "overall", episode_successes, np.ones(num_envs, dtype=bool))

        # by hue (dominant green vs blue channel in background color)
        f.write("-- by hue --\n")
        if bg_colors is not None:
            green_mask = np.all(bg_colors == [  0, 180,   0], axis=1)
            blue_mask  = np.all(bg_colors == [  0,   0, 255], axis=1)
            _write_breakdown(f, "green", episode_successes, green_mask)
            _write_breakdown(f, "blue",  episode_successes, blue_mask)
        else:
            f.write("  # variation.background.color not in dataset\n")

        # by teleport pixel present/absent
        f.write("-- teleport pixel --\n")
        if teleport_en is not None:
            _write_breakdown(f, "absent",  episode_successes, teleport_en == 0)
            _write_breakdown(f, "present", episode_successes, teleport_en == 1)
        else:
            f.write("  # variation.teleport.enabled not in dataset\n")

        # by door crossing required (agent and goal on different sides of the wall)
        f.write("-- door crossing --\n")
        if agent_pos is not None and target_pos is not None:
            # wall.axis=1 → vertical wall, compare x (dim 0); axis=0 → horizontal, compare y (dim 1)
            axes = wall_axes if wall_axes is not None else np.ones(num_envs, dtype=int)
            dims = np.where(np.asarray(axes) == 1, 0, 1)
            a_coord = agent_pos[np.arange(num_envs), dims]
            g_coord = target_pos[np.arange(num_envs), dims]
            cross_needed = (a_coord - 112.0) * (g_coord - 112.0) < 0
            _write_breakdown(f, "same room",         episode_successes, ~cross_needed)
            _write_breakdown(f, "crossing required", episode_successes,  cross_needed)
        else:
            f.write("  # variation.agent/target.position not in dataset\n")

        # by teleport pixel actually used during rollout
        f.write("-- teleport used --\n")
        _write_breakdown(f, "not used", episode_successes, ~teleported_any)
        _write_breakdown(f, "used",     episode_successes,  teleported_any)

        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()

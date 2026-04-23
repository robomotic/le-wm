"""Modal app for running LeWM training and evaluation on cloud GPUs.

The dataset is pre-loaded in the 'swm-cache' Modal volume (priamai-team workspace).

Usage
-----
# Training (100 epochs, logs to W&B lewm-causality / paoloai-robomotic)
modal run modaldotcom/app.py --do-train

# Training with overrides
modal run modaldotcom/app.py --do-train --max-epochs 50 --no-wandb

# Option B — train with teleport patch masking (p=0.5)
modal run modaldotcom/app.py --do-train --mask-teleport-prob 0.5

# Evaluation  (use the job_id printed at the end of training)
modal run modaldotcom/app.py --do-eval --policy <job_id>/lewm_epoch_100

# Smoke test (1 epoch, no W&B)
modal run modaldotcom/app.py --do-train --max-epochs 1 --no-wandb

# Causal test (Step 3 — AAP pipeline, logs causal/* metrics + plots to W&B)
modal run modaldotcom/app.py --do-causal-test --policy <job_id>/lewm_epoch_50
"""

import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("le-wm-causality")

# ---------------------------------------------------------------------------
# Volume  (already exists — created by the data-collection pipeline)
# ---------------------------------------------------------------------------

CACHE_DIR = "/root/.stable_worldmodel"
volume = modal.Volume.from_name("swm-cache")   # priamai-team workspace

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

wandb_secret = modal.Secret.from_name("wandb")   # must contain WANDB_API_KEY

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

# Resolve repo root relative to this file so add_local_dir works whether
# invoked from the repo root or from the modaldotcom/ subdirectory.
REPO_ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "libgl1",
        "libegl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "ffmpeg",
    )
    # CUDA-enabled PyTorch (cu121 wheels are compatible with Modal's A10G drivers)
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Main dependency: causality fork of stable-worldmodel (pulls stable-pretraining too)
    .pip_install(
        "stable-worldmodel[train,env] @ git+https://github.com/epokhcs/stable-worldmodel.git@causality",
        # stable-pretraining's loose constraint resolves to datasets==1.1.1 which
        # lacks the `datasets.config` submodule required at import time. Pin >=2.0.
        "datasets>=2.0",
        "wandb",
        "huggingface_hub",
        "scikit-learn",
        "matplotlib",
    )
    # Copy the local le-wm repo (train.py, eval.py, jepa.py, module.py,
    # utils.py, config/) into the container image at build time.
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/workspace",
        copy=True,
        ignore=[
            ".venv",
            ".git",
            "outputs",      # Hydra output directories
            "wandb",        # W&B local run artefacts
            "__pycache__",
            "modaldotcom",  # avoid embedding this file inside itself
        ],
    )
    .workdir("/workspace")
)

# ---------------------------------------------------------------------------
# Common environment variables
# ---------------------------------------------------------------------------

ENV = {
    "STABLEWM_HOME": CACHE_DIR,
    "MUJOCO_GL": "egl",   # headless OpenGL for eval environments
    "MODAL_VOLUME_NAME": "swm-cache",
}

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: volume},
    secrets=[wandb_secret],
    env=ENV,
    timeout=86400,   # 24 h ceiling — full 100-epoch run fits comfortably
)
def train(
    data: str = "glitched_hue_tworoom",
    max_epochs: int = 100,
    wandb_enabled: bool = True,
    mask_teleport_prob: float = 0.0,
) -> str:
    """Run train.py on a cloud A10G and persist checkpoints to the volume.

    Returns the Hydra job ID so you can pass it to evaluate() as the policy prefix.
    """
    import glob
    import os

    import time
    run_ts = str(int(time.time()))

    cmd = [
        "python", "train.py",
        f"data={data}",
        f"trainer.max_epochs={max_epochs}",
        f"wandb.enabled={'True' if wandb_enabled else 'False'}",
        f"subdir=ts_{run_ts}",
    ]
    if mask_teleport_prob > 0.0:
        cmd += [
            "augmentation.teleport_patch_mask.enabled=True",
            f"augmentation.teleport_patch_mask.mask_probability={mask_teleport_prob}",
        ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace")

    # Persist all new checkpoint files to the volume before the container exits.
    volume.commit()

    # Find the checkpoint written in this run and report the job_id.
    # train.py saves to $STABLEWM_HOME/<hydra_job_id>/lewm_epoch_*_object.ckpt
    ckpts = sorted(
        glob.glob(f"{CACHE_DIR}/**/*_object.ckpt", recursive=True),
        key=os.path.getmtime,
    )
    if ckpts:
        # e.g. /root/.stable_worldmodel/2024-01-01/0/lewm_epoch_100_object.ckpt
        # policy path = everything between CACHE_DIR and _object.ckpt
        latest = ckpts[-1]
        policy_path = latest.removeprefix(CACHE_DIR + "/").removesuffix("_object.ckpt")
        print(f"\n✅ Training complete.")
        print(f"   Checkpoint : {latest}")
        print(f"   Policy arg : {policy_path}")
        print(f"\nTo evaluate, run:")
        print(f"   modal run modaldotcom/app.py --eval --policy {policy_path}")
        return policy_path
    else:
        print("⚠️  No checkpoint found after training.")
        return ""


# ---------------------------------------------------------------------------
# Dataset inspection function (no GPU — reads HDF5 from volume)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={CACHE_DIR: volume},
    env=ENV,
    timeout=120,
)
def inspect_dataset(name: str = "glitched_hue_tworoom_half") -> None:
    """Print HDF5 dataset columns, shapes, dtypes, and a sample of variation fields.

    Args:
        name: Dataset name (without .h5 extension) relative to STABLEWM_HOME.
    """
    import h5py
    import numpy as np

    path = f"{CACHE_DIR}/{name}.h5"
    print(f"\nInspecting: {path}\n{'='*60}")

    with h5py.File(path, "r") as f:
        ep_len = f["ep_len"][:]
        print(f"Episodes : {len(ep_len)}")
        print(f"Steps    : {int(ep_len.sum())}")
        print(f"Ep length: {int(ep_len.min())} – {int(ep_len.max())}")
        print(f"\n{'Column':<45} {'Shape':<25} {'Dtype'}")
        print("-" * 80)
        for k in sorted(f.keys()):
            if k in ("ep_len", "ep_offset"):
                continue
            ds = f[k]
            print(f"{k:<45} {str(ds.shape):<25} {ds.dtype}")

        # Sample first episode of all variation.* columns to show range
        print(f"\n{'='*60}")
        print("First-episode sample of variation.* columns:")
        ep0_start = int(f["ep_offset"][0])
        ep0_end = ep0_start + int(ep_len[0])
        for k in sorted(f.keys()):
            if not k.startswith("variation."):
                continue
            data = f[k][ep0_start:ep0_end]
            unique = np.unique(data.reshape(len(data), -1), axis=0)
            print(f"  {k}: first_ep unique values = {unique[:5].tolist()}"
                  f"{'...' if len(unique) > 5 else ''}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Option C — reversed-confound data collection (no GPU)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={CACHE_DIR: volume},
    env=ENV,
    timeout=7200,   # 2 h ceiling for 5000 episodes
)
def collect_optionc(n_episodes: int = 5000, seed: int = 42) -> str:
    """Collect the Option C reversed-confound dataset on a cloud CPU worker.

    Runs two fixed-option passes (blue/disabled + green/enabled) via
    World.record_dataset(), then merges them into one HDF5 file.

    Args:
        n_episodes: Total episodes (split 50/50 between the two conditions).
        seed: RNG seed for reproducibility.

    Returns:
        Absolute path of the merged HDF5 file on the volume.
    """
    import subprocess
    cmd = [
        "python", "research/collect_option_c.py",
        f"--n-episodes={n_episodes}",
        f"--seed={seed}",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace")
    volume.commit()
    out = f"{CACHE_DIR}/glitched_hue_optionc.h5"
    print(f"\n✅ Dataset written to volume: {out}")
    return out


# ---------------------------------------------------------------------------
# Stats function  (no GPU — pure WandB API query)
# ---------------------------------------------------------------------------

WANDB_ENTITY  = "paoloai-robomotic"
WANDB_PROJECT = "lewm-causality"

@app.function(
    image=image,
    secrets=[wandb_secret],
    timeout=120,
)
def stats(run_id: str = "") -> None:
    """Print key training stats from WandB for a finished (or crashed) run.

    Args:
        run_id: WandB run ID (the Hydra job subdir printed at training start).
                Leave blank to use the most recent run in the project.
    """
    import wandb

    api = wandb.Api()

    if run_id:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    else:
        runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            order="-created_at",
            per_page=1,
        )
        run = next(iter(runs))

    print(f"\n{'='*60}")
    print(f"Run      : {run.id}  ({run.name})")
    print(f"State    : {run.state}")
    print(f"Epochs   : {int(run.summary.get('trainer/global_step', 0))} steps  |  "
          f"last epoch logged: {int(run.summary.get('epoch', -1)) + 1}")
    print(f"Runtime  : {run.summary.get('_runtime', 0) / 3600:.2f} h")
    print(f"{'='*60}")

    # Loss keys as logged by Lightning via stable-pretraining (fit/validate prefixes)
    summary = dict(run.summary)
    loss_items = {k: v for k, v in summary.items()
                  if "loss" in k.lower() and isinstance(v, (int, float))}

    print("Last-epoch losses:")
    if loss_items:
        for k, v in sorted(loss_items.items()):
            print(f"  {k:<40} {v:.6f}")
    else:
        print("  (no loss keys found — dumping all numeric summary keys)")
        for k, v in sorted(summary.items()):
            if isinstance(v, (int, float)) and not k.startswith("_"):
                print(f"  {k:<40} {v}")

    # Best val loss epoch from full history
    for val_key in ("validate/loss_epoch", "val/loss", "val/loss_epoch"):
        try:
            hist = run.history(keys=[val_key, "epoch"], pandas=True)
            if not hist.empty and val_key in hist.columns:
                best_row = hist.dropna(subset=[val_key]).loc[
                    lambda df: df[val_key].idxmin()
                ]
                print(f"\nBest {val_key} : {best_row[val_key]:.6f}  "
                      f"at epoch {int(best_row.get('epoch', -1)) + 1}")
                break
        except Exception:
            pass

    print(f"\nWandB URL: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run.id}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: volume},
    secrets=[wandb_secret],
    env=ENV,
    timeout=7200,   # 2 h — 50 eval episodes with CEM planning
)
def evaluate(
    policy: str,
    config_name: str = "glitched_hue_tworoom",
) -> None:
    """Run eval.py on a cloud A10G using a checkpoint from the volume.

    Args:
        policy:      Checkpoint path relative to STABLEWM_HOME, without the
                     '_object.ckpt' suffix.  Example: '2024-01-01/0/lewm_epoch_100'
        config_name: Hydra eval config name (without .yaml). Defaults to
                     'glitched_hue_tworoom' which lives in config/eval/.
    """
    cmd = [
        "python", "eval.py",
        f"--config-name={config_name}.yaml",
        f"policy={policy}",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace")

    # Persist results file to the volume.
    volume.commit()
    print(f"\n✅ Evaluation complete. Results written to volume (swm-cache).")


# ---------------------------------------------------------------------------
# Causal test function (Step 3 — AAP disentanglement pipeline)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: volume},
    secrets=[wandb_secret],
    env=ENV,
    timeout=3600,   # 1 h — probe training + AAP rollout well within budget
)
def causal_test(
    policy: str,
    no_wandb: bool = False,
    mask_teleport: bool = False,
    dataset_name: str = "glitched_hue_tworoom_half",
) -> str:
    """Run research/glitched_hue_experiment.py on a cloud A10G (Step 3 of runme.md).

    Executes five stages: trajectory encoding, position+hue probe training,
    AAP cycle (Abduction-Action-Prediction), structural invariance check,
    and surprise-ratio verdict.  Writes causal_test_results.json plus two
    diagnostic plots to the volume alongside the checkpoint.

    Args:
        policy:    Checkpoint path relative to STABLEWM_HOME, without the
                   '_object.ckpt' suffix.
                   Example: '2024-01-01/0/lewm_epoch_50'
        no_wandb:  If True, skip W&B logging (dry run).

    Returns:
        Absolute path of the JSON results file written to the volume.
    """
    import os

    ckpt_path = f"{CACHE_DIR}/{policy}_object.ckpt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found on volume: {ckpt_path}\n"
            "Verify the policy path and that the volume is mounted."
        )

    cmd = ["python", "research/glitched_hue_experiment.py", ckpt_path]
    if no_wandb:
        cmd.append("--no-wandb")
    if mask_teleport:
        cmd.append("--mask-teleport")
    if dataset_name != "glitched_hue_tworoom_half":
        cmd += ["--dataset-name", dataset_name]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace")

    volume.commit()

    ds_suffix = f"_{dataset_name}" if dataset_name != "glitched_hue_tworoom_half" else ""
    suffix = ("_masked" if mask_teleport else "") + ds_suffix
    results_file = f"{CACHE_DIR}/{os.path.dirname(policy)}/causal_test{suffix}_results.json"
    print(f"\n✅ Causal test complete. Results: {results_file}")
    return results_file


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    do_train: bool = False,
    do_eval: bool = False,
    do_stats: bool = False,
    do_causal_test: bool = False,
    do_inspect: bool = False,
    do_collect_optionc: bool = False,
    data: str = "glitched_hue_tworoom",
    max_epochs: int = 100,
    policy: str = "",
    config_name: str = "glitched_hue_tworoom",
    run_id: str = "",
    no_wandb: bool = False,
    dataset_name: str = "glitched_hue_tworoom_half",
    mask_causal_test: bool = False,
    mask_teleport_prob: float = 0.0,
    optionc_episodes: int = 5000,
) -> None:
    """Orchestrate training and/or evaluation on Modal.

    Examples
    --------
    # Full training run (100 epochs, W&B enabled)
    modal run modaldotcom/app.py --do-train

    # Smoke test — 1 epoch, no W&B
    modal run modaldotcom/app.py --do-train --max-epochs 1 --no-wandb

    # Evaluation with a known checkpoint
    modal run modaldotcom/app.py --do-eval --policy 2024-01-01/0/lewm_epoch_100

    # Causal disentanglement test (AAP pipeline)
    modal run modaldotcom/app.py --do-causal-test --policy 2024-01-01/0/lewm_epoch_50

    # Option A — Ladder 3 test: mask the teleport pixel patch before every encode
    modal run modaldotcom/app.py --do-causal-test --policy lewm_epoch_50 --mask-causal-test

    # Option B — retrain with teleport patch masking (p=0.5) then run masked causal test
    modal run modaldotcom/app.py --do-train --mask-teleport-prob 0.5
    modal run modaldotcom/app.py --do-causal-test --policy <new_job_id>/lewm_epoch_100 --mask-causal-test

    # Option C — collect reversed-confound dataset then run causal test on it
    modal run modaldotcom/app.py --do-collect-optionc
    modal run modaldotcom/app.py --do-inspect --dataset-name glitched_hue_optionc
    modal run modaldotcom/app.py --do-causal-test --policy lewm_epoch_50 --dataset-name glitched_hue_optionc
    modal run modaldotcom/app.py --do-causal-test --policy lewm_epoch_50 --dataset-name glitched_hue_optionc --mask-causal-test

    # Inspect dataset columns and variation fields
    modal run modaldotcom/app.py --do-inspect
    modal run modaldotcom/app.py --do-inspect --dataset-name glitched_hue_tworoom

    # Stats for the most recent WandB run
    modal run modaldotcom/app.py --do-stats

    # Stats for a specific run
    modal run modaldotcom/app.py --do-stats --run-id 80aovwgh
    """
    if not any([do_train, do_eval, do_stats, do_causal_test, do_inspect, do_collect_optionc]):
        print("Nothing to do. Pass --do-train, --do-eval, --do-stats, --do-causal-test, --do-inspect, and/or --do-collect-optionc.")
        return

    policy_path = policy
    if do_train:
        policy_path = train.remote(
            data=data,
            max_epochs=max_epochs,
            wandb_enabled=not no_wandb,
            mask_teleport_prob=mask_teleport_prob,
        )
        # If the user also requested eval in the same invocation, chain it.
        if do_eval and not policy and policy_path:
            policy = policy_path

    if do_eval:
        if not policy:
            print("--do-eval requires --policy <path>. Example:")
            print("  modal run modaldotcom/app.py --do-eval --policy <job_id>/lewm_epoch_100")
            return
        evaluate.remote(policy=policy, config_name=config_name)

    if do_stats:
        stats.remote(run_id=run_id)

    if do_collect_optionc:
        out = collect_optionc.remote(n_episodes=optionc_episodes)
        print(f"Option C dataset on volume: {out}")

    if do_causal_test:
        if not policy:
            print("--do-causal-test requires --policy <path>. Example:")
            print("  modal run modaldotcom/app.py --do-causal-test --policy <job_id>/lewm_epoch_50")
            return
        results_file = causal_test.remote(
            policy=policy,
            no_wandb=no_wandb,
            mask_teleport=mask_causal_test,
            dataset_name=dataset_name,
        )
        print(f"Results file on volume: {results_file}")

    if do_inspect:
        inspect_dataset.remote(name=dataset_name)

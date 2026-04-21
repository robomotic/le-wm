"""Modal app for running LeWM training and evaluation on cloud GPUs.

The dataset is pre-loaded in the 'swm-cache' Modal volume (priamai-team workspace).

Usage
-----
# Training (100 epochs, logs to W&B lewm-causality / paoloai-robomotic)
modal run modaldotcom/app.py --do-train

# Training with overrides
modal run modaldotcom/app.py --do-train --max-epochs 50 --no-wandb

# Evaluation  (use the job_id printed at the end of training)
modal run modaldotcom/app.py --do-eval --policy <job_id>/lewm_epoch_100

# Smoke test (1 epoch, no W&B)
modal run modaldotcom/app.py --do-train --max-epochs 1 --no-wandb
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
) -> str:
    """Run train.py on a cloud A10G and persist checkpoints to the volume.

    Returns the Hydra job ID so you can pass it to evaluate() as the policy prefix.
    """
    import glob
    import os

    cmd = [
        "python", "train.py",
        f"data={data}",
        f"trainer.max_epochs={max_epochs}",
        f"wandb.enabled={'True' if wandb_enabled else 'False'}",
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
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    do_train: bool = False,
    do_eval: bool = False,
    do_stats: bool = False,
    data: str = "glitched_hue_tworoom",
    max_epochs: int = 100,
    policy: str = "",
    config_name: str = "glitched_hue_tworoom",
    run_id: str = "",
    no_wandb: bool = False,
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

    # Stats for the most recent WandB run
    modal run modaldotcom/app.py --do-stats

    # Stats for a specific run
    modal run modaldotcom/app.py --do-stats --run-id 80aovwgh
    """
    if not do_train and not do_eval and not do_stats:
        print("Nothing to do. Pass --do-train, --do-eval, and/or --do-stats.")
        return

    policy_path = policy
    if do_train:
        policy_path = train.remote(
            data=data,
            max_epochs=max_epochs,
            wandb_enabled=not no_wandb,
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

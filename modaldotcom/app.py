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
    data: str = "glitched_hue_tworoom",
    max_epochs: int = 100,
    policy: str = "",
    config_name: str = "glitched_hue_tworoom",
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
    """
    if not do_train and not do_eval:
        print("Nothing to do. Pass --do-train and/or --do-eval.")
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

"""Modal app for running LeWM training and evaluation on cloud GPUs.

The GlitchedHueTwoRoom dataset lives in the 'swm-cache' Modal volume
(priamai-team workspace).  All causal ladder experiments are run via
the --do-causal-test flag; see the runbook below.

Causal Ladder Runbook
---------------------
Baseline — train the default model, then run the AAP causal test:

    modal run modaldotcom/app.py --do-train
    modal run modaldotcom/app.py --do-causal-test --policy <job_id>/lewm_epoch_50

Option A — test-time pixel masking (Ladder 1 → 2 boundary):
    Hide the teleport marker at inference to force the model to rely on
    whatever latent representation it built from context.

    modal run modaldotcom/app.py --do-causal-test \\
        --policy lewm_epoch_50 --mask-causal-test

Option B — training-time pixel masking (forces latent causal inference):
    Retrain with 50% patch-masking probability so the model cannot memorise
    the direct pixel cue, then evaluate with the pixel fully masked.

    modal run modaldotcom/app.py --do-train --mask-teleport-prob 0.5
    modal run modaldotcom/app.py --do-causal-test \\
        --policy <job_id>/lewm_epoch_50 --mask-causal-test

Option C — reversed-confound dataset (decouples hue from teleport at data level):
    Collect a held-out dataset where blue rooms have teleport DISABLED and
    green rooms have teleport ENABLED — the opposite of the training confound.
    Evaluate the original model on this reversed dataset, with and without masking.

    modal run modaldotcom/app.py --do-collect-optionc
    modal run modaldotcom/app.py --do-causal-test \\
        --policy lewm_epoch_50 --dataset-name glitched_hue_optionc
    modal run modaldotcom/app.py --do-causal-test \\
        --policy lewm_epoch_50 --dataset-name glitched_hue_optionc --mask-causal-test

SIGReg ablation — prove SIGReg is responsible for hue/position disentanglement:
    Same as Option B but with the isotropic Gaussian regulariser disabled (λ=0).
    If structural invariance error rises, SIGReg is causal for the ICM property.

    modal run modaldotcom/app.py --do-train \\
        --mask-teleport-prob 0.5 --sigreg-weight 0.0 --max-epochs 50 --no-wandb
    modal run modaldotcom/app.py --do-causal-test \\
        --policy <ablation_job_id>/lewm_epoch_50 --mask-causal-test

Theme C — validate the hue intervention itself (on-manifold + single-factor checks):
    No retraining. Adds (A) an on-manifold check comparing z_cf against a
    random-direction negative control, and (B) InvErr probes for extra
    decodable factors (teleported, step_idx, distance_to_target).

    modal run modaldotcom/app.py --do-causal-test \\
        --policy lewm_epoch_50 --extended-validation

Theme D — paired factual/counterfactual ground-truth trajectories (CMTV Critical):
    No retraining. Collects real paired rollouts (same seed/start/actions,
    hue + teleport-gating flipped per the training confound) directly from
    GlitchedHueTwoRoom-v1, then compares the model's prediction against a
    REAL encoded counterfactual frame instead of the translation-based
    z_cf = z_fact + delta_hue used everywhere else. Run BOTH validate calls:
    --mask-theme-d is the paper's primary Ladder 3 test (Option A protocol);
    the unmasked run is only a sanity-check baseline.

    modal run modaldotcom/app.py --do-collect-theme-d
    modal run modaldotcom/app.py --do-theme-d-validate --policy lewm_epoch_50 --mask-theme-d
    modal run modaldotcom/app.py --do-theme-d-validate --policy lewm_epoch_50

Other commands
--------------
    modal run modaldotcom/app.py --do-train --max-epochs 1 --no-wandb   # smoke test
    modal run modaldotcom/app.py --do-eval --policy <job_id>/lewm_epoch_100
    modal run modaldotcom/app.py --do-stats
    modal run modaldotcom/app.py --do-inspect --dataset-name glitched_hue_optionc
    modal run modaldotcom/app.py --do-audit  --dataset-name glitched_hue_tworoom_half
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
    # HDF5 compression plugins (LZ4, blosc, zstd…) — separate layer so it doesn't
    # invalidate the stable-worldmodel cache when hdf5plugin version changes.
    .pip_install("hdf5plugin")
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
    sigreg_weight: float = 0.09,
    seed: int = 3072,
) -> str:
    """Run train.py on a cloud A10G and persist checkpoints to the volume.

    Returns the Hydra job ID so you can pass it to evaluate() as the policy prefix.
    """
    import glob
    import os
    import time
    import uuid

    # Append a short UUID to prevent subdir collisions when multiple containers
    # start within the same second (common when jobs are spawned in parallel).
    run_id = f"ts_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    cmd = [
        "python", "train.py",
        f"data={data}",
        f"trainer.max_epochs={max_epochs}",
        f"wandb.enabled={'True' if wandb_enabled else 'False'}",
        f"subdir={run_id}",
    ]
    if mask_teleport_prob > 0.0:
        cmd += [
            "augmentation.teleport_patch_mask.enabled=True",
            f"augmentation.teleport_patch_mask.mask_probability={mask_teleport_prob}",
        ]
    if sigreg_weight != 0.09:
        cmd.append(f"loss.sigreg.weight={sigreg_weight}")
    if seed != 3072:
        cmd.append(f"seed={seed}")
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
# Dataset audit — room colour + teleport breakdown (no GPU)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={CACHE_DIR: volume},
    env=ENV,
    timeout=300,
)
def audit_dataset(name: str = "glitched_hue_tworoom_half") -> None:
    """Print room-colour and teleport-event breakdown for an HDF5 dataset.

    Background colour is inferred from a background pixel (row=40, col=30)
    when variation.background.color is not stored in the file.
    """
    import h5py
    import hdf5plugin  # noqa: F401 — registers LZ4/blosc filters
    import numpy as np

    volume.reload()
    path = f"{CACHE_DIR}/{name}.h5"
    print(f"\nAudit: {path}\n{'='*60}")

    with h5py.File(path, "r") as f:
        tp     = f["teleported"][:]
        ep_len = f["ep_len"][:]
        ep_off = f["ep_offset"][:]

        has_bg_color = "variation.background.color" in f
        if has_bg_color:
            bg = f["variation.background.color"][ep_off.tolist()]  # (E, 3)
        else:
            bg = f["pixels"][ep_off.tolist(), 40, 30, :]           # (E, 3) inferred

    n_ep = len(ep_len)
    is_blue  = (bg[:, 2].astype(int) - bg[:, 1].astype(int)) > 50
    is_green = (bg[:, 1].astype(int) - bg[:, 2].astype(int)) > 50

    tp_per_ep = np.zeros(n_ep, dtype=int)
    for i in range(n_ep):
        s, e = int(ep_off[i]), int(ep_off[i]) + int(ep_len[i])
        tp_per_ep[i] = int(tp[s:e].sum())
    has_tp = tp_per_ep > 0

    src = "stored" if has_bg_color else "inferred from pixel"
    print(f"Room colour source : {src}")
    print(f"Total episodes     : {n_ep}")
    print(f"Total steps        : {len(tp)}")
    print(f"Teleport steps     : {int(tp.sum())}")
    print()
    print(f"Blue  episodes : {int(is_blue.sum()):>6}  ({100*is_blue.mean():.1f}%)")
    print(f"Green episodes : {int(is_green.sum()):>6}  ({100*is_green.mean():.1f}%)")
    print(f"Other/mixed    : {int(n_ep - is_blue.sum() - is_green.sum()):>6}")
    print()
    print(f"Episodes WITH teleport    : {int(has_tp.sum()):>6}")
    print(f"  of which blue           : {int((has_tp & is_blue).sum()):>6}")
    print(f"  of which green          : {int((has_tp & is_green).sum()):>6}")
    print()
    print(f"Episodes WITHOUT teleport : {int((~has_tp).sum()):>6}")
    print(f"  of which blue           : {int((~has_tp & is_blue).sum()):>6}")
    print(f"  of which green          : {int((~has_tp & is_green).sum()):>6}")
    print(f"{'='*60}\n")


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

    volume.reload()

    path = f"{CACHE_DIR}/{name}.h5"
    print(f"\nInspecting: {path}\n{'='*60}")

    import os
    print(f"Exists : {os.path.exists(path)}")
    if os.path.exists(path):
        print(f"Size   : {os.path.getsize(path):,} bytes")
        with open(path, "rb") as fraw:
            magic = fraw.read(8)
        print(f"Magic  : {magic.hex()}  (HDF5 should start with 894844460d0a1a0a)")
        print(f"is_hdf5: {h5py.is_hdf5(path)}")

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
    World.record_dataset(), then merges them into one HDF5 file using
    chunked I/O to avoid loading all pixels into RAM at once.

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


@app.function(
    image=image,
    volumes={CACHE_DIR: volume},
    env=ENV,
    timeout=3600,   # merge of two 3 GB files should finish in < 1 h
    memory=16384,   # 16 GB — chunked I/O keeps peak RAM low but give headroom
)
def remerge_optionc() -> str:
    """Re-run the merge step from existing half-files (skips collection).

    Runs the merge inline (not as a subprocess) so that h5py writes go
    directly to the Modal volume mount — no cross-process cache coherency
    issues before volume.commit().

    Use this when the half-files (_blue.h5 / _green.h5) are already on the
    volume but the merged glitched_hue_optionc.h5 is corrupt or missing.
    """
    import os
    import sys
    from pathlib import Path

    volume.reload()

    sys.path.insert(0, "/workspace")
    from research.collect_option_c import _merge_halves  # noqa: E402

    cache = Path(CACHE_DIR)
    out = _merge_halves(cache)

    fsize = os.path.getsize(str(out))
    print(f"File size before commit: {fsize:,} bytes")
    volume.commit()
    print(f"\n✅ Merged dataset on volume: {out}  ({fsize:,} bytes)")
    return str(out)


# ---------------------------------------------------------------------------
# Theme D — paired factual/counterfactual ground-truth trajectories (no GPU
# for collection; GPU for the encode/predict comparison)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={CACHE_DIR: volume},
    env=ENV,
    timeout=14400,  # 4 h ceiling -- teleport hit rate is ~20-30%, so 200 usable
                    # episodes needs on the order of ~1000 attempted rollouts
)
def collect_theme_d(n_episodes: int = 200, episode_len: int = 100, seed: int = 42) -> str:
    """Collect the Theme D paired factual/counterfactual dataset on a cloud CPU worker.

    Runs research/collect_theme_d_paired.py, which drives two envs (factual:
    blue/teleport-enabled, counterfactual: green/teleport-disabled) through an
    identical seed + action sequence, recording only episodes where a teleport
    actually fires. Args:
        n_episodes:  Target number of USABLE (teleported) paired episodes.
        episode_len: Raw env steps per episode (default: 100, matches
                     world.max_episode_steps in glitched_hue.yaml).
        seed:        Base seed; attempt i uses seed+i.

    Returns:
        Absolute path of the factual HDF5 file on the volume (the paired
        counterfactual file sits alongside it, same prefix + "_cf.h5").
    """
    import subprocess
    cmd = [
        "python", "research/collect_theme_d_paired.py",
        f"--n-episodes={n_episodes}",
        f"--episode-len={episode_len}",
        f"--seed={seed}",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace")
    volume.commit()
    out = f"{CACHE_DIR}/glitched_hue_theme_d_fact.h5"
    print(f"\n✅ Paired dataset written to volume: {out}  (+ _cf.h5 alongside it)")
    return out


@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: volume},
    secrets=[wandb_secret],
    env=ENV,
    timeout=3600,
)
def theme_d_validate(
    policy: str,
    paired_dataset_name: str = "glitched_hue_theme_d",
    n_probe_batches: int = 200,
    mask_teleport: bool = False,
    no_wandb: bool = False,
) -> str:
    """Run research/theme_d_paired_validation.py on a cloud A10G.

    Compares the model's prediction against a REAL encoded counterfactual
    frame from the paired dataset (collect_theme_d), instead of the
    translation-based z_cf = z_fact + delta_hue used by causal_test(). Writes
    theme_d_paired_results.json plus two diagnostic plots to the volume
    alongside the checkpoint.

    Args:
        policy:               Checkpoint path relative to STABLEWM_HOME,
                               without the '_object.ckpt' suffix.
        paired_dataset_name:  Prefix used by collect_theme_d (default:
                               glitched_hue_theme_d).
        mask_teleport:        Option A protocol -- zero out the teleport-pixel
                               patch before every encode() call. This is the
                               paper's primary Ladder 3 test (Option A/B/C+A);
                               the unmasked run is only a sanity-check
                               baseline. Run both.
        no_wandb:              If True, skip W&B logging (dry run).

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

    cmd = [
        "python", "research/theme_d_paired_validation.py", ckpt_path,
        "--paired-dataset-name", paired_dataset_name,
        "--n-probe-batches", str(n_probe_batches),
    ]
    if mask_teleport:
        cmd.append("--mask-teleport")
    if no_wandb:
        cmd.append("--no-wandb")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace")

    volume.commit()

    suffix = "_masked" if mask_teleport else ""
    results_file = f"{CACHE_DIR}/{os.path.dirname(policy)}/theme_d_paired_results{suffix}.json"
    print(f"\n✅ Theme D validation complete. Results: {results_file}")
    return results_file


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
    n_aap_episodes: int = 200,
    extended_validation: bool = False,
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
    if n_aap_episodes != 200:
        cmd += ["--n-aap-episodes", str(n_aap_episodes)]
    if extended_validation:
        cmd.append("--extended-validation")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace")

    volume.commit()

    ds_suffix = f"_{dataset_name}" if dataset_name != "glitched_hue_tworoom_half" else ""
    suffix = ("_masked" if mask_teleport else "") + ds_suffix
    results_file = f"{CACHE_DIR}/{os.path.dirname(policy)}/causal_test{suffix}_results.json"
    print(f"\n✅ Causal test complete. Results: {results_file}")
    return results_file


# ---------------------------------------------------------------------------
# Statistical study — 3 seeds × 3 conditions, fully parallel
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={CACHE_DIR: volume},
    secrets=[wandb_secret],
    env=ENV,
    timeout=86400,   # 24 h ceiling — covers all parallel training + tests
)
def run_statistical_study(
    max_epochs: int = 50,
    n_aap_episodes: int = 200,
    seeds: list = None,
    batch_size: int = 4,
) -> dict:
    """Full parallel statistical study: 3 seeds × 3 conditions.

    Spawns training and causal-test jobs in batches of `batch_size` to stay
    within the 10-GPU workspace concurrency limit. Existing checkpoints from
    prior runs are reused so no retraining is needed for them.
    Returns aggregated mean ± std per condition.
    """
    import json
    import numpy as np
    from pathlib import Path

    if seeds is None:
        seeds = [3072, 1234, 5678]

    # Existing checkpoints — no retraining needed.
    # option_b seeds 1234/5678 are intentionally absent: their prior 50-epoch
    # runs are insufficient; option_b always trains for 100 epochs.
    EXISTING = {
        ("baseline", 3072): "lewm_epoch_50",
        ("option_b",  3072): "ts_1776884938/lewm_epoch_50",   # 100 epochs
        ("ablation",  3072): "ts_1777735299/lewm_epoch_50",
        ("baseline",  1234): "ts_1777994910/lewm_epoch_50",
        ("option_b",  1234): "ts_1778145022_93c1c4/lewm_epoch_91",  # 91 epochs (~converged)
        ("ablation",  1234): "ts_1778145024_04e194/lewm_epoch_50",
        ("baseline",  5678): "ts_1777991006/lewm_epoch_50",
        ("option_b",  5678): "ts_1778145025_368123/lewm_epoch_90",  # 90 epochs (~converged)
        ("ablation",  5678): "ts_1777735306/lewm_epoch_50",
    }

    # Per-condition epoch counts: option_b needs 100 epochs to converge.
    COND_EPOCHS = {
        "baseline": max_epochs,
        "option_b": 100,
        "ablation": max_epochs,
    }

    # Determine which (condition, seed) pairs still need fresh training.
    to_train = []
    for seed in seeds:
        for cond, mask_prob, sigreg in [
            ("baseline", 0.0, 0.09),
            ("option_b", 0.5, 0.09),
            ("ablation", 0.5, 0.0),
        ]:
            if (cond, seed) not in EXISTING:
                to_train.append((cond, seed, mask_prob, sigreg, COND_EPOCHS[cond]))

    policy_paths = dict(EXISTING)

    def _spawn_and_collect_training(batch):
        """Spawn one batch of training jobs and block until all complete."""
        handles = {}
        for (cond, seed, mask_prob, sigreg, epochs) in batch:
            print(f"  spawning train: {cond} seed={seed} mask={mask_prob} sigreg={sigreg} epochs={epochs}")
            h = train.spawn(
                max_epochs=epochs,
                wandb_enabled=False,
                mask_teleport_prob=mask_prob,
                sigreg_weight=sigreg,
                seed=seed,
            )
            handles[(cond, seed)] = h
        for (cond, seed, *_), handle in zip(batch, handles.values()):
            try:
                path = handle.get()
                policy_paths[(cond, seed)] = path
                print(f"  ✓ {cond} seed={seed}  →  {path}")
            except Exception as e:
                print(f"  ✗ {cond} seed={seed} FAILED: {e}")

    # Phase 1 — train in batches to respect the 10-GPU limit.
    print(f"Phase 1: {len(to_train)} training run(s) needed, batch_size={batch_size}")
    for i in range(0, len(to_train), batch_size):
        batch = to_train[i:i + batch_size]
        print(f"  batch {i // batch_size + 1}/{-(-len(to_train) // batch_size)}: {[(c, s, e) for c, s, _, __, e in batch]}")
        _spawn_and_collect_training(batch)

    # Phase 2 — spawn causal tests in batches.
    TEST_VARIANTS = {
        "baseline": dict(mask_teleport=False, dataset_name="glitched_hue_tworoom_half"),
        "option_b": dict(mask_teleport=True,  dataset_name="glitched_hue_tworoom_half"),
        "ablation": dict(mask_teleport=True,  dataset_name="glitched_hue_tworoom_half"),
    }

    # Build the full list of (key, policy, kwargs) causal tests to run.
    causal_todo = []
    for (cond, seed), policy in policy_paths.items():
        variant = TEST_VARIANTS.get(cond)
        if variant is None:
            continue
        causal_todo.append(((cond, seed), policy, variant))
    # Option C and C+A use the baseline seed-3072 checkpoint.
    baseline_3072 = EXISTING[("baseline", 3072)]
    for ds_cond, mask in [("option_c", False), ("option_ca", True)]:
        causal_todo.append(
            ((ds_cond, 3072), baseline_3072,
             dict(mask_teleport=mask, dataset_name="glitched_hue_optionc"))
        )

    print(f"\nPhase 2: {len(causal_todo)} causal test(s), batch_size={batch_size}")
    result_files = {}
    for i in range(0, len(causal_todo), batch_size):
        batch = causal_todo[i:i + batch_size]
        print(f"  batch {i // batch_size + 1}/{-(-len(causal_todo) // batch_size)}: {[k for k,*_ in batch]}")
        handles = {}
        for (key, policy, kwargs) in batch:
            h = causal_test.spawn(
                policy=policy, no_wandb=True, n_aap_episodes=n_aap_episodes, **kwargs
            )
            handles[key] = h
        for key, handle in handles.items():
            try:
                fpath = handle.get()
                result_files[key] = fpath
                print(f"  ✓ causal test {key}  →  {fpath}")
            except Exception as e:
                print(f"  ✗ causal test {key} FAILED: {e}")

    # Phase 3 — aggregate across seeds.
    volume.reload()
    aggregated = {}
    for cond in ["baseline", "option_b", "ablation", "option_c", "option_ca"]:
        cond_results = []
        for key, fpath in result_files.items():
            if key[0] != cond:
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
                cond_results.append(data["metrics"])
            except Exception as e:
                print(f"  warning: could not read {fpath}: {e}")

        if not cond_results:
            print(f"  warning: no results for condition '{cond}' — skipping")
            continue

        agg = {}
        for metric in cond_results[0]:
            if metric == "per_episode_ratios":
                continue
            vals = [r[metric] for r in cond_results if isinstance(r.get(metric), (int, float))]
            if vals:
                agg[metric + "_mean"] = float(np.mean(vals))
                agg[metric + "_std"]  = float(np.std(vals))
        agg["n_seeds"] = len(cond_results)
        aggregated[cond] = agg
        print(f"  {cond}: n_seeds={len(cond_results)}, ratio={agg.get('surprise_ratio_mean', '?'):.4f}")

    out_path = Path(CACHE_DIR) / "statistical_study_results.json"
    with open(out_path, "w") as f:
        json.dump({"seeds": seeds, "n_aap_episodes": n_aap_episodes,
                   "conditions": aggregated}, f, indent=2)
    volume.commit()
    print(f"\n✅ Aggregated results → {out_path}")
    return aggregated


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
    do_remerge_optionc: bool = False,
    do_audit: bool = False,
    do_statistical_study: bool = False,
    do_collect_theme_d: bool = False,
    do_theme_d_validate: bool = False,
    mask_theme_d: bool = False,
    data: str = "glitched_hue_tworoom",
    max_epochs: int = 100,
    policy: str = "",
    config_name: str = "glitched_hue_tworoom",
    run_id: str = "",
    no_wandb: bool = False,
    dataset_name: str = "glitched_hue_tworoom_half",
    mask_causal_test: bool = False,
    extended_validation: bool = False,
    mask_teleport_prob: float = 0.0,
    sigreg_weight: float = 0.09,
    optionc_episodes: int = 5000,
    theme_d_episodes: int = 200,
    theme_d_episode_len: int = 100,
    theme_d_dataset_name: str = "glitched_hue_theme_d",
    study_seeds: str = "3072,1234,5678",
    study_epochs: int = 50,
    study_n_aap: int = 200,
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

    # SIGReg ablation — same as Option B but with SIGReg disabled (λ=0)
    modal run modaldotcom/app.py --do-train --mask-teleport-prob 0.5 --sigreg-weight 0.0 --max-epochs 50 --no-wandb
    modal run modaldotcom/app.py --do-causal-test --policy <ablation_job_id>/lewm_epoch_50 --mask-causal-test

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

    # Statistical study — 3 seeds × 3 conditions, all training runs in parallel (~7.5 h)
    modal run modaldotcom/app.py --do-statistical-study
    modal run modaldotcom/app.py --do-statistical-study --study-seeds 3072,1234,5678 --study-epochs 50 --study-n-aap 200

    # Re-run only causal tests without retraining (after changing n_aap_episodes)
    modal run modaldotcom/app.py --do-causal-test --policy lewm_epoch_50 --n-aap-episodes 200

    # Theme C — on-manifold check + extra-factor probes (no retraining)
    modal run modaldotcom/app.py --do-causal-test --policy lewm_epoch_50 --extended-validation

    # Theme D — paired factual/counterfactual ground-truth trajectories (no retraining)
    modal run modaldotcom/app.py --do-collect-theme-d
    modal run modaldotcom/app.py --do-theme-d-validate --policy lewm_epoch_50 --mask-theme-d
    modal run modaldotcom/app.py --do-theme-d-validate --policy lewm_epoch_50
    """
    if not any([do_train, do_eval, do_stats, do_causal_test, do_inspect,
                do_collect_optionc, do_remerge_optionc, do_audit, do_statistical_study,
                do_collect_theme_d, do_theme_d_validate]):
        print(
            "Nothing to do. Pass --do-train, --do-eval, --do-stats, --do-causal-test, "
            "--do-inspect, --do-collect-optionc, --do-statistical-study, "
            "--do-collect-theme-d, or --do-theme-d-validate."
        )
        return

    policy_path = policy
    if do_train:
        policy_path = train.remote(
            data=data,
            max_epochs=max_epochs,
            wandb_enabled=not no_wandb,
            mask_teleport_prob=mask_teleport_prob,
            sigreg_weight=sigreg_weight,
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

    if do_remerge_optionc:
        out = remerge_optionc.remote()
        print(f"Option C merged dataset on volume: {out}")

    if do_collect_theme_d:
        out = collect_theme_d.remote(
            n_episodes=theme_d_episodes,
            episode_len=theme_d_episode_len,
        )
        print(f"Theme D paired dataset on volume: {out}")

    if do_theme_d_validate:
        if not policy:
            print("--do-theme-d-validate requires --policy <path>. Example:")
            print("  modal run modaldotcom/app.py --do-theme-d-validate --policy lewm_epoch_50")
            return
        results_file = theme_d_validate.remote(
            policy=policy,
            paired_dataset_name=theme_d_dataset_name,
            mask_teleport=mask_theme_d,
            no_wandb=no_wandb,
        )
        print(f"Theme D results file on volume: {results_file}")

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
            extended_validation=extended_validation,
        )
        print(f"Results file on volume: {results_file}")

    if do_inspect:
        inspect_dataset.remote(name=dataset_name)

    if do_audit:
        audit_dataset.remote(name=dataset_name)

    if do_statistical_study:
        seeds = [int(s) for s in study_seeds.split(",")]
        print(f"Launching statistical study: seeds={seeds}, epochs={study_epochs}, n_aap={study_n_aap}")
        # .remote() blocks the local entrypoint so Modal --detach keeps the
        # orchestrator alive even if the local terminal disconnects.
        result = run_statistical_study.remote(
            max_epochs=study_epochs,
            n_aap_episodes=study_n_aap,
            seeds=seeds,
        )
        print("\n=== Statistical Study Results ===")
        import json
        print(json.dumps(result, indent=2))

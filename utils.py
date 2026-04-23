import os
import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from torchvision.transforms import v2 as tv_transforms
from lightning.pytorch.callbacks import Callback


def detect_teleport_bbox(dataset_path: str, n_samples: int = 100) -> tuple:
    """Return (row_min, row_max, col_min, col_max) of the teleport marker in pixel space.

    Uses differential comparison between teleport frames (blue room, marker visible)
    and green-room frames (no marker, but door still white) to isolate the marker
    from other persistent white features like the door.

    Works whether the teleport position was fixed or varied — no coordinates hardcoded.
    """
    import h5py

    rng = np.random.default_rng(0)

    with h5py.File(dataset_path, "r") as f:
        tp_mask = f["teleported"][:]                       # (N,) bool
        n_total = len(tp_mask)

        tp_indices = np.where(tp_mask)[0][:n_samples]
        if len(tp_indices) == 0:
            raise RuntimeError("No teleported=True steps found in dataset.")

        # Sample random frames; identify green-room ones from a background pixel at
        # (row=40, col=30) — away from wall, door, agent, and teleport marker.
        candidates = np.sort(rng.choice(n_total, min(n_total, 8000), replace=False))
        bg_px = f["pixels"][candidates.tolist(), 40, 30, :]  # (n, 3) uint8
        is_green = (bg_px[:, 1].astype(int) - bg_px[:, 2].astype(int)) > 50
        green_indices = np.sort(candidates[is_green][:n_samples])
        if len(green_indices) < 10:
            raise RuntimeError("Could not find enough green-room frames for differential detection.")

        tp_frames    = f["pixels"][np.sort(tp_indices).tolist()]   # (n, H, W, 3)
        green_frames = f["pixels"][green_indices.tolist()]          # (m, H, W, 3)

    def _bright(frames):
        return (
            (frames[:, :, :, 0] > 200)
            & (frames[:, :, :, 1] > 200)
            & (frames[:, :, :, 2] > 200)
        ).mean(axis=0)

    delta = _bright(tp_frames) - _bright(green_frames)
    marker_mask = delta > 0.5

    rows = np.where(marker_mask.any(axis=1))[0]
    cols = np.where(marker_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise RuntimeError(
            "Teleport marker could not be isolated via differential detection. "
            "Ensure teleported=True frames and green-room frames are present."
        )

    PATCH = 14
    r0 = (int(rows.min()) // PATCH) * PATCH
    r1 = (int(rows.max()) // PATCH + 1) * PATCH
    c0 = (int(cols.min()) // PATCH) * PATCH
    c1 = (int(cols.max()) // PATCH + 1) * PATCH
    return r0, r1, c0, c1


class TeleportPatchMask:
    """Zero out the teleport pixel patch in ImageNet-normalized CHW tensors.

    Use mask_prob=0.5 during training and 1.0 at evaluation time (Option B).
    """

    def __init__(self, tp_bbox: tuple, mask_prob: float = 0.5):
        self.tp_bbox = tp_bbox
        self.mask_prob = mask_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_prob < 1.0 and torch.rand(1).item() >= self.mask_prob:
            return x
        r0, r1, c0, c1 = self.tp_bbox
        x = x.clone()
        x[..., r0:r1, c0:c1] = 0.0
        return x


def get_stablewm_home() -> Path:
    """Return the stable-worldmodel cache directory.

    Respects $STABLEWM_HOME if set, otherwise falls back to ~/.stable-wm/.
    Creates the directory if it does not exist.
    """
    try:
        from stable_worldmodel.data.utils import get_cache_dir
        return Path(get_cache_dir())
    except Exception:
        path = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
        path.mkdir(parents=True, exist_ok=True)
        return path

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    # Avoid stable_pretraining.Resize which has a broken __call__ (references
    # self.transform but the attribute is stored as self._transform). Use
    # WrapTorchTransform + torchvision directly — same pattern as get_column_normalizer.
    resize = dt.transforms.WrapTorchTransform(
        tv_transforms.Resize(img_size, antialias=True),
        source=source,
        target=target,
    )
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModalVolumeCommitCallback(Callback):
    """Commit the Modal volume every N epochs so checkpoints survive container crashes.

    Reads the volume name from the MODAL_VOLUME_NAME env var; silently skips when
    not running inside Modal (env var absent or modal package unavailable).
    """

    def __init__(self, commit_every_n_epochs: int = 10):
        self.volume_name = os.environ.get("MODAL_VOLUME_NAME", "")
        self.commit_every_n_epochs = commit_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.volume_name:
            return
        if (trainer.current_epoch + 1) % self.commit_every_n_epochs != 0:
            return
        try:
            import modal
            modal.Volume.from_name(self.volume_name).commit()
            print(f"[ModalVolumeCommitCallback] volume '{self.volume_name}' committed at epoch {trainer.current_epoch + 1}")
        except Exception as e:
            print(f"[ModalVolumeCommitCallback] warning: could not commit volume: {e}")


class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")
import os
import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from torchvision.transforms import v2 as tv_transforms
from lightning.pytorch.callbacks import Callback


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
"""Download a dataset from Hugging Face into $STABLEWM_HOME.

Usage
-----
# Download the default Glitched Hue Two Room dataset:
    python scripts/download_dataset.py

# Override repo or filename:
    python scripts/download_dataset.py \
        --repo robomotic/causality-two-room-modal \
        --file glitched_hue_tworoom_half.h5

# Download to an explicit directory (overrides $STABLEWM_HOME):
    python scripts/download_dataset.py --dest /data/my-datasets
"""

import argparse
import os
import sys
from pathlib import Path

HF_REPO = "robomotic/causality-two-room-modal"
HF_FILE = "glitched_hue_tworoom_half.h5"


def get_stablewm_home() -> Path:
    try:
        from stable_worldmodel.data.utils import get_cache_dir
        return Path(get_cache_dir())
    except Exception:
        path = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
        path.mkdir(parents=True, exist_ok=True)
        return path


def download(repo: str, filename: str, dest: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} from {repo} → {dest}/")
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest),
    )
    print(f"Saved to: {path}")
    return Path(path)


def main():
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face")
    parser.add_argument("--repo", default=HF_REPO, help="HuggingFace dataset repo ID")
    parser.add_argument("--file", default=HF_FILE, help="Filename inside the repo")
    parser.add_argument("--dest", default=None, type=Path,
                        help="Destination directory (default: $STABLEWM_HOME)")
    args = parser.parse_args()

    dest = args.dest if args.dest is not None else get_stablewm_home()
    download(args.repo, args.file, dest)


if __name__ == "__main__":
    main()

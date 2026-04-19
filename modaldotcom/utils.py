"""Utility commands for managing the swm-cache Modal volume.

Usage
-----
# Dry-run: list .ckpt files that would be deleted
python modaldotcom/utils.py

# Delete all .ckpt files from swm-cache
python modaldotcom/utils.py --clean-ckpts

# Confirm without prompt
python modaldotcom/utils.py --clean-ckpts --yes
"""

import argparse
import sys

import modal

VOLUME_NAME = "swm-cache"


def list_ckpts(vol: modal.Volume) -> list[str]:
    """Return paths of all .ckpt and config.yaml files in the volume (recursive)."""
    results: list[str] = []
    _collect(vol, "/", results)
    return results


def _collect(vol: modal.Volume, path: str, out: list[str]) -> None:
    for entry in vol.listdir(path):
        # entry.type: FileEntryType.DIRECTORY == 2, FILE == 1
        if entry.type.value == 2:
            _collect(vol, entry.path, out)
        elif entry.path.endswith(".ckpt") or entry.path.endswith("config.yaml"):
            out.append(entry.path)


def clean_ckpts(dry_run: bool = False, yes: bool = False) -> None:
    vol = modal.Volume.from_name(VOLUME_NAME)
    ckpts = list_ckpts(vol)

    if not ckpts:
        print("No .ckpt files found in swm-cache.")
        return

    print(f"Found {len(ckpts)} .ckpt file(s) in {VOLUME_NAME}:")
    for name in ckpts:
        print(f"  {name}")

    if dry_run:
        print("\nDry-run — nothing deleted.")
        return

    if not yes:
        answer = input(f"\nDelete all {len(ckpts)} file(s)? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    for name in ckpts:
        vol.remove_file(name)
        print(f"  ✓ deleted {name}")

    print(f"\nDone. Deleted {len(ckpts)} file(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="swm-cache volume utilities")
    parser.add_argument(
        "--clean-ckpts",
        action="store_true",
        help="Delete all .ckpt files from swm-cache",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without deleting",
    )
    args = parser.parse_args()

    if args.clean_ckpts:
        clean_ckpts(dry_run=args.dry_run, yes=args.yes)
    else:
        # Default: show what's there
        vol = modal.Volume.from_name(VOLUME_NAME)
        ckpts = list_ckpts(vol)
        if ckpts:
            print(f"{len(ckpts)} .ckpt file(s) in {VOLUME_NAME}:")
            for name in ckpts:
                print(f"  {name}")
        else:
            print("No .ckpt files found.")


if __name__ == "__main__":
    main()

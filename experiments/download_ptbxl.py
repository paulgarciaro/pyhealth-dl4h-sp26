"""
download_ptbxl.py — Download PTB-XL from Kaggle (faster than PhysioNet).

Dataset: https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset
Contains: ptbxl_database.csv, scp_statements.csv, records500/

Prerequisites
-------------
1. Install Kaggle CLI:
       pip install kaggle

2. Place your API key at ~/.kaggle/kaggle.json
   (Download from https://www.kaggle.com/settings → "Create New Token")
   Then: chmod 600 ~/.kaggle/kaggle.json

Usage
-----
    python experiments/download_ptbxl.py --dest /path/to/ptb-xl
    python experiments/download_ptbxl.py --dest /path/to/ptb-xl --verify
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


# Kaggle dataset slugs to try in order (first available wins)
_KAGGLE_SLUGS = [
    "khyeh0719/ptb-xl-dataset",
    "bjoernjostein/ptb-xl-electrocardiography-dataset",
]

_REQUIRED_FILES = [
    "ptbxl_database.csv",
    "scp_statements.csv",
]

_REQUIRED_DIRS = [
    "records500",
]


def check_kaggle_cli():
    try:
        result = subprocess.run(
            ["kaggle", "--version"], capture_output=True, text=True
        )
        print(f"Kaggle CLI: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("ERROR: kaggle CLI not found. Install with: pip install kaggle")
        return False


def check_kaggle_credentials():
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print(
            "ERROR: Kaggle credentials not found.\n"
            f"  Expected: {cred}\n"
            "  Get your API key at: https://www.kaggle.com/settings → Create New Token\n"
            f"  Then:  chmod 600 {cred}"
        )
        return False
    return True


def already_downloaded(dest: Path) -> bool:
    for f in _REQUIRED_FILES:
        if not (dest / f).is_file():
            return False
    for d in _REQUIRED_DIRS:
        if not (dest / d).is_dir():
            return False
    return True


def download(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)

    if already_downloaded(dest):
        print(f"PTB-XL already present at {dest} — skipping download.")
        return

    last_err = None
    for slug in _KAGGLE_SLUGS:
        print(f"\nTrying Kaggle dataset: {slug} ...")
        try:
            subprocess.run(
                [
                    "kaggle", "datasets", "download",
                    "-d", slug,
                    "-p", str(dest),
                    "--unzip",
                ],
                check=True,
            )
            print(f"Download complete → {dest}")
            return
        except subprocess.CalledProcessError as e:
            print(f"  Failed ({slug}): {e}")
            last_err = e

    raise RuntimeError(
        f"All Kaggle slugs failed. Last error: {last_err}\n"
        "Fallback: download manually from "
        "https://physionet.org/content/ptb-xl/1.0.3/"
    )


def verify(dest: Path):
    print(f"\nVerifying {dest} ...")
    ok = True
    for f in _REQUIRED_FILES:
        p = dest / f
        if p.is_file():
            print(f"  ✓ {f} ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  ✗ {f} MISSING")
            ok = False
    for d in _REQUIRED_DIRS:
        p = dest / d
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*.hea"))
            print(f"  ✓ {d}/  ({n} .hea files)")
        else:
            print(f"  ✗ {d}/ MISSING")
            ok = False
    if ok:
        print("\nAll files present ✓")
    else:
        print("\nSome files missing — re-run without --verify to re-download.")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Download PTB-XL from Kaggle")
    parser.add_argument(
        "--dest",
        default=str(Path.home() / "data" / "ptb-xl-1.0.3"),
        help="Destination directory (default: ~/data/ptb-xl-1.0.3)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing download, do not re-download",
    )
    args = parser.parse_args()
    dest = Path(args.dest)

    if args.verify:
        sys.exit(0 if verify(dest) else 1)

    if not check_kaggle_cli():
        sys.exit(1)
    if not check_kaggle_credentials():
        sys.exit(1)

    download(dest)
    verify(dest)
    print(f"\nSet PTB_XL_ROOT={dest} when running experiments.")


if __name__ == "__main__":
    main()

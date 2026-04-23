"""
Convert *_synced.MP4 and *_synced_depth.MP4 videos under a directory
to 720p, 30fps H.264.

Usage:
  python3 vilma_compress_videos.py --input-dir /path/to/videos

Equivalent ffmpeg settings:
  ffmpeg -i x_synced.MP4 \
    -vf "scale=-2:720,fps=30" \
    -c:v libx264 -crf 20 -preset fast -an x_compressed.MP4
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert *_synced.MP4 and *_synced_depth.MP4 files to "
            "720p/30fps compressed mp4."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Root directory to search recursively for "
            "*_synced.MP4 and *_synced_depth.MP4 files."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_compressed.MP4 and *_depth_compressed.MP4 outputs.",
    )
    return parser.parse_args()


def output_path_for(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_synced_depth"):
        out_stem = stem[: -len("_synced_depth")] + "_depth_compressed"
    elif stem.endswith("_synced"):
        out_stem = stem[: -len("_synced")] + "_compressed"
    else:
        out_stem = stem + "_compressed"
    return input_path.with_name(out_stem + input_path.suffix)


def convert_one(input_path: Path, output_path: Path, overwrite: bool) -> None:
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vf",
        "scale=-2:720,fps=30", #"scale=-2:720:reset_sar=1,fps=30",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "fast",
        "-an",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    root = Path(args.input_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Input directory not found: {root}")
        return 1

    files = sorted(
        set(root.rglob("*_synced.MP4")) | set(root.rglob("*_synced_depth.MP4"))
    )
    if not files:
        print(f"No *_synced.MP4 or *_synced_depth.MP4 files found under: {root}")
        return 0

    converted = 0
    skipped = 0
    failed = 0

    for src in files:
        dst = output_path_for(src)
        if dst.exists() and not args.overwrite:
            print(f"Skipping (exists): {dst}")
            skipped += 1
            continue
        try:
            print(f"Converting: {src} -> {dst}")
            convert_one(src, dst, overwrite=args.overwrite)
            converted += 1
        except subprocess.CalledProcessError as e:
            print(f"Failed: {src} ({e})")
            failed += 1

    print(f"Done. converted={converted} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
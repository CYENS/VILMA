"""
python3 dataset_creation/vilma_print_hdf5_contents.py --h5 vilma_dataset.h5 > the_contents.txt

Print HDF5 file contents recursively.

For array datasets, prints:
  - shape
  - first line (row 0 for 2D+, first element for 1D)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print HDF5 contents")
    parser.add_argument("--h5", required=True, help="Path to .h5 file")
    return parser.parse_args()


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.astype(str)
    return value


def first_line_of_array(arr: np.ndarray):
    if arr.ndim == 0:
        return decode_if_bytes(arr.item())
    if arr.ndim == 1:
        return decode_if_bytes(arr[0]) if arr.shape[0] > 0 else "<empty>"
    if arr.shape[0] == 0:
        return "<empty>"
    row0 = arr[0]
    if isinstance(row0, np.ndarray):
        return [decode_if_bytes(x) for x in row0.tolist()]
    return decode_if_bytes(row0)


def print_attrs(obj, indent: str) -> None:
    if not obj.attrs:
        return
    for k, v in obj.attrs.items():
        v = decode_if_bytes(v)
        print(f"{indent}  @ {k}: {v}")


def print_dataset(name: str, dset: h5py.Dataset, indent: str) -> None:
    print(f"{indent}- dataset: {name} (dtype={dset.dtype})")
    arr = dset[()]
    arr_np = np.asarray(arr)

    if arr_np.ndim == 0:
        value = decode_if_bytes(arr_np.item())
        print(f"{indent}  value: {value}")
        return

    first = first_line_of_array(arr_np)
    print(f"{indent}  shape: {arr_np.shape}")
    print(f"{indent}  first_line: {first}")


def walk_group(group: h5py.Group, indent: str = "") -> None:
    print(f"{indent}+ group: /{group.name.lstrip('/')}")
    print_attrs(group, indent)

    for key in sorted(group.keys()):
        obj = group[key]
        if isinstance(obj, h5py.Group):
            walk_group(obj, indent + "  ")
        elif isinstance(obj, h5py.Dataset):
            print_dataset(key, obj, indent + "  ")


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5).resolve()
    if not h5_path.exists():
        print(f"H5 file not found: {h5_path}")
        return 1

    with h5py.File(h5_path, "r") as h5f:
        walk_group(h5f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import h5py


STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize video files following HDF5 data/repetition hierarchy."
    )
    parser.add_argument("--h5", required=True, help="Path to HDF5 dataset file.")
    parser.add_argument(
        "--recordings-root",
        required=True,
        help="Filesystem root corresponding to paths starting with recordings/...",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root folder where organized hierarchy will be created.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default is copy).",
    )
    return parser.parse_args()


def read_string_dataset(group: h5py.Group, name: str) -> Optional[str]:
    if name not in group:
        return None
    value = group[name][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def write_string_dataset(group: h5py.Group, name: str, value: str) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=value, dtype=STRING_DTYPE)


def resolve_source_path(
    path_str: str, recordings_root: Path, dataset_root: Path, role: Optional[str] = None
) -> Path:
    normalized = str(path_str).replace("\\", "/")
    if normalized.startswith("recordings/"):
        rel = normalized[len("recordings/") :]
        return (recordings_root / rel).resolve()
    if normalized.startswith("data/"):
        # Preferred when files were already organized.
        candidate = (dataset_root / normalized).resolve()
        if candidate.exists():
            return candidate
        # Fallback: HDF5 may already store logical data/... paths while files are still
        # physically in recordings/{head,right,left}/.
        filename = Path(normalized).name
        if role:
            role_candidate = (recordings_root / role / filename).resolve()
            if role_candidate.exists():
                return role_candidate
        for subdir in ("head", "right", "left"):
            c = (recordings_root / subdir / filename).resolve()
            if c.exists():
                return c
        return candidate
    return Path(normalized).resolve()


def role_from_sensor_name(sensor_name: str) -> Optional[str]:
    if sensor_name == "head_camera":
        return "head"
    if sensor_name == "right_gripper":
        return "right"
    if sensor_name == "left_gripper":
        return "left"
    return None


def data_folder_name(data_group_name: str) -> str:
    # data_001 -> data001
    return data_group_name.replace("_", "")

def iter_data_groups(data_root: h5py.Group):
    """
    Yield (family_name_or_none, data_group_name, data_group) for both layouts:
    - Legacy: /data/data_001
    - New:    /data/D_C01/D_C01.01
    """
    for top_name in sorted(data_root.keys()):
        top_group = data_root[top_name]
        if not isinstance(top_group, h5py.Group):
            continue
        # Legacy flat layout.
        if top_group.get("repetitions") is not None:
            yield None, top_name, top_group
            continue
        # Nested family layout.
        for data_name in sorted(top_group.keys()):
            dg = top_group[data_name]
            if isinstance(dg, h5py.Group) and dg.get("repetitions") is not None:
                yield top_name, data_name, dg


def organize_one_path(
    sensor_group: h5py.Group,
    dataset_name: str,
    role: str,
    recordings_root: Path,
    dataset_root: Path,
    target_dir: Path,
    h5_relative_prefix: str,
    move_files: bool,
) -> tuple[int, int]:
    """
    Returns (updated_count, missing_count)
    """
    current = read_string_dataset(sensor_group, dataset_name)
    if not current:
        return 0, 0

    src = resolve_source_path(current, recordings_root, dataset_root, role=role)
    if not src.exists():
        print(f"Missing source file, skipping: {src}")
        return 0, 1

    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / src.name

    if move_files:
        if src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))
    else:
        if src.resolve() != dst.resolve():
            if dst.exists():
                print(f"Destination file already exists, skipping copy: {dst}")
            else:
                shutil.copy2(src, dst)

    h5_path = f"{h5_relative_prefix.rstrip('/')}/{src.name}"
    write_string_dataset(sensor_group, dataset_name, h5_path)
    return 1, 0


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5).resolve()
    recordings_root = Path(args.recordings_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not h5_path.exists():
        print(f"HDF5 file not found: {h5_path}")
        return 1
    if not recordings_root.exists():
        print(f"recordings-root not found: {recordings_root}")
        return 1

    updated = 0
    missing = 0

    with h5py.File(h5_path, "a") as h5f:
        data_root = h5f.get("data")
        if data_root is None:
            print("HDF5 has no /data group.")
            return 1

        for family_name, data_name, data_group in iter_data_groups(data_root):
            repetitions = data_group.get("repetitions")
            if repetitions is None:
                continue

            for rep_name in sorted(repetitions.keys()):
                rep_group = repetitions[rep_name]
                sensors_data = rep_group.get("sensors_data")
                if sensors_data is None:
                    continue

                sensor_groups = []
                head = sensors_data.get("head_camera")
                if head is not None:
                    sensor_groups.append(("head_camera", head))
                grippers = sensors_data.get("grippers")
                if grippers is not None:
                    for gname in ("right_gripper", "left_gripper"):
                        gg = grippers.get(gname)
                        if gg is not None:
                            sensor_groups.append((gname, gg))

                for sensor_name, sensor_group in sensor_groups:
                    role = role_from_sensor_name(sensor_name)
                    if role is None:
                        continue

                    if family_name is None:
                        data_leaf = data_folder_name(data_name)
                    else:
                        data_leaf = f"{family_name}/{data_name}"
                    target_dir = (
                        output_root / "data" / data_leaf / rep_name / role
                    )
                    rel_prefix = f"data/{data_leaf}/{rep_name}/{role}"

                    u, m = organize_one_path(
                        sensor_group=sensor_group,
                        dataset_name="rgb_video_path",
                        role=role,
                        recordings_root=recordings_root,
                        dataset_root=output_root,
                        target_dir=target_dir,
                        h5_relative_prefix=rel_prefix,
                        move_files=args.move,
                    )
                    updated += u
                    missing += m

                    u, m = organize_one_path(
                        sensor_group=sensor_group,
                        dataset_name="depth_video_path",
                        role=role,
                        recordings_root=recordings_root,
                        dataset_root=output_root,
                        target_dir=target_dir,
                        h5_relative_prefix=rel_prefix,
                        move_files=args.move,
                    )
                    updated += u
                    missing += m

    print(f"Updated HDF5 video paths: {updated}")
    print(f"Missing source files skipped: {missing}")
    print(f"Output hierarchy root: {output_root}")
    print(f"Mode: {'move' if args.move else 'copy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

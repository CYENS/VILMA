from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create / append VILMA HDF5 dataset")
    parser.add_argument("--json", required=True, help="Sessions JSON file")
    parser.add_argument(
        "--output",
        default="vilma_dataset.h5",
        help="Output HDF5 path (default: vilma_dataset.h5)",
    )
    parser.add_argument(
        "--tasks-info",
        default="tasks_info.json",
        help="Path to tasks_info JSON (default: tasks_info.json)",
    )
    parser.add_argument(
        "--recordings-root",
        default="recordings",
        help="Filesystem path that corresponds to JSON paths starting with recordings/... (default: recordings)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print why tracking/fingers datasets are skipped.",
    )
    parser.add_argument(
        "--clear-data",
        action="store_true",
        help="Remove existing /data group before export (recommended for a clean rebuild).",
    )
    return parser.parse_args()


def resolve_recordings_root(raw: Path, sessions_path: Path) -> Path:
    """Resolve relative roots against JSON dir, parent, then cwd."""
    if raw.is_absolute():
        return raw.resolve()
    candidates = [
        sessions_path.parent / raw,
        sessions_path.parent.parent / raw,
        Path.cwd() / raw,
    ]
    for c in candidates:
        try:
            r = c.resolve()
        except OSError:
            continue
        if r.exists():
            return r
    return (sessions_path.parent / raw).resolve()


def normalize_instruction(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def build_instruction_lookup(tasks_info: Dict) -> Dict[str, Tuple[str, str]]:
    lookup: Dict[str, Tuple[str, str]] = {}
    for task_id, task_info in tasks_info.items():
        for variant_id, variant_info in task_info.get("variants", {}).items():
            instruction = variant_info.get("task_instruction", "")
            key = normalize_instruction(instruction)
            if not key:
                continue
            lookup[key] = (task_id, variant_id)
    return lookup


def load_tasks_info(tasks_info_path: Path) -> Dict:
    with open(tasks_info_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("tasks_info JSON must be a dictionary at top level.")
    return data


def json_path_to_abs(
    recordings_root: Path,
    json_path: Optional[str],
    dataset_root: Optional[Path] = None,
) -> Optional[Path]:
    if not json_path:
        return None
    normalized = str(json_path).replace("\\", "/")
    if normalized.startswith("recordings/"):
        rel = normalized[len("recordings/") :]
        return (recordings_root / rel).resolve()
    if normalized.startswith("data/") and dataset_root is not None:
        return (dataset_root / normalized).resolve()
    return Path(normalized).resolve()


def _norm_col_name(c: str) -> str:
    """Lowercase key for column lookup; spaces/dots/dashes -> underscores."""
    s = str(c).strip().lstrip("\ufeff").lower()
    s = re.sub(r"[\s\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def read_csv_df(csv_path: Path) -> pd.DataFrame:
    """Read CSV with tolerant encoding; normalize column names."""
    last_err: Optional[BaseException] = None
    df: Optional[pd.DataFrame] = None
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"Failed to read {csv_path}: {last_err}")
    df.columns = [_norm_col_name(str(c)) for c in df.columns]
    if len(df.columns) == 1:
        try:
            df2 = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
            df2.columns = [_norm_col_name(str(c)) for c in df2.columns]
            if len(df2.columns) > 1:
                df = df2
        except Exception:
            pass
    return df


def _column_norm_map(df: pd.DataFrame) -> Dict[str, str]:
    return {_norm_col_name(c): c for c in df.columns}


def _resolve_tracker_xyz(
    norm_map: Dict[str, str], tracker: str
) -> Optional[List[str]]:
    t = tracker.lower()
    keys = [f"{t}_x", f"{t}_y", f"{t}_z"]
    if all(k in norm_map for k in keys):
        return [norm_map[k] for k in keys]
    for norm, orig in norm_map.items():
        if not norm.endswith("_x"):
            continue
        p = norm[:-2]
        yk, zk = f"{p}_y", f"{p}_z"
        if yk not in norm_map or zk not in norm_map:
            continue
        if p == t or p.endswith("_" + t) or p.startswith(t + "_"):
            return [orig, norm_map[yk], norm_map[zk]]
    return None


def _resolve_tracker_orientation(
    norm_map: Dict[str, str], tracker: str
) -> Optional[List[str]]:
    t = tracker.lower()
    keys = [f"{t}_yaw", f"{t}_pitch", f"{t}_roll"]
    if all(k in norm_map for k in keys):
        return [norm_map[k] for k in keys]
    for norm, orig in norm_map.items():
        if not norm.endswith("_yaw"):
            continue
        p = norm[: -len("_yaw")]
        pk, rk = f"{p}_pitch", f"{p}_roll"
        if pk in norm_map and rk in norm_map:
            if p == t or p.endswith("_" + t) or p.startswith(t + "_"):
                return [orig, norm_map[pk], norm_map[rk]]
    return None


def create_or_replace_dataset(group: h5py.Group, name: str, data, dtype=None) -> None:
    if name in group:
        del group[name]
    if dtype == STRING_DTYPE:
        group.create_dataset(name, data=str(data), dtype=STRING_DTYPE)
        return
    arr = np.ascontiguousarray(np.asarray(data))
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
        group.create_dataset(name, data=arr, dtype=dtype)
    else:
        group.create_dataset(name, data=arr)


def set_string_dataset(group: h5py.Group, name: str, value: str) -> None:
    create_or_replace_dataset(group, name, str(value), dtype=STRING_DTYPE)


def ensure_base_structure(h5f: h5py.File) -> None:
    tasks_group = h5f.require_group("tasks_info")
    data_group = h5f.require_group("data")
    _ = tasks_group
    _ = data_group


def populate_tasks_info_group(h5f: h5py.File, tasks_info: Dict) -> None:
    tasks_group = h5f.require_group("tasks_info")
    for task_id, task_info in tasks_info.items():
        task_group = tasks_group.require_group(task_id)
        set_string_dataset(task_group, "task_family", task_info.get("task_family", ""))
        variants_group = task_group.require_group("variants")
        for variant_id, variant_info in task_info.get("variants", {}).items():
            var_group = variants_group.require_group(variant_id)
            set_string_dataset(
                var_group,
                "task_instruction",
                variant_info.get("task_instruction", ""),
            )


def _resolve_columns_case_insensitive(
    df: pd.DataFrame, logical_names: list[str]
) -> Optional[list[str]]:
    cmap = {_norm_col_name(c): c for c in df.columns}
    out: list[str] = []
    for name in logical_names:
        key = _norm_col_name(name)
        if key not in cmap:
            return None
        out.append(cmap[key])
    return out


def read_tracking_arrays(
    csv_path: Path, tracker: str, verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not csv_path.exists():
        return None, None
    try:
        df = read_csv_df(csv_path)
        norm_map = _column_norm_map(df)
        position = None
        orientation = None
        pos_cols = _resolve_tracker_xyz(norm_map, tracker)
        if pos_cols is not None:
            position = (
                df[pos_cols]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float32)
            )
        ori_cols = _resolve_tracker_orientation(norm_map, tracker)
        if ori_cols is not None:
            orientation = (
                df[ori_cols]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float32)
            )
        return position, orientation
    except Exception as e:
        if verbose:
            print(f"    [tracking] failed to read {csv_path}: {e}")
        return None, None


def find_three_axis_columns(
    columns: Iterable[str], keywords: Iterable[str]
) -> Optional[list[str]]:
    matches = []
    lower_to_original = {c.lower(): c for c in columns}
    for key in keywords:
        for axis in ("x", "y", "z"):
            exact = f"{key}_{axis}"
            if exact in lower_to_original:
                matches.append(lower_to_original[exact])
    if len(matches) == 3:
        return matches

    contains = []
    for col in columns:
        col_lower = col.lower()
        if any(key in col_lower for key in keywords):
            contains.append(col)
    if len(contains) >= 3:
        return contains[:3]
    return None


def read_finger_distance(
    csv_path: Optional[Path], verbose: bool = False
) -> Optional[np.ndarray]:
    if csv_path is None or not csv_path.exists():
        return None
    try:
        df = read_csv_df(csv_path)
    except Exception as e:
        if verbose:
            print(f"    [fingers] failed to read {csv_path}: {e}")
        return None
    cmap = {_norm_col_name(c): c for c in df.columns}
    try:
        for key in ("fingers_dist_cm", "fingers_distance_cm", "finger_distance_cm"):
            if key in cmap:
                return (
                    df[cmap[key]]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype=np.float32)
                )
    except Exception:
        return None
    return None


def choose_tracking_path(variant: Dict, recordings_root: Path) -> Optional[Path]:
    tracking = variant.get("tracking", {})
    rel = tracking.get("edited_tracking_path") or tracking.get("tracking_path")
    return json_path_to_abs(recordings_root, rel)


def choose_fingers_path(video: Dict, recordings_root: Path) -> Optional[Path]:
    rel = video.get("edited_fingers_dist_path") or video.get("fingers_dist_path")
    return json_path_to_abs(recordings_root, rel)


def video_by_role(variant: Dict, role: str) -> Optional[Dict]:
    for v in variant.get("videos", []):
        if str(v.get("role", "")).lower() == role.lower():
            return v
    return None


def sensor_group_for_role(repetition_group: h5py.Group, role: str) -> h5py.Group:
    sensors = repetition_group.require_group("sensors_data")
    if role == "head":
        return sensors.require_group("head_camera")
    grippers = sensors.require_group("grippers")
    return grippers.require_group(f"{role}_gripper")


def data_leaf_from_h5_data_group(data_group: h5py.Group) -> str:
    """
    Build logical folder under data/ used by video paths.

    Example:
      /data/D_C01/D_C01.01 -> D_C01/D_C01.01
    """
    name = Path(data_group.name).name
    parent = Path(data_group.parent.name).name
    if parent != "data":
        return f"{parent}/{name}"
    return name


def basename_from_sessions_path(path_str: str) -> str:
    if not path_str:
        return ""
    return Path(str(path_str).replace("\\", "/")).name


def choose_rgb_source_path(video: Dict, recordings_root: Path) -> str:
    rgb_src = video.get("synced_video_path") or video.get("video_path") or ""
    if not rgb_src:
        return ""
    norm = str(rgb_src).replace("\\", "/")
    lower = norm.lower()
    if "_synced.mp4" in lower:
        idx = lower.rfind("_synced.mp4")
        compressed_norm = norm[:idx] + "_compressed" + norm[idx + len("_synced") :]
        compressed_abs = json_path_to_abs(recordings_root, compressed_norm)
        if compressed_abs is not None and compressed_abs.exists():
            return compressed_norm
    return str(rgb_src)


def choose_depth_source_path(video: Dict, recordings_root: Path) -> str:
    depth_src = video.get("depth_path", "")
    if not depth_src:
        return ""
    norm = str(depth_src).replace("\\", "/")
    lower = norm.lower()
    marker = "_synced_depth.mp4"
    if marker in lower:
        idx = lower.rfind(marker)
        return norm[:idx] + "_depth_compressed.MP4"
    return str(depth_src)


def add_video_paths(
    target_group: h5py.Group,
    video: Dict,
    recordings_root: Path,
    data_leaf: str,
    rep_name: str,
    role: str,
) -> None:
    """
    Store paths relative to dataset root, starting with data/
    Layout: data/{data_leaf}/{rep_name}/{role}/{filename}
    """
    rgb_src = choose_rgb_source_path(video, recordings_root)
    depth_src = choose_depth_source_path(video, recordings_root)
    rgb_name = basename_from_sessions_path(rgb_src)
    depth_name = basename_from_sessions_path(depth_src)
    prefix = f"data/{data_leaf}/{rep_name}/{role}".replace("\\", "/")
    rgb_h5 = f"{prefix}/{rgb_name}" if rgb_name else ""
    depth_h5 = f"{prefix}/{depth_name}" if depth_name else ""
    set_string_dataset(target_group, "rgb_video_path", rgb_h5)
    set_string_dataset(target_group, "depth_video_path", depth_h5)


def add_tracking_data(
    target_group: h5py.Group,
    tracking_csv: Optional[Path],
    tracker: str,
    verbose: bool,
    label: str,
) -> None:
    if tracking_csv is None:
        if verbose:
            print(f"    [tracking] {label}: no tracking CSV path")
        return
    if not tracking_csv.exists():
        if verbose:
            print(f"    [tracking] {label}: file not found: {tracking_csv}")
        return
    position, orientation = read_tracking_arrays(tracking_csv, tracker, verbose)
    if position is None and orientation is None:
        if verbose:
            print(
                f"    [tracking] {label}: no columns for tracker '{tracker}' in {tracking_csv}"
            )
        return
    try:
        tracking_group = target_group.require_group("tracking")
        if position is not None:
            create_or_replace_dataset(
                tracking_group, "position", position, dtype=np.float32
            )
        if orientation is not None:
            create_or_replace_dataset(
                tracking_group, "orientation", orientation, dtype=np.float32
            )
    except Exception as e:
        print(f"    [tracking] {label}: failed to write HDF5 datasets: {e}")

def add_fingers_data(
    target_group: h5py.Group, fingers_csv: Optional[Path], verbose: bool, label: str
) -> None:
    fingers = read_finger_distance(fingers_csv, verbose)
    if fingers is None:
        if verbose:
            if fingers_csv is None:
                print(f"    [fingers] {label}: no fingers CSV path")
            elif not fingers_csv.exists():
                print(f"    [fingers] {label}: file not found: {fingers_csv}")
            else:
                print(f"    [fingers] {label}: no fingers column in {fingers_csv}")
        return
    try:
        create_or_replace_dataset(
            target_group, "finger_distance_cm", fingers, dtype=np.float32
        )
    except Exception as e:
        print(f"    [fingers] {label}: failed to write HDF5 dataset: {e}")


def variant_hand_usage(variant: Dict) -> str:
    trackers = {str(x).lower() for x in variant.get("trackers_used", [])}
    if "left" in trackers and "right" in trackers:
        return "bimanual"
    if "left" in trackers:
        return "unimanual_left"
    if "right" in trackers:
        return "unimanual_right"
    return "unknown"


def repetition_name(
    task_id: str, data_group_name: str, repetition_idx: int
) -> str:
    """
    Format: R_C01.01.01
      - task_id: C01
      - data_group_name: D_C01.01
      - repetition_idx: 1 -> 01
    """
    data_suffix = "01"
    prefix = f"D_{task_id}."
    if data_group_name.startswith(prefix):
        data_suffix = data_group_name[len(prefix) :]
    return f"R_{task_id}.{data_suffix}.{repetition_idx:02d}"


def find_or_create_data_group(
    data_root: h5py.Group,
    family_group_name: str,
    task_id: str,
    participant_id: str,
    variant_id: str,
    location: str,
) -> h5py.Group:
    family_group = data_root.require_group(family_group_name)

    for name in sorted(family_group.keys()):
        group = family_group[name]
        if (
            group.attrs.get("participant_id", "") == participant_id
            and group.attrs.get("task_id", "") == task_id
            and group.attrs.get("variant_id", "") == variant_id
            and group.attrs.get("location", "") == location
        ):
            return group

    # D_C01.01, D_C01.02, ...
    max_idx = 0
    prefix = f"D_{task_id}."
    for group_name in family_group.keys():
        if not group_name.startswith(prefix):
            continue
        suffix = group_name[len(prefix) :]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))

    next_idx = max_idx + 1
    group = family_group.require_group(f"D_{task_id}.{next_idx:02d}")
    group.attrs["participant_id"] = participant_id
    group.attrs["task_id"] = task_id
    group.attrs["variant_id"] = variant_id
    group.attrs["location"] = location
    group.require_group("repetitions")
    return group


def add_variant_to_hdf5(
    h5f: h5py.File,
    variant: Dict,
    task_id: str,
    variant_h5_id: str,
    recordings_root: Path,
    verbose: bool,
) -> None:
    participant_id = f"P{variant.get('user_id', 'unknown')}"
    location = str(variant.get("location", "unknown"))
    family_group_name = f"D_{task_id}"
    data_group = find_or_create_data_group(
        h5f["data"],
        family_group_name,
        task_id,
        participant_id,
        variant_h5_id,
        location,
    )
    repetitions_group = data_group.require_group("repetitions")
    repetition_idx = len(repetitions_group) + 1
    rep_name = repetition_name(task_id, Path(data_group.name).name, repetition_idx)
    repetition_group = repetitions_group.require_group(rep_name)

    repetition_info = repetition_group.require_group("repetition_info")
    set_string_dataset(
        repetition_info, "unimanual_or_bimanual", variant_hand_usage(variant)
    )

    data_leaf = data_leaf_from_h5_data_group(data_group)

    tracking_csv = choose_tracking_path(variant, recordings_root)

    for role in ("head", "right", "left"):
        video = video_by_role(variant, role)
        if video is None:
            continue
        target_group = sensor_group_for_role(repetition_group, role)
        add_video_paths(target_group, video, recordings_root, data_leaf, rep_name, role)
        if role in ("right", "left"):
            label = f"{rep_name}/{role}"
            add_tracking_data(target_group, tracking_csv, role, verbose, label)
            add_fingers_data(
                target_group,
                choose_fingers_path(video, recordings_root),
                verbose,
                label,
            )


def main() -> int:
    args = parse_args()
    sessions_path = Path(args.json).resolve()
    output_path = Path(args.output).resolve()
    tasks_info_path = Path(args.tasks_info).resolve()

    recordings_root = resolve_recordings_root(Path(args.recordings_root), sessions_path)
    if args.verbose:
        print(
            f"Resolved recordings root: {recordings_root} (exists: {recordings_root.exists()})"
        )
    if not recordings_root.exists():
        print(
            f"Warning: recordings-root does not exist: {recordings_root}\n"
            "RGB/depth paths in HDF5 are still written from filenames, but tracking/fingers "
            "need this folder to contain the CSV files referenced under recordings/ in the JSON."
        )

    if not tasks_info_path.exists():
        print(f"tasks_info file not found: {tasks_info_path}")
        return 1

    try:
        tasks_info = load_tasks_info(tasks_info_path)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Failed to load tasks_info JSON: {e}")
        return 1

    instruction_lookup = build_instruction_lookup(tasks_info)
    if not instruction_lookup:
        print("tasks_info is empty. Fill tasks_info.json before exporting.")
        return 1

    with open(sessions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sessions = data if isinstance(data, list) else [data]

    exported = 0
    skipped = 0
    skipped_task_failure = 0
    skipped_instructions: List[str] = []

    with h5py.File(output_path, "a") as h5f:
        if args.clear_data and "data" in h5f:
            del h5f["data"]
        ensure_base_structure(h5f)
        populate_tasks_info_group(h5f, tasks_info)

        for session in sessions:
            instruction_text = session.get("instruction", {}).get("text_instruction", "")
            task_key = normalize_instruction(instruction_text)
            for variant in session.get("variants", []):
                if variant.get("task_failure") is True:
                    skipped_task_failure += 1
                    continue
                if task_key not in instruction_lookup:
                    skipped += 1
                    skipped_instructions.append(instruction_text)
                    continue
                task_id, variant_h5_id = instruction_lookup[task_key]
                add_variant_to_hdf5(
                    h5f,
                    variant,
                    task_id,
                    variant_h5_id,
                    recordings_root,
                    args.verbose,
                )
                exported += 1

    print(f"Exported variants into HDF5 repetitions: {exported}")
    print(f"Skipped variants (instruction not in tasks_info): {skipped}")
    print(f"Skipped variants (task_failure=true): {skipped_task_failure}")
    if skipped_instructions:
        unique = sorted({str(x) for x in skipped_instructions})
        print("Missing TASKS_INFO mappings for instructions:")
        for item in unique:
            print(f"- {item}")
    print(f"Recordings root used for loading sensor files: {recordings_root}")
    print(f"Output file: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

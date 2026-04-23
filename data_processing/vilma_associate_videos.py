import argparse
import copy
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


ROLE_ORDER = ["left", "right", "head"]
DISTANCE_CM = 4.25


def collect_variants(sessions: List[dict]) -> List[dict]:
    variants: List[dict] = []
    for session in sessions:
        variants.extend(session.get("variants", []))
    return variants


def list_video_files(folder_path: Optional[str]) -> List[str]:
    if folder_path is None:
        return []
    if not os.path.isdir(folder_path):
        raise ValueError(f"Invalid folder path: {folder_path}")

    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    files = [f for f in files if f.lower().endswith(".mp4")]
    files.sort(key=video_sort_key)
    return files


def video_sort_key(filename: str) -> Tuple[int, str]:
    stem, _ext = os.path.splitext(filename)
    digits = re.findall(r"\d+", stem)
    if digits:
        # Sort primarily by numeric sequence in filename, ignoring prefix letters.
        return (int(digits[-1]), filename)
    return (-1, filename)


def print_files_with_numbers(role: str, files: List[str]) -> None:
    print(f"\n{role} videos:")
    for i, filename in enumerate(files, start=1):
        print(f"{i}. {filename}")
    print(f"Number of {role} videos: {len(files)}")


def windows_video_path(role: str, filename: str) -> str:
    return f"recordings\\{role}\\{filename}"


def basename_from_path(path: str) -> str:
    if not isinstance(path, str):
        return ""
    normalized = path.replace("\\", "/")
    return os.path.basename(normalized)


def is_correct_video_entry(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    role = entry.get("role")
    path = entry.get("video_path")
    video_id = entry.get("video_id")
    if role not in ROLE_ORDER:
        return False
    if not isinstance(path, str) or not path:
        return False
    if not isinstance(video_id, int):
        return False

    if role in ("left", "right"):
        if entry.get("closed_fingers_tag_distance_cm") != DISTANCE_CM:
            return False
    else:
        if "closed_fingers_tag_distance_cm" in entry:
            return False
    return True


def parse_existing_videos(
    cameras_used: List[str], videos: List
) -> Tuple[Dict[str, str], bool]:
    """
    Returns:
      role_to_filename, already_correct_format
    """
    role_to_filename: Dict[str, str] = {}
    already_correct_format = True

    if not isinstance(videos, list):
        return role_to_filename, False

    has_dict_entries = any(isinstance(v, dict) for v in videos)
    has_legacy_entries = any(isinstance(v, list) for v in videos)

    if has_dict_entries:
        for entry in videos:
            if not isinstance(entry, dict):
                already_correct_format = False
                continue
            role = entry.get("role")
            if role not in ROLE_ORDER:
                already_correct_format = False
                continue
            filename = basename_from_path(entry.get("video_path", ""))
            if filename:
                role_to_filename[role] = filename
            if not is_correct_video_entry(entry):
                already_correct_format = False
    elif has_legacy_entries:
        already_correct_format = False
        # Legacy nested-list entries follow the cameras_used order.
        expected_roles = [r for r in cameras_used if r in ROLE_ORDER]
        for idx, legacy_item in enumerate(videos):
            if idx >= len(expected_roles):
                break
            if (
                isinstance(legacy_item, list)
                and legacy_item
                and isinstance(legacy_item[0], str)
            ):
                filename = basename_from_path(legacy_item[0])
                if filename:
                    role_to_filename[expected_roles[idx]] = filename
    else:
        already_correct_format = False

    expected_roles_set = set(cameras_used)
    if expected_roles_set:
        current_roles_set = set(role_to_filename.keys())
        if not current_roles_set.issuperset(expected_roles_set):
            already_correct_format = False
    return role_to_filename, already_correct_format


def build_entry(video_id: int, role: str, filename: str) -> dict:
    entry = {
        "video_id": video_id,
        "video_path": windows_video_path(role, filename),
        "role": role,
    }
    if role in ("left", "right"):
        entry["closed_fingers_tag_distance_cm"] = DISTANCE_CM
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Associate left/right/head videos to variants in a sessions JSON."
    )
    parser.add_argument("--json", required=True, help="Path to input/output sessions JSON.")
    parser.add_argument("--left", help="Folder with left camera videos.")
    parser.add_argument("--right", help="Folder with right camera videos.")
    parser.add_argument("--head", help="Folder with head camera videos.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing the JSON file.",
    )
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    variants = collect_variants(sessions)
    if not variants:
        print("No variants found in JSON.")
        return 0
    print(f"Number of variants: {len(variants)}")

    role_folders = {"left": args.left, "right": args.right, "head": args.head}
    role_files: Dict[str, List[str]] = {}
    for role in ROLE_ORDER:
        role_files[role] = list_video_files(role_folders[role])
        print_files_with_numbers(role, role_files[role])

    parsed_variants = []
    role_preassigned: Dict[str, List[str]] = {r: [] for r in ROLE_ORDER}
    missing_count: Dict[str, int] = {r: 0 for r in ROLE_ORDER}
    needed_count: Dict[str, int] = {r: 0 for r in ROLE_ORDER}

    for variant in variants:
        cameras_used = variant.get("cameras_used", [])
        cameras_used = [c for c in cameras_used if c in ROLE_ORDER]
        videos = variant.get("videos", [])

        role_to_filename, already_correct = parse_existing_videos(cameras_used, videos)
        parsed_variants.append((variant, cameras_used, role_to_filename, already_correct))

        for role in cameras_used:
            needed_count[role] += 1
            filename = role_to_filename.get(role)
            if filename:
                role_preassigned[role].append(filename)
            else:
                missing_count[role] += 1

    print("Camera usage counts from cameras_used:")
    for role in ROLE_ORDER:
        print(f"- {role}: {needed_count[role]}")

    role_assignment_queues: Dict[str, List[str]] = {}
    for role in ROLE_ORDER:
        available = role_files[role][:]
        if not role_folders[role]:
            print(
                f"Strict counts failed for '{role}': folder missing (--{role}) "
                f"but {needed_count[role]} assignments are required."
            )
            return 1
        # Strict check: all MP4 files in folder must match role usage count.
        if len(role_files[role]) != needed_count[role]:
            print(
                f"Strict counts failed for '{role}': files in folder={len(role_files[role])}, "
                f"required assignments={needed_count[role]}."
            )
            return 1
        if missing_count[role] > 0:
            if not role_folders[role]:
                print(
                    f"Missing {missing_count[role]} '{role}' assignments but no --{role} folder was provided."
                )
                return 1
            if len(available) < missing_count[role]:
                print(
                    f"Not enough '{role}' videos. Missing: {missing_count[role]}, "
                    f"available for assignment: {len(available)} (total files: {len(role_files[role])})."
                )
                return 1
        role_assignment_queues[role] = available

    role_indices = {r: 0 for r in ROLE_ORDER}
    changed = 0
    skipped_correct = 0
    dry_run_updates = []

    for variant, cameras_used, role_to_filename, already_correct in parsed_variants:
        # Keep output order aligned with cameras_used for each variant.
        ordered_roles = [r for r in cameras_used if r in ROLE_ORDER]
        new_videos = []
        for role in ordered_roles:
            queue = role_assignment_queues[role]
            idx = role_indices[role]
            if idx >= len(queue):
                print(f"Ran out of available '{role}' videos during assignment.")
                return 1
            filename = queue[idx]
            role_indices[role] = idx + 1
            new_videos.append(build_entry(video_id=len(new_videos) + 1, role=role, filename=filename))

        old_videos = copy.deepcopy(variant.get("videos", []))
        if old_videos == new_videos:
            skipped_correct += 1
            continue

        variant["videos"] = new_videos
        changed += 1

        if args.dry_run:
            dry_run_updates.append(
                {
                    "session_id": variant.get("session_id", "unknown"),
                    "variant_id": variant.get("variant_id", "unknown"),
                    "cameras_used": cameras_used,
                    "old_videos": old_videos,
                    "new_videos": new_videos,
                }
            )

    print(f"Total variants: {len(variants)}")
    print(f"Variants already in correct format (skipped): {skipped_correct}")
    print(f"Variants updated: {changed}")
    print("\nPer-role summary:")
    for role in ROLE_ORDER:
        total_files = len(role_files[role])
        already_assigned = len(role_preassigned[role])
        to_assign = missing_count[role]
        unused_files = total_files - needed_count[role]
        print(
            f"- {role}: required={needed_count[role]}, already_assigned={already_assigned}, "
            f"to_assign={to_assign}, files_in_folder={total_files}, unused_files={unused_files}"
        )

    if args.dry_run:
        if dry_run_updates:
            print("\nPlanned updates:")
            for idx, update in enumerate(dry_run_updates, start=1):
                print(f"\n[{idx}] variant_id={update['variant_id']}")
                print(f"cameras_used: {update['cameras_used']}")
                print("old videos:")
                print(json.dumps(update["old_videos"], indent=4))
                print("new videos:")
                print(json.dumps(update["new_videos"], indent=4))
        else:
            print("\nNo variant updates needed.")
        print("Dry run complete. No file written.")
        return 0

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=4)
        f.write("\n")
    print(f"Updated JSON written to: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

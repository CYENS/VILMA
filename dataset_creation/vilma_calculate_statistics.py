import json
import os
from pathlib import Path
from collections import defaultdict
import argparse
import re
import textwrap


def calculate_task_durations(json_files):
    """
    Read JSON files and sum tracking_duration for each unique task (instruction).
    
    Args:
        json_files: List of file paths to JSON files
    
    Returns:
        Dictionary with task instructions as keys and dict containing count and duration
    """
    task_stats = defaultdict(
        lambda: {
            'count': 0,
            'duration': 0.0,
            'unimanual_right': 0,
            'unimanual_left': 0,
            'bimanual': 0,
        }
    )
    ignored = defaultdict(int)
    tasks_per_location = defaultdict(
        lambda: {
            'count': 0,
            'duration': 0.0,
            'unimanual_right': 0,
            'unimanual_left': 0,
            'bimanual': 0,
        }
    )
    tasks_per_user = defaultdict(
        lambda: {
            'count': 0,
            'duration': 0.0,
            'unimanual_right': 0,
            'unimanual_left': 0,
            'bimanual': 0,
        }
    )
    
    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"Warning: File not found: {json_file}")
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file}: {e}")
            continue
        
        # Handle both single session and list of sessions
        sessions = data if isinstance(data, list) else [data]
        
        for session in sessions:
            # Get the instruction text
            instruction = session.get('instruction', {})
            task_name = instruction.get('text_instruction', 'Unknown')
            task_name_lower = str(task_name).lower()
            
            # Sum tracking_duration from all variants
            variants = session.get('variants', [])
            for variant in variants:
                tracking = variant.get('tracking', {})
                duration = tracking.get('tracking_duration', 0)
                trackers_used = variant.get('trackers_used', [])
                role_set = {str(role).lower() for role in trackers_used}
                location = str(variant.get('location', 'unknown'))
                raw_user_id = variant.get('user_id', 'unknown')
                if isinstance(raw_user_id, str) and raw_user_id.isdigit():
                    user_id = f"P{int(raw_user_id):02d}"
                elif isinstance(raw_user_id, int):
                    user_id = f"P{raw_user_id:02d}"
                else:
                    user_id = f"P{raw_user_id}"

                # Ignore explicit failure variants.
                if variant.get('task_failure') is True:
                    ignored['task_failure'] += 1
                    continue

                # Ignore missing instructions.
                if task_name == 'Unknown' or not str(task_name).strip():
                    ignored['no_instruction'] += 1
                    continue

                # Ignore test/demo instructions.
                if 'test' in task_name_lower:
                    ignored['testing_instruction'] += 1
                    continue
                
                # Only add if duration is a number
                if isinstance(duration, (int, float)):
                    task_stats[task_name]['duration'] += duration
                    task_stats[task_name]['count'] += 1
                    tasks_per_location[location]['duration'] += duration
                    tasks_per_location[location]['count'] += 1
                    tasks_per_user[user_id]['duration'] += duration
                    tasks_per_user[user_id]['count'] += 1
                    if 'left' in role_set and 'right' in role_set:
                        task_stats[task_name]['bimanual'] += 1
                        tasks_per_location[location]['bimanual'] += 1
                        tasks_per_user[user_id]['bimanual'] += 1
                    elif 'right' in role_set:
                        task_stats[task_name]['unimanual_right'] += 1
                        tasks_per_location[location]['unimanual_right'] += 1
                        tasks_per_user[user_id]['unimanual_right'] += 1
                    elif 'left' in role_set:
                        task_stats[task_name]['unimanual_left'] += 1
                        tasks_per_location[location]['unimanual_left'] += 1
                        tasks_per_user[user_id]['unimanual_left'] += 1
                else:
                    ignored['invalid_or_missing_duration'] += 1
    
    return task_stats, ignored, tasks_per_location, tasks_per_user


def main():
    parser = argparse.ArgumentParser(
        description='Calculate total tracking duration per task from JSON session files'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='JSON file(s) to process (e.g., sessions.json sessions_.json)'
    )
    parser.add_argument(
        '--sort',
        choices=['name', 'duration'],
        default='name',
        help='Sort results by task name or duration (default: name)'
    )
    parser.add_argument(
        '--tasks-info-output',
        default='vilma_tasks_info.json',
        help='Output JSON file for tasks_info mapping (default: vilma_tasks_info.json)'
    )
    
    args = parser.parse_args()
    
    # Calculate durations
    task_stats, ignored, tasks_per_location, tasks_per_user = calculate_task_durations(args.files)
    
    if not task_stats:
        print("No data found in provided files.")
        return
    
    # Sort results
    if args.sort == 'duration':
        sorted_tasks = sorted(task_stats.items(), key=lambda x: x[1]['duration'], reverse=True)
    else:
        sorted_tasks = sorted(task_stats.items())
    
    # Final task categories (explicit labels + keyword matching)
    categories = [
        {"label": "Open drawer and stow an item.", "keyword": "drawer", "count": 0, "duration": 0.0, "unimanual_right": 0, "unimanual_left": 0, "bimanual": 0, "tasks": []},
        {"label": "Fold clothes and stack.", "keyword": "fold", "count": 0, "duration": 0.0, "unimanual_right": 0, "unimanual_left": 0, "bimanual": 0, "tasks": []},
        {"label": "Spray and wipe a small area.", "keyword": "wipe", "count": 0, "duration": 0.0, "unimanual_right": 0, "unimanual_left": 0, "bimanual": 0, "tasks": []},
        {"label": "Hang a towel.", "keyword": "hang", "count": 0, "duration": 0.0, "unimanual_right": 0, "unimanual_left": 0, "bimanual": 0, "tasks": []},
        {"label": "Place a cup on coaster.", "keyword": "coaster", "count": 0, "duration": 0.0, "unimanual_right": 0, "unimanual_left": 0, "bimanual": 0, "tasks": []},
        {"label": "Charge a phone.", "keyword": "charge", "count": 0, "duration": 0.0, "unimanual_right": 0, "unimanual_left": 0, "bimanual": 0, "tasks": []},
        {"label": "Prepare a glass of water.", "keyword": "water", "count": 0, "duration": 0.0, "unimanual_right": 0, "unimanual_left": 0, "bimanual": 0, "tasks": []},
        {"label": "Open washing machine, place clothes, and close.", "keyword": "washing machine", "count": 0, "duration": 0.0, "unimanual_right": 0, "unimanual_left": 0, "bimanual": 0, "tasks": []},
        {"label": "Throw trash in the bin.", "keyword": "trash", "count": 0, "duration": 0.0, "unimanual_right": 0, "unimanual_left": 0, "bimanual": 0, "tasks": []},
        {"label": "Open fridge, place item, and close the fridge.", "keyword": "fridge", "count": 0, "duration": 0.0, "unimanual_right": 0, "unimanual_left": 0, "bimanual": 0, "tasks": []},
        {"label": "Open dishwasher, place or take out item, and close.", "keyword": "dishwasher", "count": 0, "duration": 0.0, "unimanual_right": 0, "unimanual_left": 0, "bimanual": 0, "tasks": []},
    ]
    category_task_ids = {cat["label"]: f"C{i+1}" for i, cat in enumerate(categories)}

    def normalize_for_keyword_match(s: str) -> str:
        # Treat underscores the same as spaces (some task names may use either).
        return s.lower().replace("_", " ")

    total_duration = 0
    total_count = 0
    total_uni_r = 0
    total_uni_l = 0
    total_bi = 0

    # Assign each (non-ignored) specific task to the first matching category
    # based on the order in `categories`.
    for task, stats in sorted_tasks:
        count = stats["count"]
        duration = stats["duration"]
        uni_r = stats["unimanual_right"]
        uni_l = stats["unimanual_left"]
        bi = stats["bimanual"]

        total_duration += duration
        total_count += count
        total_uni_r += uni_r
        total_uni_l += uni_l
        total_bi += bi

        task_norm = normalize_for_keyword_match(task)

        for cat in categories:
            keyword_norm = normalize_for_keyword_match(cat["keyword"])
            if keyword_norm in task_norm:
                cat["count"] += count
                cat["duration"] += duration
                cat["unimanual_right"] += uni_r
                cat["unimanual_left"] += uni_l
                cat["bimanual"] += bi
                cat["tasks"].append((task, count, duration, uni_r, uni_l, bi))
                break

    # Sort category rows and subtask rows according to --sort mode.
    if args.sort == 'duration':
        categories = sorted(categories, key=lambda c: (c["duration"], c["count"]), reverse=True)
        for cat in categories:
            cat["tasks"] = sorted(cat["tasks"], key=lambda t: (t[2], t[1]), reverse=True)
    else:
        categories = sorted(categories, key=lambda c: c["label"].lower())
        for cat in categories:
            cat["tasks"] = sorted(cat["tasks"], key=lambda t: t[0].lower())

    # ANSI colors for terminal output (ignored if your terminal doesn't support them)
    ANSI_RESET = "\033[0m"
    # One color per category for subtask rows (label + count + duration)
    ANSI_CATEGORY_COLORS = [
        "\033[36m",  # cyan
        "\033[32m",  # green
        "\033[34m",  # blue
        "\033[35m",  # magenta
        "\033[33m",  # yellow
    ]

    task_width = 70
    line_width = 120

    def wrap_label(label: str, width: int) -> list[str]:
        """Wrap long labels across multiple lines instead of truncating."""
        if not label:
            return [""]
        return textwrap.wrap(str(label), width=width) or [str(label)]

    def print_metric_row(
        label: str,
        uni_r: int,
        uni_l: int,
        bi: int,
        count: int,
        duration: float,
        color: str = "",
    ) -> None:
        lines = wrap_label(label, task_width)
        first = (
            f"{lines[0]:<{task_width}}"
            f"{uni_r:>8}{uni_l:>8}{bi:>8}"
            f"{count:>10} {duration:>14.2f}"
        )
        if color:
            print(f"{color}{first}{ANSI_RESET}")
        else:
            print(first)
        continuation_indent = "      "
        for extra in lines[1:]:
            extra_line = continuation_indent + extra
            if color:
                print(f"{color}{extra_line:<{task_width}}{ANSI_RESET}")
            else:
                print(f"{extra_line:<{task_width}}")

    print()
    print(f"{'Task':<{task_width}}{'Uni-R':>8}{'Uni-L':>8}{'Bi':>8}{'Count':>10} {'Duration (s)':>14}")
    print("=" * line_width)

    for cat_idx, cat in enumerate(categories):
        sub_color = ANSI_CATEGORY_COLORS[cat_idx % len(ANSI_CATEGORY_COLORS)]
        print_metric_row(
            label=cat["label"],
            uni_r=cat["unimanual_right"],
            uni_l=cat["unimanual_left"],
            bi=cat["bimanual"],
            count=cat["count"],
            duration=cat["duration"],
            color=sub_color,
        )
        for task, count, duration, uni_r, uni_l, bi in cat["tasks"]:
            print_metric_row(
                label=f"    - {task}",
                uni_r=uni_r,
                uni_l=uni_l,
                bi=bi,
                count=count,
                duration=duration,
            )

    print("=" * line_width + "\n")

    print(f"{'Location':<{task_width}}{'Uni-R':>8}{'Uni-L':>8}{'Bi':>8}{'Count':>10} {'Duration (s)':>14}")
    print("=" * line_width)
    for idx, (location, stats) in enumerate(
        sorted(tasks_per_location.items(), key=lambda x: (-x[1]['count'], x[0]))
    ):
        row_color = ANSI_CATEGORY_COLORS[idx % len(ANSI_CATEGORY_COLORS)]
        print_metric_row(
            label=location,
            uni_r=stats['unimanual_right'],
            uni_l=stats['unimanual_left'],
            bi=stats['bimanual'],
            count=stats['count'],
            duration=stats['duration'],
            color=row_color,
        )
    print("=" * line_width + "\n")

    print(f"{'Participant':<{task_width}}{'Uni-R':>8}{'Uni-L':>8}{'Bi':>8}{'Count':>10} {'Duration (s)':>14}")
    print("=" * line_width)
    for idx, (user_id, stats) in enumerate(
        sorted(tasks_per_user.items(), key=lambda x: (-x[1]['count'], x[0]))
    ):
        row_color = ANSI_CATEGORY_COLORS[idx % len(ANSI_CATEGORY_COLORS)]
        print_metric_row(
            label=user_id,
            uni_r=stats['unimanual_right'],
            uni_l=stats['unimanual_left'],
            bi=stats['bimanual'],
            count=stats['count'],
            duration=stats['duration'],
            color=row_color,
        )
    print("=" * line_width + "\n")

    print(
        f"{'TOTAL':<{task_width}}"
        f"{total_uni_r:>8}{total_uni_l:>8}{total_bi:>8}"
        f"{total_count:>10} {total_duration:>14.2f}"
    )
    print(f"TOTAL duration in minutes: {total_duration/60.0:.2f} min")
    print(f"\033[31mTOTAL duration in hours: {total_duration/3600.0:.2f} h\033[0m\n")

    ignored_total = sum(ignored.values())
    print("Ignored variants summary:")
    print(f"- task_failure=true: {ignored.get('task_failure', 0)}")
    print(f"- no instruction: {ignored.get('no_instruction', 0)}")
    print(f"- testing instruction: {ignored.get('testing_instruction', 0)}")
    print(f"- invalid/missing tracking_duration: {ignored.get('invalid_or_missing_duration', 0)}")
    print(f"- total ignored variants: {ignored_total}\n")

    # Export tasks_info JSON for HDF5 script usage.
    tasks_info = {}
    for cat in categories:
        if not cat["tasks"]:
            continue
        task_id = category_task_ids[cat["label"]]
        unique_instructions = sorted({task for task, *_ in cat["tasks"]})
        variants = {}
        for idx, instruction_text in enumerate(unique_instructions, start=1):
            variants[f"V{task_id[1:]}.{idx}"] = {"task_instruction": instruction_text}
        tasks_info[task_id] = {
            "task_family": cat["label"],
            "variants": variants,
        }

    # Keep deterministic order of task and variant IDs in exported JSON.
    sorted_tasks_info = {}
    for task_id in sorted(tasks_info.keys(), key=lambda x: int(re.sub(r'\D', '', x) or 0)):
        task = tasks_info[task_id]
        variant_items = sorted(
            task["variants"].items(),
            key=lambda kv: [int(x) for x in re.findall(r'\d+', kv[0])]
        )
        sorted_tasks_info[task_id] = {
            "task_family": task["task_family"],
            "variants": {k: v for k, v in variant_items},
        }

    output_path = Path(args.tasks_info_output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_tasks_info, f, indent=4, ensure_ascii=False)
        f.write("\n")
    print(f"tasks_info JSON written to: {output_path}")


if __name__ == '__main__':
    main()

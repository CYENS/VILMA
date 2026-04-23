# -------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------

# Audio transcription
from openai import OpenAI
# Initialize the client
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ["openai_api_key"]
client = OpenAI(
    api_key=openai_api_key,
)

# Audio detection and video cropping 
import numpy as np
import librosa
from scipy.signal import correlate
import ffmpeg
import io
import subprocess
import csv
from pathlib import Path
SYNC_DURATION = 1.697959

# April tag detection and fingers distance visualization
import cv2
import matplotlib.pyplot as plt
from pupil_apriltags import Detector
import pandas as pd

# Load sessions
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="VILMA data processing pipeline")
    parser.add_argument("--json", required=True, help="Path to sessions JSON file")
    parser.add_argument("--recordings", required=True, help="Path to recordings folder")
    return parser.parse_args()

args = parse_args()
JSON_FILE = args.json

# Video directories
IN_DIR = os.path.abspath(args.recordings)
VIDEO_DIR = [
    os.path.abspath(os.path.join(IN_DIR, "head")),
    os.path.abspath(os.path.join(IN_DIR, "right")),
    os.path.abspath(os.path.join(IN_DIR, "left"))
]

def json_to_abs_path(path_str):
    if path_str is None:
        return None
    norm = str(path_str).replace("\\", "/")
    if norm.startswith("recordings/"):
        rel = norm[len("recordings/"):]
        return os.path.abspath(os.path.join(IN_DIR, rel))
    return os.path.abspath(path_str)

def abs_to_json_recordings_path(abs_path):
    rel = os.path.relpath(os.path.abspath(abs_path), IN_DIR)
    rel = rel.replace("/", "\\")
    return f"recordings\\{rel}"

with open(JSON_FILE, "r") as f:
    sessions = json.load(f)

# Optional pixel ROI for AprilTag detection per role: (x_min, y_min, x_max, y_max)
# height corresponds to y range, and width corresponds to x range
# Set to None to use full frame.
TAG_DETECTION_ROI = {
    "left": (1000, 1600, 2840, 2160),
    "right": (1000, 1600, 2840, 2160),
}

# -------------------------------------------------------------------
# Audio transcription - Functions
# -------------------------------------------------------------------
def transcribe_audio(wav_path: str) -> str:
    """
    Send a WAV file to OpenAI for transcription.
    """
    with open(wav_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language="en"
        )
    return response.text

# -------------------------------------------------------------------
# Audio detection and video duration and croping - Functions
# -------------------------------------------------------------------
def extract_audio_to_memory(video_path, sr=44100):
    out, _ = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='wav', ac=1, ar=sr)
        .run(capture_stdout=True, capture_stderr=True)
    )
    audio, _ = librosa.load(io.BytesIO(out), sr=sr, mono=True)
    audio = audio / np.max(np.abs(audio))
    return audio

def find_sync_offset(video_path, sync_wav_path, sr=44100):
    video_audio = extract_audio_to_memory(video_path, sr)
    sync_audio, _ = librosa.load(sync_wav_path, sr=sr, mono=True)
    sync_audio = sync_audio / np.max(np.abs(sync_audio))

    corr = correlate(video_audio, sync_audio, mode="valid")
    peak = np.argmax(corr)

    offset_sec = peak / sr
    confidence = corr[peak] / np.mean(corr)

    return offset_sec, confidence

def trim_and_optimize_video(input_path, output_path, start_time, duration):
    cmd = [
        "ffmpeg",
        "-y",                   # overwrite output
        "-i", input_path,
        "-ss", str(start_time), # where to start
    ]
    if duration is not None:
        cmd.append("-t")        # how long to keep
        cmd.append(str(duration)) 
    cmd.append("-c:v")
    cmd.append("libx264")       # video codec    
    cmd.append("-an")           # remove audio track
    cmd.append(output_path)     # output path
    subprocess.run(cmd, check=True)

def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


# ------------------------------------------------------------------- 
# Finger distance extraction - Functions 
# ------------------------------------------------------------------- 

def _clamp_roi_to_frame(roi, frame_shape):
    if roi is None:
        return None
    h, w = frame_shape[:2]
    x_min, y_min, x_max, y_max = roi
    x_min = max(0, min(int(x_min), w - 1))
    y_min = max(0, min(int(y_min), h - 1))
    x_max = max(1, min(int(x_max), w))
    y_max = max(1, min(int(y_max), h))
    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max, y_max)


def compute_finger_distance(video_path, filename, closed_dist, threads=12, coeff=[-0.00000035, 0.00967888, -0.1224], roi=None): 
    """ 
    Output: The distance between the two tags for each frame in pixels and cm. 
    """ 

    # Create detector 
    detector = Detector( 
        families="tag36h11", 
        nthreads=threads, 
        quad_decimate=1.0, 
        quad_sigma=0.0, 
        refine_edges=1, 
        decode_sharpening=0.25, 
        debug=0 
    ) 

    # Open video file 
    cap = cv2.VideoCapture(video_path) 
    if not cap.isOpened(): 
        print("Error opening video - skipping fingers distance for this video")
        return False

    # Video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = str(Path(video_path).with_suffix('')) + "_tags.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with open(filename, "w", newline="") as f: 
        writer = csv.writer(f) 

        header = ["frame_idx", "t_sec", "tags_dist_px", "tags_dist_cm", "fingers_dist_cm"] 
        writer.writerow(header) 

        frame_idx = 0 

        while True: 
            ret, frame = cap.read() 
            if not ret: 
                break # end of video 
            
            t_sec = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Run tag detection only inside ROI when provided.
            roi_clamped = _clamp_roi_to_frame(roi, frame.shape)
            if roi_clamped is not None:
                x_min, y_min, x_max, y_max = roi_clamped
                gray_for_detect = gray[y_min:y_max, x_min:x_max]
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    "ROI",
                    (x_min, max(20, y_min - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
            else:
                x_min, y_min = 0, 0
                gray_for_detect = gray
            
            # Detect tags 
            results = detector.detect(gray_for_detect)

            # Draw detected tag boxes and centers on full frame.
            for r in results:
                corners = r.corners
                center = r.center
                tag_id = r.tag_id

                corners_full = corners.copy()
                corners_full[:, 0] += x_min
                corners_full[:, 1] += y_min
                center_full = (int(center[0] + x_min), int(center[1] + y_min))

                for i in range(4):
                    pt1 = tuple(corners_full[i].astype(int))
                    pt2 = tuple(corners_full[(i + 1) % 4].astype(int))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                cv2.circle(frame, center_full, 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"id:{tag_id}",
                    (center_full[0] + 8, center_full[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

            # Compute distance in pixels and cm 
            if len(results) == 2: 
                x1, y1 = results[0].center
                x2, y2 = results[1].center
                # Convert ROI-local detections to original frame coordinates.
                x1 += x_min
                y1 += y_min
                x2 += x_min
                y2 += y_min
                tags_dist_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) 
                tags_dist_cm = tags_dist_px * coeff[0] * tags_dist_px + coeff[1] * tags_dist_px + coeff[2] 
                fingers_dist_cm = tags_dist_cm - closed_dist # subtract smallest distance (when fingers are closed)
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"px:{tags_dist_px:.1f} cm:{tags_dist_cm:.2f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
            else: 
                tags_dist_px = None 
                tags_dist_cm = None 
                fingers_dist_cm = None
            writer.writerow([frame_idx, t_sec, tags_dist_px, tags_dist_cm, fingers_dist_cm]) 
            out.write(frame)
            frame_idx += 1 

    cap.release()
    out.release()
    return True

def plot_fingers_distance(filename):
    df = pd.read_csv(filename)

    if "t_sec" in df:
        t = df["t_sec"]
        label = "Time (s)"
    else:
        t = df["frame_idx"]
        label = "Frame index"

    plt.figure()
    plt.plot(t, df["fingers_dist_cm"])

    plt.title(filename)
    plt.xlabel(label)
    plt.ylabel("Fingers Distance (cm)")
    plt.grid()

    plot_filename = f"{Path(filename).with_suffix('')}.png"
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename


# ------------------------------------------------------------------- 
# Plot tracking data - Functions 
# ------------------------------------------------------------------- 
def plot_tracking_data(filename): 
    df = pd.read_csv(filename)
    t = df["t_sec"]

    trackers = sorted({
        col.rsplit("_", 1)[0]
        for col in df.columns
        if col.endswith("_x")
    })

    plot_filenames = []

    for tracker in trackers:
        variables = ["x", "y", "z", "yaw", "pitch", "roll"]

        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

        for i, var in enumerate(variables):
            row = i % 3
            col = 0 if i < 3 else 1

            ax = axes[row][col]

            data = df[f"{tracker}_{var}"]

            # unwrap angles
            if var in ["yaw", "pitch", "roll"]:
                data = np.rad2deg(np.unwrap(np.deg2rad(data)))

            ax.plot(t, data)
            ax.set_title(f"{tracker}_{var}")
            ax.grid()

        axes[2][0].set_xlabel("Time (s)")
        axes[2][1].set_xlabel("Time (s)")

        fig.suptitle(f"{tracker} Tracking Data", fontsize=14)
        plt.tight_layout()

        plot_filename = f"{Path(filename).with_suffix('')}_{tracker}.png"
        plt.savefig(plot_filename)
        plot_filenames.append(plot_filename)
        plt.show()
        plt.close()

    return plot_filenames


# -------------------------------------------------------------------
# MAIN 
# -------------------------------------------------------------------
print("-----------------------")
print("VILMA - Data Processing")
print("-----------------------")

# Parse sessions
counter = 0
for session in sessions: 
    session_id = session["session_id"]
    print("\nProcessing session", session_id)

    # Transcribe audio instruction (speech-to-text)
    if "instruction" in session:
        if "audio_instruction_path" in session["instruction"]:
            if not "text_instruction" in session["instruction"]:
                audio_instruction_abs = json_to_abs_path(session["instruction"]["audio_instruction_path"])
                transcription = transcribe_audio(audio_instruction_abs)
                print("\tTranscription: ", transcription)
                sessions[counter]["instruction"]["text_instruction"] = transcription
                # Update sessions
                with open(JSON_FILE, "w") as f:
                    json.dump(sessions, f, indent=4)

    if "variants" in session:
        for variant in session["variants"]:
            variant_id = variant["variant_id"] #! supposing no variant is skipped, otherwise use a new counter
            print("\tProcessing variant", variant_id)
            update_variant = False
            if "tracking" in variant:
                if "tracking_path" in variant["tracking"]:
                    tracking_logs = variant["tracking"]["tracking_path"]
                    tracking_logs_abs = json_to_abs_path(tracking_logs)
                    # Create tracking plots if they are missing
                    if "tracking_plots" in variant["tracking"]:
                        if len(variant["tracking"]["tracking_plots"])==0:
                            print("\t\tCreating tracking plots...")
                            tracking_plot_filenames_abs = plot_tracking_data(tracking_logs_abs)
                            tracking_plot_filenames = [abs_to_json_recordings_path(p) for p in tracking_plot_filenames_abs]
                            sessions[counter]["variants"][variant_id-1]["tracking"]["tracking_plots"] = tracking_plot_filenames
                            update_variant = True
                    # Get first timestamp
                    if "first_tracking_ts" in variant["tracking"] and "tracking_duration" in variant["tracking"]:
                        first_ts = variant["tracking"]["first_tracking_ts"]
                        tracking_duration = variant["tracking"]["tracking_duration"]
                    else:
                        with open(tracking_logs_abs, newline="") as f:
                            reader = csv.reader(f)
                            header = next(reader, None)      # header (or None if file empty)
                            first_ts = None
                            last_ts = None
                            for row in reader:
                                if not row:  # skip empty rows
                                    continue
                                if first_ts is None:
                                    first_ts = float(row[0]) # first data row first column
                                last_ts = float(row[0]) # keeps updating until last row
                            if first_ts is None:
                                print("\t\tNo tracking data found.")
                                first_ts = 0
                                tracking_duration = 0
                            else:
                                print("\t\tFirst tracking timestamp: ", first_ts)
                                print("\t\tLast tracking timestamp: ", last_ts)
                                tracking_duration = last_ts - first_ts
                        sessions[counter]["variants"][variant_id-1]["tracking"]["first_tracking_ts"] = first_ts
                        sessions[counter]["variants"][variant_id-1]["tracking"]["tracking_duration"] = tracking_duration
                        update_variant = True

                if "videos" in variant:
                    videos = variant["videos"] 
                    if isinstance(videos, list):
                        for v in videos:
                            v_path = v["video_path"]
                            v_path_abs = json_to_abs_path(v_path)
                            v_path_without_extension = Path(v_path).with_suffix('')
                            video_id = v["video_id"] #! supposing no video is skipped, otherwise use a new counter
                            print("\t\tProcessing video", v_path)
                            
                            # Get duration of raw video
                            if "raw_video_duration" not in v:
                                raw_duration = get_video_duration(v_path_abs)
                                sessions[counter]["variants"][variant_id-1]["videos"][video_id-1]["raw_video_duration"] = raw_duration
                                update_variant = True
                                print("\t\t\tDuration of raw video:", raw_duration)
                            else:
                                raw_duration = v["raw_video_duration"]

                            # Audio detection 
                            if "offset" in v:
                                offset = v["offset"]
                            else:
                                offset, conf = find_sync_offset(v_path_abs, "sync.wav")
                                print(f"\t\t\t{v_path}: Sync at {offset} seconds (confidence={conf:.1f})")
                                sessions[counter]["variants"][variant_id-1]["videos"][video_id-1]["offset"] = offset
                                sessions[counter]["variants"][variant_id-1]["videos"][video_id-1]["confidence"] = float(conf)
                                update_variant = True
                            # Video cropping (for syncing with tracking data) and H264 optimization
                            if not "synced_video_path" in v:
                                start_time = offset + first_ts + SYNC_DURATION
                                if raw_duration < start_time + tracking_duration:
                                    # Crop only the beginning
                                    synced_v_path = str(v_path_without_extension) + "_synced.MP4"
                                    synced_v_path_abs = json_to_abs_path(synced_v_path)
                                    print(f"\t\t\toffset={offset}, tracking start={first_ts}, sync duration={SYNC_DURATION}")
                                    print(f"\t\t\tWarning: {start_time + tracking_duration}s exceeds video duration ({raw_duration}s)")
                                    print(f"\t\t\tTrim only {start_time} from beginning")
                                    trim_and_optimize_video(
                                        input_path=v_path_abs,
                                        output_path=synced_v_path_abs,
                                        start_time=start_time,
                                        duration=None
                                    )
                                    sessions[counter]["variants"][variant_id-1]["videos"][video_id-1]["comment"] = "Video cropped only from beginning"
                                elif tracking_duration == 0:
                                    print(f"\t\t\tWarning: tracking duration is 0, skipping video cropping")
                                    synced_v_path = v_path
                                    synced_v_path_abs = v_path_abs
                                    sessions[counter]["variants"][variant_id-1]["videos"][video_id-1]["comment"] = "Video not cropped because tracking duration is 0"
                                else:
                                    synced_v_path = str(v_path_without_extension) + "_synced.MP4"
                                    synced_v_path_abs = json_to_abs_path(synced_v_path)
                                    print(f"\t\t\toffset={offset}, tracking start={first_ts}, sync duration={SYNC_DURATION}")
                                    print(f"\t\t\tTrim {start_time} from beginning and keep {tracking_duration} seconds")
                                    trim_and_optimize_video(
                                        input_path=v_path_abs,
                                        output_path=synced_v_path_abs,
                                        start_time=start_time,
                                        duration=tracking_duration
                                    )
                                # Save in json
                                sessions[counter]["variants"][variant_id-1]["videos"][video_id-1]["synced_video_path"] = synced_v_path
                                update_variant = True
                            else:
                                synced_v_path = v["synced_video_path"]
                                synced_v_path_abs = json_to_abs_path(synced_v_path)
                            # Get duration of cropped video
                            if "synced_video_duration" not in v:
                                duration = get_video_duration(synced_v_path_abs)
                                sessions[counter]["variants"][variant_id-1]["videos"][video_id-1]["synced_video_duration"] = duration
                                update_variant = True
                                print("\t\t\tDuration of cropped video:", duration)

                            v_path_without_extension = Path(synced_v_path).with_suffix('')
                            if v["role"] != "head":
                                can_plot_fingers = False
                                # Get fingers distance
                                if "fingers_dist_path" not in v:
                                    print("\t\t\tDetecting april tags and computing fingers distance from video...")
                                    fingers_path = str(v_path_without_extension) + "_fingers_dist.csv"
                                    fingers_path_abs = json_to_abs_path(fingers_path)
                                    roi = TAG_DETECTION_ROI.get(v["role"])
                                    success = compute_finger_distance(
                                        synced_v_path_abs,
                                        fingers_path_abs,
                                        v["closed_fingers_tag_distance_cm"],
                                        roi=roi,
                                    )
                                    if success:
                                        sessions[counter]["variants"][variant_id-1]["videos"][video_id-1]["fingers_dist_path"] = fingers_path
                                        update_variant = True
                                        can_plot_fingers = True
                                        print("\t\t\tResults written in ", fingers_path)
                                    else:
                                        print("\t\t\tSkipping this video due to unreadable stream.")
                                else:
                                    fingers_path = v["fingers_dist_path"]
                                    fingers_path_abs = json_to_abs_path(fingers_path)
                                    can_plot_fingers = True
                                # Visualize fingers distance - save plot
                                if can_plot_fingers and "fingers_dist_plot" not in v:
                                    print("\t\t\tCreating plot for fingers distance")
                                    fingers_plot_abs = plot_fingers_distance(fingers_path_abs)
                                    fingers_plot = abs_to_json_recordings_path(fingers_plot_abs)
                                    sessions[counter]["variants"][variant_id-1]["videos"][video_id-1]["fingers_dist_plot"] = fingers_plot
                                    update_variant = True
                                    print("\t\t\tPlot saved in ", fingers_plot)
                    else:
                        print("\t\tVideo paths in wrong format")  
                else:
                    print("\t\tNo video paths for this variant")
            else:
                print("\t\tNo tracking log found for this variant")
            
            # Update sessions
            if update_variant:
                with open(JSON_FILE, "w") as f:
                    json.dump(sessions, f, indent=4)
    else:
        print("\tNo variants found for this session")
    counter += 1

"""
This program collects data from microphone, cameras, and trackers for unimanual and bimanual tasks. 
Sync sound is played to indicate the start of each recording.
"""


# -------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------
# Tracking
from triad_openvr import triad_openvr
import time
import sys
import csv
import threading
import simpleaudio as sa # for sync sound
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# GoPro
from open_gopro import WirelessGoPro
import asyncio
import os
import subprocess

# Audio
import soundfile as sf
import numpy as np
from datetime import datetime
import json

os.environ.setdefault("LANG", "en_US.UTF-8")

EXPECTED_TRACKER_IDS = {
    "LHR-38003502": "right",
    "LHR-6B90A355": "left",
}

EXPECTED_CAMERA_IDS = {
    "GP50499501": "right",
    "GP50513751": "left",
}

CAMERA_TARGETS = {
    "right": "9501",
    "left": "3751",
}

CLOSED_FINGERS_TAG_DISTANCE = {
    "s": 4.25, # soft fingers
    "h": 5.15, # hard fingers
}

# Output Directory
OUT_DIR = "./recordings"
OUT_DIR_INSTR = os.path.join(OUT_DIR, "instructions")
OUT_DIR_TRACK = os.path.join(OUT_DIR, "trackers")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_INSTR, exist_ok=True)
os.makedirs(OUT_DIR_TRACK, exist_ok=True)

# Load existing sessions if present
JSON_FILE = "./sessions.json"
if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r") as f:
        sessions = json.load(f)
else:
    sessions = []


# -------------------------------------------------------------------
# Tracking Functions
# -------------------------------------------------------------------

def discover_and_validate_trackers(expected_trackers):
    # Discover connected objects
    print("\nDiscovered tracking devices:")
    v = triad_openvr.triad_openvr()
    v.print_discovered_objects()

    trackers_roles = []
    trackers_names = []
    for name, device in v.devices.items():
        if name.startswith("tracker_"):
            serial = device.get_serial()
            if serial not in EXPECTED_TRACKER_IDS:
                print(f"Unknown tracker: {serial}")
            role = EXPECTED_TRACKER_IDS[serial]
            trackers_names.append(name)
            trackers_roles.append(role)
            print(f"Accepted {name} → serial='{serial}' → role='{role}'")

    # Terminate if not all expected trackers are connected
    if set(trackers_roles) == set(expected_trackers):
        return v, trackers_names, trackers_roles
    else:
        print("Found trackers: ", trackers_roles, "- Expected trackers:", expected_trackers)
        print("Program terminated")
        sys.exit(1)

def play_sync_wav(path="data_collection/sync.wav"):
    wave_obj = sa.WaveObject.from_wave_file(path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

class TrackerLogger:
    def __init__(self, v, trackers_names, trackers_roles, rate_hz=250):
        self.v = v
        self.trackers = trackers_names
        self.trackers_roles = trackers_roles
        self.interval = 1 / rate_hz
        self.running = False
        self.thread = None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(OUT_DIR, "trackers", f"trackers_{ts}.csv")

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            self.thread = None

    def _loop(self):
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)

            header = ["t_sec"]
            for t in self.trackers_roles:
                header += [f"{t}_x", f"{t}_y", f"{t}_z",
                           f"{t}_yaw", f"{t}_pitch", f"{t}_roll"]
            writer.writerow(header)
                        
            play_sync_wav() # start recording
            t0 = time.time()

            while self.running:                
                start = time.time()
                row = [time.time() - t0]

                for t in self.trackers:
                    pose = self.v.devices[t].get_pose_euler()
                    row += list(pose) if pose else [None] * 6

                writer.writerow(row)
                #print(row, end="\r", flush=True)

                sleep_time = self.interval - (time.time() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

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
# Audio Functions
# -------------------------------------------------------------------

def get_wireless_go_device():
    # import sounddevice here so it doesn't affect cameras connection (important on Windows) 
    import sounddevice as sd
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        name = device["name"]
        hostapi = sd.query_hostapis()[device["hostapi"]]["name"]

        if "Wireless GO RX" in name and "WASAPI" in hostapi:
            print(f"Using device {name}")
            return i

    print("Wireless GO RX (WASAPI) device not found.")
    print("Program terminated")
    sys.exit(1)

def record_audio(sample_rate=48000, channels=2):
    # import sounddevice here so it doesn't affect cameras connection (important on Windows) 
    import sounddevice as sd
    device_index = get_wireless_go_device()

    input("\nPress ENTER to record audio instruction...")
    print("Recording audio... Press ENTER to stop.")

    frames = []

    def callback(indata, frames_count, time, status):
        frames.append(indata.copy())

    with sd.InputStream(
        device=device_index,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        callback=callback
    ):
        input()

    audio = np.concatenate(frames, axis=0)
    return audio

def save_wav(audio, sample_rate=48000):
    # saving audio timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rec_{ts}.wav"
    path = os.path.join(OUT_DIR, "instructions", filename)
    sf.write(path, audio, sample_rate)
    return path

# -------------------------------------------------------------------
# GoPro Functions
# -------------------------------------------------------------------
async def identify_camera_ssid(gopro, retries=3):
    for attempt in range(retries):
        try:
            # Find the StatusId enum type dynamically
            status = await gopro.ble_command.get_camera_statuses()
            data = status.data
            StatusId = type(next(iter(data.keys()))) 
            ssid = data[StatusId.ACCESS_POINT_SSID]
            return ssid
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1.0)
            
async def connect_and_validate(connected_ssids, connected_roles, target):
    print("Connecting to camera...")
    gopro = WirelessGoPro(ble_device=target, debug=True)
    await gopro.__aenter__()

    print("Getting SSID...")
    try:
        await asyncio.sleep(1.0)
        ssid = await identify_camera_ssid(gopro)
    except Exception:
        await gopro.__aexit__(None, None, None)
        raise

    # Reject unknown camera
    if ssid not in EXPECTED_CAMERA_IDS:
        await gopro.__aexit__(None, None, None)
        raise RuntimeError(f"Unknown camera SSID: {ssid}")

    role = EXPECTED_CAMERA_IDS[ssid]

    # Reject already-connected camera
    if ssid in connected_ssids:
        await gopro.__aexit__(None, None, None)
        raise RuntimeError(f"Camera already connected: {ssid}")

    # Reject role collision
    if role in connected_roles:
        await gopro.__aexit__(None, None, None)
        raise RuntimeError(f"Role already occupied: {role}")

    print(f"Accepted camera → SSID='{ssid}' → role='{role}'")
    return role, ssid, gopro

async def connect_all_cameras(expected_cameras):

    cameras = {}           # role -> gopro
    connected_ssids = set()
    connected_roles = set()

    print("\nWaiting for expected GoPro camera(s)...\n")

    for role in expected_cameras:
        target = CAMERA_TARGETS[role]
        try:
            role, ssid, gopro = await connect_and_validate(
                connected_ssids,
                connected_roles,
                target
            )
        except RuntimeError as e:
            print(f"Rejected connection for {role}: {e}")
            continue

        cameras[role] = gopro
        connected_ssids.add(ssid)
        connected_roles.add(role)

    print("\nAll expected cameras connected and locked.")
    return cameras

async def start_recording(cameras):
    await asyncio.gather(
        *[cam.ble_command.set_shutter(shutter=True) for cam in cameras.values()]
    )

async def stop_recording(cameras):
    await asyncio.gather(
        *[cam.ble_command.set_shutter(shutter=False) for cam in cameras.values()]
    )

def get_current_ssid():
    result = subprocess.run(
        "netsh wlan show interfaces",
        shell=True,
        capture_output=True,
        text=True
    )
    output = result.stdout.splitlines()
    for line in output:
        line = line.strip()
        # Important: avoid matching "BSSID"
        if line.startswith("SSID") and "BSSID" not in line:
            ssid = line.split(":", 1)[1].strip()
            print("Connected to WiFi: ", ssid)
            return ssid

def validate_and_connect_wifi(expected_ssid):
    # Find current wifi
    connected_ssid = get_current_ssid()

    # Connect to the expected wifi
    while connected_ssid != expected_ssid:
        print(f"Switching to WiFi: {expected_ssid}")
        # Disconnect first (important!)
        subprocess.run("netsh wlan disconnect", shell=True)
        time.sleep(2)
        # Connect to target SSID
        result = subprocess.run(
            f'netsh wlan connect name="{expected_ssid}"',
            shell=True,
            capture_output=True,
            text=True
        )
        # Wait for connection to stabilize
        time.sleep(5)
        # Find current wifi
        connected_ssid = get_current_ssid()

async def get_media_snapshot(ssid, gopro):
    validate_and_connect_wifi(ssid)
    resp = await gopro.http_command.get_media_list()
    return {
        f.filename
        for folder in resp.data.media
        for f in folder.file_system
    }

def diff_media(old, new):
    # return files that are in new but not in old
    return sorted(new - old)

async def download_files(gopro, files, out_dir):
    print("Downloading...")
    downloaded = []

    for camera_path in files:
        await gopro.http_command.download_file(
            camera_file=camera_path,
            local_file=os.path.join(out_dir, os.path.basename(camera_path)),
        )
        downloaded.append(camera_path)

    return downloaded


# -------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------
async def main():
    print("-----------------------")
    print("VILMA - Data Collection")
    print("-----------------------")

    # Expected grippers (cameras/trackers)
    num_expected_grippers = int(input("\nHow many grippers(cameras and trackers) are you connecting? [1/2]: ").strip())
    if num_expected_grippers == 1:
        unimanual = input("\nWhich gripper are you using? [r/l]: ").strip().lower()
        if unimanual == "r":
            expected_grippers = ["right"]
        elif unimanual == "l":
            expected_grippers = ["left"]
        else:
            print("Invalid input. Enter 'r' or 'l'.")
            sys.exit(1)
    elif num_expected_grippers == 2:
        expected_grippers = ["right", "left"]
    else:
        print("Invalid input. Enter '1' or '2'.")
        sys.exit(1)

    # Head-mounted camera
    head_camera = input("\nAre you using a head-mounted camera? [y/n]: ").strip().lower()
    if head_camera == "y":
        head_camera = True
    else:
        head_camera = False

    # Connect trackers
    v, trackers_names, trackers_roles = discover_and_validate_trackers(
        expected_trackers=expected_grippers
    )

    # Connect cameras
    cameras = await connect_all_cameras(
        expected_cameras=expected_grippers
    )

    # Media baselines per camera role
    media_state = {}
    for role, cam in cameras.items():
        print(f"\nFetching initial media list for {role} camera...")
        ssid = [cam_id for cam_id, cam_role in EXPECTED_CAMERA_IDS.items() if cam_role == role][0]
        media_state[role] = await get_media_snapshot(ssid, cam)
        # print(sorted(list(media_state[role])))
        print(f"Last video: {sorted(list(media_state[role]))[-1]}")

    # Ready when all expected trackers and cameras are connected
    print("Rig ready.")

    location = input("\nEnter location: ").strip()

    try:
        # AUDIO LOOP - per task
        while True:
            print("\nTASK ", len(sessions)+1)

            # Record and save the audio instruction
            audio = record_audio()        
            wav_path = save_wav(audio)
            last_id = sessions[-1]["session_id"] if len(sessions)>0 else 0
            session_id = last_id + 1
            created_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sessions.append({
                "session_id": session_id,
                "timestamp": created_at,
                "instruction": {
                    "audio_instruction_path": wav_path
                },
                "variants": []
            })

            # TRACKERS/CAMERAS LOOP - per variant
            count_rec = 1
            while True:
                print("\nVARIANT ", count_rec)

                # User ID
                userID = input("\nEnter user ID: ").strip()
                # Number of grippers
                fingers_type = {}
                if num_expected_grippers == 2:
                    # Fingers type (soft/hard)
                    bimanual = input("\nAre you using 2 grippers? [y/n]: ").strip().lower()
                    # if bimanual == "y":
                    #     fingers_type["right"] = input("\nFor the right hand, are you using soft or hard fingers? [s/h]: ").strip().lower()
                    #     fingers_type["left"] = input("\nFor the left hand, are you using soft or hard fingers? [s/h]: ").strip().lower()
                    if bimanual == "n":
                        unimanual = input("\nWhich gripper are you using? [r/l]: ").strip().lower()
                        if unimanual == "r":
                            unimanual = "right"
                            # fingers_type["right"] = input("\nAre you using soft or hard fingers? [s/h]: ").strip().lower()
                        else:
                            unimanual = "left"
                #             fingers_type["left"] = input("\nAre you using soft or hard fingers? [s/h]: ").strip().lower()
                # elif num_expected_grippers == 1:
                #     fingers_type[unimanual] = input("\nAre you using soft or hard fingers? [s/h]: ").strip().lower()

                print("\nClosed soft fingers tag distance (cm):", CLOSED_FINGERS_TAG_DISTANCE["s"])
                print("Closed hard fingers tag distance (cm):", CLOSED_FINGERS_TAG_DISTANCE["h"])

                # Used cameras
                used_cameras = {}
                for role, camera in cameras.items():
                    if num_expected_grippers == 2 and bimanual == "n":
                        if role != unimanual:
                            continue
                    used_cameras[role] = camera
                    
                # Used trackers - Construct TrackerLogger 
                used_trackers_names = []
                used_trackers_roles = []
                for i, role in enumerate(trackers_roles):
                    if num_expected_grippers == 2 and bimanual == "n":
                        if role != unimanual:
                            continue
                    used_trackers_names.append(trackers_names[i])
                    used_trackers_roles.append(trackers_roles[i])
                tracker = TrackerLogger(v, used_trackers_names, used_trackers_roles) # search for original names

                if head_camera:
                    # Wait for user to start manually the head mounted camera
                    input("\nStart head mounted camera MANUALLY, and press ENTER to continue...")

                # Wait for user input to start
                input("\nPress ENTER to start recording with gripper(s)...")
                # Start video recording and tracking
                await start_recording(used_cameras)
                tracker.start()
                print("Recording started.")

                # Wait for user input to stop
                input("\nPress ENTER to stop.")
                # Stop tracking and video recording
                tracker.stop()
                await stop_recording(used_cameras)
                print("Recording stopped.")
                await asyncio.sleep(1)

                # # Download new videos
                # variant_files = {}
                # for role, cam in used_cameras.items():
                #     print(f"\nChecking new media on {role} camera")
                #     ssid = [cam_id for cam_id, cam_role in EXPECTED_CAMERA_IDS.items() if cam_role == role][0]
                #     new_snapshot = await get_media_snapshot(ssid, cam)
                #     new_files = diff_media(media_state[role], new_snapshot)
                #     if not new_files:
                #         print(f"No new files found on {role} camera")
                #         continue
                #     print("New files: ", new_files)
                #     role_dir = os.path.join(OUT_DIR, role)
                #     os.makedirs(role_dir, exist_ok=True)
                #     downloaded = await download_files(cam, new_files, role_dir)
                #     print(f"Downloaded new files found on {role} camera")
                #     variant_files[role] = downloaded
                #     # Update baseline
                #     media_state[role] = new_snapshot

                # # Build structured video list
                # videos_structured = []
                # video_id = 1
                # for role, files in variant_files.items():
                #     if num_expected_grippers == 2 and bimanual == "n":
                #         if role != unimanual:
                #             continue
                #     for camera_path in files:
                #         filename = os.path.basename(camera_path)
                #         normalized_path = os.path.join(OUT_DIR, role, filename)
                #         videos_structured.append({
                #             "video_id": video_id,
                #             "video_path": normalized_path,
                #             "role": role,
                #             "closed_fingers_tag_distance_cm": CLOSED_FINGERS_TAG_DISTANCE[fingers_type[(role)]]
                #         })
                #         video_id += 1

                # Update sessions
                cameras_used = list(used_cameras.keys())
                if head_camera:
                    cameras_used.append("head")
                sessions[-1]["variants"].append({
                    "variant_id": count_rec,
                    "user_id": userID,
                    "location": location,
                    "trackers_used": used_trackers_roles,
                    "cameras_used": cameras_used,
                    "tracking": {
                        "tracking_path": tracker.filename
                    },
                    #"videos": videos_structured,
                })
                with open(JSON_FILE, "w") as f:
                    json.dump(sessions, f, indent=4)
                
                if head_camera:
                    # Wait for user input to stop manually the head mounted camera
                    input("\nStop head mounted camera MANUALLY, and press ENTER to continue...")

                # Plot tracking data
                tracking_plot_filenames = plot_tracking_data(tracker.filename)
                sessions[-1]["variants"][-1]["tracking"]["tracking_plots"] = tracking_plot_filenames
                # Mark tracking outliers
                mark_outliers = input("\nMark outliers? [y/n]: ").strip().lower()
                if mark_outliers == "y":
                    sessions[-1]["variants"][-1]["tracking"]["tracking_outliers"] = True
                else:
                    sessions[-1]["variants"][-1]["tracking"]["tracking_outliers"] = False
                
                # Mark task failure
                mark_failure = input("\nMark task failure? [y/n]: ").strip().lower()
                if mark_failure == "y":
                    sessions[-1]["variants"][-1]["task_failure"] = True
                else:
                    sessions[-1]["variants"][-1]["task_failure"] = False
                
                # Update sessions
                with open(JSON_FILE, "w") as f:
                    json.dump(sessions, f, indent=4)
                print(f"\nSession data updated in {JSON_FILE}")

                # Next variant (repetition of the same task)
                again = input("\nNext variant? [y/n]: ").strip().lower()
                count_rec += 1
                if again == "n":
                    break

            # Next task
            again = input("\nNext task? [y/n]: ").strip().lower()
            if again == "n":
                break

    # Always disconnect cleanly
    finally: 
        await asyncio.gather(
            *[cam.__aexit__(None, None, None) for cam in cameras.values()],
            return_exceptions=True,
        )
        print("\nAll cameras disconnected")


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())


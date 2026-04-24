# VILMA

## 1. Installation

### 1.1. Clone the repository

```bash
git clone https://github.com/ctheoc/VILMA.git
cd VILMA
```

### 1.2. Create a virtual environment

```bash
python -m venv venv
```

### 1.3. Activate the virtual environment

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 1.4. Install dependencies

#### 1.4.1 ffmpeg

To check if it's installed, run: 
```bash
which ffprobe
ffprobe -version 
```

If not installed, download from https://ffmpeg.org, or run

**Ubuntu/Debian**
```bash
sudo apt updatesudo apt install ffmpeg 
```

**macOS (Homebrew)**: 
```bash
brew install ffmpeg 
```

#### 1.4.2 Python libraries

```bash
pip install -r requirements.txt
```

## 2. Data Collection

Data are collected using VIVE trackers, GoPro cameras, and microphone.

### 2.1. Finger distance calibration

To account for the severe distortion of the GoPro Hero 13 Black’s 'Ultra Wide HyperView' lens, perform a custom finger distance calibration. Map physical measurements to the pixel distances between the centers of the AprilTags on each finger. 

```bash
python data_collection/vilma_fingers_distance_calibration.py
```

### 2.2. Data Collection setup

#### 2.2.1 Tracking
VIVE trackers and base stations should be connected via SteamVR.
Prerequisites: Download and install SteamVR.
Connect base stations and trackers:
1. Connect the base stations to power.
2. Connect the trackers to the PC:
   - Connect one end of the USB Type-C cable to the dongle cradle, and then plug the dongle into the cradle.
   - Connect the other end of the USB Type-C cable to a USB port on your computer.
   - Note: Keep the dongle at least 45 cm away from the computer and place it where it won’t be moved.
   - From your computer, open the SteamVR app.
   - Click **Menu > Devices > Pair Controller**.
   - Press the Power button for around 2 seconds. The status light will blink blue.
   - Wait for the status light to turn green. This means pairing is successful.
   - In the Controller Pairing window, click **Done**.

#### 2.2.2 Video recording
We use two GoPro HERO13 cameras, each one mounted on each handheld gripper, and a head-mounted GoPro HERO9 camera. HERO13 cameras can be manipulated via this script, while HERO9 is manipulated manually, or via a mobile phone.
Prerequisites: Download GoPro Quik app on the mobile phone.
Setup and connect cameras:
1. Camera settings:
   - GoPro HERO13: 16:9 | 4K | 60 | UHV
   - GoPro HERO9: 4K | 60 | L
2. Connect GoPro HERO13 to the PC:
   - Turn on the cameras and reset wireless connections: **Preferences > Wireless Connections > Reset Connections > Reset All**.
   - Turn on the PC's Bluetooth and Wi-Fi, and delete GoPro cameras from the computers previous connected Bluetooth devices, if any.
   - Select **Pair Device** on cameras.
      
3. Connect GoPro HERO9 to a mobile phone:
   - Turn on the phone's Bluetooth and Wi-Fi, open GoPro Quik, and select GoPro tab.
   - Turn on the camera and select **Connections > Connect Device > Quick App** to connect.

#### 2.2.3 Audio recording
Set up RODE microphone:
1. Connect RX (Receiver) to the PC via USB.
2. Connect TX (Transmitter) to the RX.
3. Select **Wireless GO GX** as the input device in the PC settings.

#### 2.2.4 Check PC audio
Ensure the PC volume is **not muted** so you can hear the 'beep' sound when a recording starts. This sound will be later used to synchronize tracking and video recordings.

### 2.3. Run the data collection script

```bash
python data_collection/vilma_collect_data.py
```

This script creates in the repository root: 
 - a folder with the recorded data named **recordings**, and 
 - a JSON file named **sessions.json** that includes the paths to the data files.

For tracking we are using:
https://github.com/TriadSemi/triad_openvr
commit: d389aacf2a4caa392398613a9daddba15ee24f92

## 3. Data Processing

### 3.1. Data Processing setup

Make sure that videos exists in the **recordings** folder.
 - Videos captured by the camera mounted on the left gripper under **recordings/left**
 - Videos captured by the camera mounted on the right gripper under **recordings/right**
 - Videos captured by the head-mounted camera under **recordings/head**

To associate these videos with the rest of the recorded data, run:

```bash
python data_processing/vilma_associate_videos.py --json sessions.json --left recordings/left --right recordings/right --head recordings/head
```

Also, an OpeanAI API key is required. 
Create a `.env` file in the repository root with your OpenAI API key:

```bash
echo 'openai_api_key=YOUR_OPENAI_API_KEY' > .env
```

### 3.2. Run the data processing script

The script prepares data before saving into the final dataset:
- Transcribes the recorded instruction (speech-to-text) using OpenAI
- Synchronizes tracking and videos by trimming the videos
- Optimizes videos (H264 codec, and removes audio)
- Computes the distance between the fingers of each gripper by detecting the AprilTags

```bash
python data_processing/vilma_process_data.py --json sessions.json --recordings recordings
```

### 3.3. Extract depth maps

To extract depth maps from the videos using Depth-Anything-V2, run:

```bash
python data_processing/Depth-Anything-V2/run_video.py --encoder vits --sessions-path sessions.json --pred-only --grayscale
```

For depth depth extraction we are using:
https://github.com/DepthAnything/Depth-Anything-V2
commit: a561b849ebae10a6f5ef49e26c83cbbcd36c71bf

### 3.4. Video compression

To convert videos (including depth maps) to 720p at 30FPS, run:

```bash
python data_processing/vilma_compress_videos.py --input-dir recordings
```

## 4. Dataset creation

### 4.1. Calculate statistics

Generate task, participants, and locations statistics, and export `tasks_info` from the json file:

```bash
python dataset_creation/vilma_calculate_statistics.py sessions.json --sort duration --tasks-info-output vilma_tasks_info.json
```

### 4.2 Create the dataset

Create/append the HDF5 structure from the json file:

```bash
python dataset_creation/vilma_create_hdf5_dataset.py --json sessions.json --tasks-info vilma_tasks_info.json --recordings-root recordings --output vilma_dataset.h5
```

Organize video files according to the HDF5 hierarchy:

```bash
python dataset_creation/vilma_organize_videos_by_hdf5.py --h5 vilma_dataset.h5 --recordings-root recordings --output-root /path/to/dataset_root
```

The VILMA Dataset can be downloaded from our EuroCore Zenodo repository here: https://doi.org/10.5281/zenodo.19708163

### 4.3 Navigate the dataset

Print HDF5 contents (for array datasets, prints only shape and first element or line):

```bash
python dataset_creation/vilma_print_hdf5_contents.py --h5 vilma_dataset.h5
```


## 5. Hardware

The repository includes:

- Hardware guide
- Full CAD models (STEP, STL, and editable source references)
- Bill of Materials (BOM)
- 3D printing guidelines 
- Assembly instructions with deployment setup guidance
- Interchangeable hard and soft finger configurations
- Integrated HTC VIVE tracker mounting for motion tracking
- GoPro mounting support for egocentric recording
- AprilTags

The design preserves the original UMI pistol-style interaction while introducing project-specific modifications for improved tracking integration, deployment readiness, and real-world data collection reliability.

📁 Full hardware documentation is available in the `hardware/` folder.



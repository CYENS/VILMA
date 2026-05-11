import cv2
import os
import glob
import shutil
import subprocess
import traceback
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"insightface\.utils\.face_align"
)

from insightface.app import FaceAnalysis
from tqdm import tqdm

def get_dynamic_blur(width, height):
    """Calculates an appropriate odd-numbered blur kernel based on face size."""
    # Scale the blur kernel to be roughly 40% of the face width
    k_size = int(width * 0.4)
    # Ensure the kernel size is always an odd number (required by OpenCV)
    if k_size % 2 == 0:
        k_size += 1
    # Set a minimum blur size so tiny faces still get blurred
    k_size = max(15, k_size)
    return (k_size, k_size)

def batch_process_production(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize InsightFace
    print("Initializing SCRFD Model")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    video_files = []
    for root, _, files in os.walk(input_folder):
        for name in files:
            if name.lower().endswith('.mp4'):
                video_files.append(os.path.join(root, name))

    if not video_files:
        print("No .mp4 files found in the input directory.")
        return

    print(f"Found {len(video_files)} videos. Starting batch process...\n")

    for idx, video_file in enumerate(video_files):
        rel_path = os.path.relpath(video_file, input_folder)
        filename = os.path.basename(video_file)
        output_path = os.path.join(output_folder, rel_path)

        if os.path.exists(output_path):
            print(f"Skipping already processed: {rel_path}")
            continue

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        if 'depth' in filename.lower():
            shutil.copy2(video_file, output_path)
            print(f"Copied depth video: {rel_path}")
            continue

        faces_found = False
        try:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"\n[WARNING] Could not open {rel_path}. Skipping...")
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            temp_output_path = output_path + ".tmp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

            desc = f"Video {idx+1}/{len(video_files)}: {filename[:15]}..."
            with tqdm(total=total_frames, desc=desc, unit="frame") as pbar:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    faces = app.get(frame)
                    for face in faces:
                        box = face.bbox.astype(int)
                        conf = face.det_score

                        if conf > 0.60:
                            x1, y1, x2, y2 = box
                            w = x2 - x1
                            h = y2 - y1

                            margin_x = int(w * 0.15)
                            margin_y = int(h * 0.15)
                            x1 = max(0, x1 - margin_x)
                            y1 = max(0, y1 - margin_y)
                            x2 = min(width, x2 + margin_x)
                            y2 = min(height, y2 + margin_y)

                            roi = frame[y1:y2, x1:x2]
                            if roi.size > 0:
                                blur_kernel = get_dynamic_blur(w, h)
                                blurred_roi = cv2.GaussianBlur(roi, blur_kernel, 0)
                                frame[y1:y2, x1:x2] = blurred_roi
                                faces_found = True

                    out.write(frame)
                    pbar.update(1)

            cap.release()
            out.release()

            if shutil.which("ffmpeg") is not None:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    temp_output_path,
                    "-c:v",
                    "libx264",
                    "-crf",
                    "20",
                    "-preset",
                    "fast",
                    "-c:a",
                    "copy",
                    output_path,
                ]
                try:
                    subprocess.run(cmd, check=True)
                    os.remove(temp_output_path)
                except subprocess.CalledProcessError:
                    print(f"ffmpeg conversion to H264 failed for {rel_path}. Saving raw mp4 output.")
                    os.replace(temp_output_path, output_path)
            else:
                print("ffmpeg not found; saving raw mp4 output without explicit H264 encoding.")
                os.replace(temp_output_path, output_path)

            if faces_found:
                print(f"Faces detected in: {rel_path}")
            else:
                print(f"No faces detected in: {rel_path}")

        except Exception:
            print(f"\n[ERROR] Video {rel_path} failed. Skipping to next file.")
            print(traceback.format_exc())
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()

if __name__ == "__main__":
    # --- CHANGE THESE PATHS ---
    INPUT_DIR = "./data_with_faces/data"   
    OUTPUT_DIR = "./data"    
    batch_process_production(INPUT_DIR, OUTPUT_DIR)
    print("\nBatch processing complete! Check the terminal for any skipped files.")
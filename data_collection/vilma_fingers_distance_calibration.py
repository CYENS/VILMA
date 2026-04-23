import cv2
import numpy as np
import matplotlib.pyplot as plt
from pupil_apriltags import Detector

# --- CONFIGURATION ---
# Output filename for your coefficients
CALIBRATION_FILE = "gripper_coefficients.npy"
 
# --- DETECTION LOGIC ---
def get_pixel_distance(video_path, max_frames=2):
    """
    Input:  The video.
    Output: The distance between the two tags in PIXELS (float).
            Return None if tags are not detected.
    """
    # ---- Create detector ----
    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    # ---- Open video file ----
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video")
        exit()

    distances = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Detect tags ----
        results = detector.detect(gray)

        if len(results) == 2:
            x1, y1 = results[0].center
            x2, y2 = results[1].center
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            print(dist)
            distances.append(dist)
        
        if len(distances) >= max_frames:
            break

    if len(distances) == 0:
        print("No tags detected")
        return None

    avg_distance = np.mean(distances)
    print("Average distance in pixels: ", avg_distance)

    return avg_distance
 
# --- MAIN CALIBRATION LOOP ---
def run_interactive_calibration(data_cm, video_paths):
    data_pixels = []
    not_detected = []
    for i, video_path in enumerate(video_paths):
        print(f"Processing {video_path}")
        current_px = get_pixel_distance(video_path)
        data_pixels.append(current_px)

        if current_px is None:
            not_detected.append(i)
    
    # remove not detected from data_cm and data_pixels
    data_cm = [data_cm[i] for i in range(len(data_cm)) if i not in not_detected]
    data_pixels = [data_pixels[i] for i in range(len(data_pixels)) if i not in not_detected]
    print("Pixels: ", data_pixels)
    print("Real cm: ", data_cm)

    print("\nCalculating Polynomial Fit...")
    # 1. Sort data (helpful for plotting)
    sorted_indices = np.argsort(data_pixels)
    x_data = np.array(data_pixels)[sorted_indices]
    y_data = np.array(data_cm)[sorted_indices]
    # 2. Polynomial Fit (Degree 2)
    # y = Ax^2 + Bx + C
    coeffs = np.polyfit(x_data, y_data, 2)
    A, B, C = coeffs
    print(f"\n--- SUCCESS! ---")
    print(f"Formula: Real_CM = ({A:.8f} * px^2) + ({B:.8f} * px) + {C:.4f}")
    # 3. Save to file
    np.save(CALIBRATION_FILE, coeffs)
    print(f"Saved coefficients to '{CALIBRATION_FILE}'")
    # 4. Visual Check (Graph)
    # This generates the curve so you can see if it fits well
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='red', label='Measured Points')
    # Generate smooth line for the plot
    x_line = np.linspace(min(x_data), max(x_data), 100)
    y_line = A * x_line**2 + B * x_line + C
    plt.plot(x_line, y_line, label=f'Fit: {A:.2e}x² + {B:.2f}x + {C:.2f}')
    plt.title("Calibration Curve: Pixels vs CM")
    plt.xlabel("Camera Pixels")
    plt.ylabel("Real Centimeters")
    plt.legend()
    plt.grid(True)
    plt.show()
 
if __name__ == "__main__":

    real_values = [
        # Add real values here    
    ]
    video_paths = [
        # Add video paths here
    ]

    run_interactive_calibration(real_values, video_paths)

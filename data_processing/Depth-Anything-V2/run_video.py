import argparse
import cv2
import matplotlib
import numpy as np
import os
import torch
import json
from pathlib import Path

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
 
    parser.add_argument('--sessions-path', type=str, required=True)
    parser.add_argument(
        '--recordings-root',
        type=str,
        default='recordings',
        help='Filesystem root for JSON paths starting with recordings/...'
    )
    parser.add_argument('--input-size', type=int, default=518)
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    # Load sessions
    sessions_path = Path(args.sessions_path).resolve()
    with open(sessions_path, "r") as f:
        sessions = json.load(f)
    sessions = sessions if isinstance(sessions, list) else [sessions]

    recordings_root = Path(args.recordings_root)
    if not recordings_root.is_absolute():
        recordings_root = (sessions_path.parent / recordings_root).resolve()
    else:
        recordings_root = recordings_root.resolve()

    def json_to_abs_path(path_str: str) -> Path:
        norm = str(path_str).replace("\\", "/")
        if norm.startswith("recordings/"):
            rel = norm[len("recordings/"):]
            return (recordings_root / rel).resolve()
        return Path(norm).resolve()

    def abs_to_json_recordings_path(abs_path: Path) -> str:
        rel = os.path.relpath(str(abs_path.resolve()), str(recordings_root))
        return f"recordings\\{rel.replace('/', '\\')}"

    counter = 0
    for session in sessions: 
        session_id = session["session_id"]
        print("\nProcessing session", session_id)
        if "variants" in session:
            for variant in session["variants"]:
                variant_id = variant["variant_id"] #! supposing no variant is skipped, otherwise use a new counter
                print("\tProcessing variant", variant_id)
                update_variant = False
                
                if "videos" in variant:
                    videos = variant["videos"] 
                    if isinstance(videos, list):
                        for v in videos:
                            if "synced_video_path" not in v:
                                print("\t\t\tNo synced_video_path, skipping")
                                continue
                            filename = str(json_to_abs_path(v["synced_video_path"]))
                            video_id = v["video_id"] #! supposing no video is skipped, otherwise use a new counter
                            print("\t\tProcessing video", filename)
                            
                            if "depth_path" not in v:
                                print("\t\t\tComputing depth from video...")

                                raw_video = cv2.VideoCapture(filename)
                                frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
                                
                                if args.pred_only: 
                                    output_width = frame_width
                                else: 
                                    output_width = frame_width * 2 + margin_width

                                output_path_abs = str(Path(filename).with_suffix('')) + "_depth.MP4"
                                os.makedirs(os.path.dirname(output_path_abs), exist_ok=True)
                                out = cv2.VideoWriter(output_path_abs, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))

                                while raw_video.isOpened():

                                    ret, raw_frame = raw_video.read()
                                    if not ret:
                                        break
                                    depth = depth_anything.infer_image(raw_frame, args.input_size)
                                    
                                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                                    depth = depth.astype(np.uint8)
                                    
                                    if args.grayscale:
                                        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                                    else:
                                        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                                    
                                    if args.pred_only:
                                        out.write(depth)
                                    else:
                                        split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                                        combined_frame = cv2.hconcat([raw_frame, split_region, depth])
                                        
                                        out.write(combined_frame)
                                
                                raw_video.release()
                                out.release()

                                # Update sessions
                                depth_json_path = abs_to_json_recordings_path(Path(output_path_abs))
                                sessions[counter]["variants"][variant_id-1]["videos"][video_id-1]["depth_path"] = depth_json_path
                                print("\t\t\tResults written in ", output_path_abs)
                                with open(sessions_path, "w") as f:
                                    json.dump(sessions, f, indent=4)
                                print("\t\t\tJSON file is updated!")
                            else:
                                print("\t\t\tDepth already computed")
                    else:
                        print("\t\tVideo paths in wrong format")  
                else:
                    print("\t\tNo video paths for this variant")
        else:
            print("\tNo variants found for this session")
        counter += 1


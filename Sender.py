import zenoh
import cv2
import numpy as np
import json
import threading
import time
import h5py
import os

# Zenoh configuration for UDP multicast scouting (no specific IPs needed)
# Using a custom port (7450) to avoid potential conflicts with default
# Set the multicast interface to 'auto' (default, but explicitly set for clarity)
# If this doesn't work, replace 'auto' with your network interface name (e.g., 'wlan0' or 'enp0s3')
# To find your interface, run 'ip link show' in terminal and look for the active network device (not 'lo')
conf = zenoh.Config()
conf.insert_json5("scouting/multicast/enabled", "true")
conf.insert_json5("scouting/multicast/address", '"224.0.0.224:7450"')
conf.insert_json5("scouting/multicast/interface", '"auto"')
conf.insert_json5("listen/endpoints", '["udp/0.0.0.0:0"]')  # Auto-assign unicast port

session = zenoh.open(conf)

print(f"Node 1 Zenoh ID: {session.info.zid()}")  # For debugging

# Fixed view configurations
views_fixed = {
    "forward": {"fixed_yaw": 0.0, "fixed_pitch": 0.0, "fixed_roll": 0.0, "use_global": True, "key": "cam_forward"},
    "rear": {"fixed_yaw": 180.0, "fixed_pitch": 0.0, "fixed_roll": 0.0, "use_global": True, "key": "cam_rear"},
    "left": {"fixed_yaw": -90.0, "fixed_pitch": 0.0, "fixed_roll": 0.0, "use_global": True, "key": "cam_left"},
    "right": {"fixed_yaw": 90.0, "fixed_pitch": 0.0, "fixed_roll": 0.0, "use_global": True, "key": "cam_right"},
    "downward": {"fixed_yaw": 0.0, "fixed_pitch": -90.0, "fixed_roll": 0.0, "use_global": False, "key": "cam_downward"}
}

# Default view parameters
defaults = {
    "forward": {"enable": True, "fov_x": 82.0, "fov_y": 82.0, "width": 640, "height": 480, "quality": 80},
    "rear": {"enable": False, "fov_x": 90.0, "fov_y": 90.0, "width": 640, "height": 480, "quality": 80},
    "left": {"enable": False, "fov_x": 90.0, "fov_y": 90.0, "width": 640, "height": 480, "quality": 80},
    "right": {"enable": False, "fov_x": 90.0, "fov_y": 90.0, "width": 640, "height": 480, "quality": 80},
    "downward": {"enable": False, "fov_x": 120.0, "fov_y": 120.0, "width": 480, "height": 640, "quality": 80}
}

# Publishers for each view
pubs = {name: session.declare_publisher(fixed["key"]) for name, fixed in views_fixed.items()}

# Current parameters
current_params = {}
last_params = {name: {} for name in views_fixed}
maps_per_view = {}
params_lock = threading.Lock()

# Handler for receiving params from Node 2
def params_handler(sample):
    global current_params
    try:
        data = json.loads(sample.payload.to_string())
        with params_lock:
            current_params = data
    except Exception as e:
        print(f"Error parsing params: {e}")

sub_params = session.declare_subscriber("view_params", params_handler)

# Recording variables
recording = False
frame_count = 0
h5_file = None
dataset_frames = None
dataset_times = None
h5_path = None
mp4_path = None
recording_lock = threading.Lock()

def start_recording():
    global recording, frame_count, h5_file, dataset_frames, dataset_times, h5_path, mp4_path
    with recording_lock:
        if recording:
            return  # Already recording
        os.makedirs('data', exist_ok=True)
        time_str = time.strftime("%Y%m%d_%H%M%S")
        h5_path = f'data/recording_{time_str}.h5'
        mp4_path = f'data/recording_{time_str}.mp4'
        recording = True
        frame_count = 0
        h5_file = h5py.File(h5_path, 'w')
        dataset_frames = h5_file.create_dataset('frames', shape=(360, 1440, 2880, 3), maxshape=(360, 1440, 2880, 3), dtype=np.uint8, chunks=(1, 1440, 2880, 3))
        dataset_times = h5_file.create_dataset('timestamps', shape=(360,), maxshape=(360,), dtype=np.float64, chunks=True)
        print("Recording started.")

def stop_and_convert(partial=False):
    global recording, h5_file, dataset_frames, dataset_times
    with recording_lock:
        if not recording:
            return
        if partial and frame_count > 0:
            dataset_frames.resize((frame_count, 1440, 2880, 3))
            dataset_times.resize((frame_count,))
        h5_file.close()
        convert_h5_to_mp4(h5_path, mp4_path)
        recording = False
        print("Recording stopped.")

def convert_h5_to_mp4(h5_path, mp4_path):
    with h5py.File(h5_path, 'r') as f:
        frames = f['frames'][:]
        times = f['timestamps'][:]
    if len(times) > 1:
        duration = times[-1] - times[0]
        fps = (len(times) - 1) / duration
    else:
        fps = 29.97
    print(f"Calculated FPS: {fps}")
    fps = 29.97  # Force to 29.97 FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (2880, 1440))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"Converted {h5_path} to {mp4_path}")

# Handler for record command
def record_handler(sample):
    cmd = sample.payload.to_string()
    if cmd == "start":
        start_recording()
    elif cmd == "stop":
        stop_and_convert(partial=True)

sub_record = session.declare_subscriber("record_command", record_handler)

# Camera setup
cap = cv2.VideoCapture(0)  # Adjust device ID if needed
if not cap.isOpened():
    print("Error: Could not open Insta360 X5 camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_FPS, 60)

actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Actual resolution: {int(actual_width)}x{int(actual_height)}")
print(f"FPS: {actual_fps}")

equi_width = int(actual_width)
equi_height = int(actual_height)

# Function to compute rotation matrix from Euler angles
def euler_to_rotation_matrix(yaw, pitch, roll):
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)

    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])

    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])

    return R_yaw @ R_pitch @ R_roll

# Function to update remap coordinates based on params
def update_maps(params):
    rot_list = params["rotation_matrix"]
    rotation_matrix = np.array(rot_list)
    fov_x_deg = params["fov_x"]
    fov_y_deg = params["fov_y"]
    out_width = int(params["width"])
    out_height = int(params["height"])

    fov_x_rad = np.deg2rad(fov_x_deg)
    fov_y_rad = np.deg2rad(fov_y_deg)

    focal_x = (out_width / 2) / np.tan(fov_x_rad / 2)
    focal_y = (out_height / 2) / np.tan(fov_y_rad / 2)

    cx = out_width / 2.0
    cy = out_height / 2.0

    u, v = np.meshgrid(np.arange(out_width), np.arange(out_height))

    x = (u - cx) / focal_x
    y = (v - cy) / focal_y
    z = np.ones_like(x)

    norm = np.sqrt(x**2 + y**2 + z**2)
    dx = x / norm
    dy = y / norm
    dz = z / norm

    directions = np.stack([dx.flatten(), dy.flatten(), dz.flatten()], axis=0)

    rotated_directions = rotation_matrix @ directions
    rdx = rotated_directions[0].reshape(out_height, out_width)
    rdy = rotated_directions[1].reshape(out_height, out_width)
    rdz = rotated_directions[2].reshape(out_height, out_width)

    theta = np.arctan2(rdx, rdz)
    phi = np.arcsin(rdy)

    map_x = equi_width * (theta / (2 * np.pi) + 0.5)
    map_y = equi_height * (phi / np.pi + 0.5)

    return map_x.astype(np.float32), map_y.astype(np.float32), out_width, out_height

# Precompute initial maps
global_yaw = 0.0
global_pitch = 0.0
global_roll = 0.0
views_params = defaults
for view_name, fixed in views_fixed.items():
    vparam = views_params[view_name]
    total_yaw = fixed["fixed_yaw"] + (global_yaw if fixed["use_global"] else 0.0)
    total_pitch = fixed["fixed_pitch"] + (global_pitch if fixed["use_global"] else 0.0)
    total_roll = fixed["fixed_roll"] + (global_roll if fixed["use_global"] else 0.0)
    rot_mat = euler_to_rotation_matrix(total_yaw, total_pitch, total_roll).tolist()
    param = {
        "rotation_matrix": rot_mat,
        "fov_x": vparam["fov_x"],
        "fov_y": vparam["fov_y"],
        "width": vparam["width"],
        "height": vparam["height"]
    }
    map_x, map_y, out_width, out_height = update_maps(param)
    maps_per_view[view_name] = (map_x, map_y, out_width, out_height)
    last_params[view_name] = param

print("Node 1 running: Streaming projected images. Press Ctrl+C to quit.")

connected = False
last_check_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Can't receive frame. Retrying...")
            continue

        with recording_lock:
            if recording:
                dataset_times[frame_count] = time.time()
                dataset_frames[frame_count] = frame
                frame_count += 1
                if frame_count == 360:
                    stop_and_convert()

        with params_lock:
            global_yaw = current_params.get("global_yaw", 0.0)
            global_pitch = current_params.get("global_pitch", 0.0)
            global_roll = current_params.get("global_roll", 0.0)
            global_shift_degree = current_params.get("shift_degree", 0.0)
            views_params = current_params.get("views", {})

        if global_shift_degree != 0:
            shift = int((global_shift_degree / 360.0) * equi_width)
            frame = np.concatenate((frame[:, shift:], frame[:, :shift]), axis=1)

        for view_name, fixed in views_fixed.items():
            vparam = views_params.get(view_name, defaults[view_name])
            if not vparam.get("enable", False):
                continue

            total_yaw = fixed["fixed_yaw"] + (global_yaw if fixed["use_global"] else 0.0)
            total_pitch = fixed["fixed_pitch"] + (global_pitch if fixed["use_global"] else 0.0)
            total_roll = fixed["fixed_roll"] + (global_roll if fixed["use_global"] else 0.0)

            rot_mat = euler_to_rotation_matrix(total_yaw, total_pitch, total_roll).tolist()

            param = {
                "rotation_matrix": rot_mat,
                "fov_x": vparam.get("fov_x", defaults[view_name]["fov_x"]),
                "fov_y": vparam.get("fov_y", defaults[view_name]["fov_y"]),
                "width": vparam.get("width", defaults[view_name]["width"]),
                "height": vparam.get("height", defaults[view_name]["height"])
            }

            quality = vparam.get("quality", defaults[view_name]["quality"])

            if param != last_params[view_name]:
                map_x, map_y, out_width, out_height = update_maps(param)
                maps_per_view[view_name] = (map_x, map_y, out_width, out_height)
                last_params[view_name] = param.copy()

            map_x, map_y, out_width, out_height = maps_per_view[view_name]

            projected = cv2.remap(frame, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)

            _, buf = cv2.imencode('.jpg', projected, [cv2.IMWRITE_JPEG_QUALITY, quality])

            pubs[view_name].put(buf.tobytes(), encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM)

        # Check connection every 1 second
        current_time = time.time()
        if current_time - last_check_time >= 1.0:
            peers = session.info.peers_zid()
            if not connected and len(peers) > 0:
                print(f"Node 1 acknowledged connection to peer: {peers[0]}")
                connected = True
            elif len(peers) == 0:
                print("Node 1: No peers connected yet...")
                with recording_lock:
                    if recording:
                        stop_and_convert(partial=True)
            last_check_time = current_time

except KeyboardInterrupt:
    with recording_lock:
        if recording:
            stop_and_convert(partial=True)

# Cleanup
cap.release()
session.close()
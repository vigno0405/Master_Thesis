# Libraries
import numpy as np # type: ignore
import sys
import time
import rospy
import os
from datetime import datetime
from dynamixel_sdk import *
from geometry_msgs.msg import PoseStamped
import cv2 # type: ignore
from scipy.spatial.transform import Rotation as R # type: ignore
import threading
from ctypes import c_uint32
import matplotlib.pyplot as plt # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore
import termios
import tty
import torch

# User-defined
from jacobians import *
from kinematics import *

# AI libraries
import os
import glob
import random
import warnings
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import torch.nn.functional as F # type: ignore
import shutil
import plotly.graph_objects as go # type: ignore
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau # type: ignore
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt # type: ignore
import random
import sys
from torch.amp import GradScaler, autocast # type: ignore

warnings.filterwarnings("ignore")
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from jacobians import *
import time
from dynamixel_sdk import *
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R # type: ignore
import threading
import cv2
from kinematics import *

# At the beginning, for the optitrack:
# roslaunch optitrack_ros_communication optitrack_nodes.launch

# ======================================================================================
#                                    Functions
# ======================================================================================

# Write register
def write_register(motor_id, address, value, size=1):
    """ Writes a single register """
    if size == 1:
        packet_handler.write1ByteTxRx(port_handler, motor_id, address, value)
    elif size == 2:
        packet_handler.write2ByteTxRx(port_handler, motor_id, address, value)
    elif size == 4:
        packet_handler.write4ByteTxRx(port_handler, motor_id, address, value)

def get_key():
    """Read a single character without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Move motors
def move_motors(d_DELTAL, TOLERANCE=20):
    """
    Moves motors by d_DELTAL increment and returns the actual delta achieved.
    Encoder values are read as uint32 and explicitly reinterpreted as int32 (Dynamixel standard).
    All computations are in float32 to avoid overflow.
    """

    # Read position as signed int32 (from raw uint32) and convert to float
    def read_position_float(motor_id):
        raw, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        signed = np.array(raw, dtype=np.uint32).view(np.int32)
        return float(signed)

    # Initial positions
    start_positions = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)

    # Target positions
    goal_positions = start_positions + (unit_scale * d_DELTAL).astype(np.float32)

    # Send goal to motors
    groupSyncWrite = GroupSyncWrite(port_handler, packet_handler, ADDR_GOAL_POSITION, 4)
    for motor_id, pos in zip(MOTOR_IDs, goal_positions):
        pos_int = int(round(pos))  # Cast to int for sync write
        param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                 DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
        groupSyncWrite.addParam(motor_id, param)
    groupSyncWrite.txPacket()
    groupSyncWrite.clearParam()

    # Wait for completion or timeout
    tic = time.time()
    while time.time() - tic < 1.5:
        current = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)
        if np.all(np.abs(current - goal_positions) <= TOLERANCE):
            break
        time.sleep(0.05)

    # Final positions
    end_positions = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)

    # Compute delta in mm
    d_tick = end_positions - start_positions
    d_DELTAL_real = d_tick / unit_scale

    return d_DELTAL_real

# Calibration function
def calibrate_motors():
    """
    Interactive calibration for each motor.
    Press:
      - 'w' to move forward (with adaptive increment)
      - 's' to move backward (with adaptive increment)
      - 'h' to confirm and go to the next motor
    """
    print("--- Calibration started ---")
    base_step = np.pi * D_pulley / 12  # [mm]
    max_multiplier = 10
    accel_threshold = 0.4  # seconds between presses for acceleration

    for idx, motor_id in enumerate(MOTOR_IDs):
        print(f"\n Motor {motor_id} selected. Press 'h' to skip, 'w' to loosen, 's' to tend.")
        last_key = None
        last_time = time.time()
        step_multiplier = 1

        while True:
            key = get_key()
            now = time.time()

            # Reset multiplier if too slow or switched key
            if key != last_key or (now - last_time) > accel_threshold:
                step_multiplier = 1
            else:
                step_multiplier = min(step_multiplier + 1, max_multiplier)

            last_time = now
            last_key = key

            if key == 'h':
                break
            elif key in ['w', 's']:
                d_DEL = np.zeros(MOTOR_Num, dtype=np.float32)
                direction = 1 if key == 'w' else -1
                step = direction * base_step * step_multiplier
                d_DEL[idx] = step
                move_motors(d_DEL)
            else:
                print("\n Invalid input. Use 'h', 'w', or 's'.")

    print("\n --- Calibration completed ---")

# ======================================================================================
#                                   Dynamixel Setting
# ======================================================================================

DEVICENAME      = '/dev/ttyUSB0'
BAUDRATE        = 57600
PROTOCOL_VERSION = 2.0
EXPOSITION_MODE = 4			# position + current (to be changed)

MOTOR_IDs = np.array([1, 2, 3])	# 3 motors (as in the setup)
MOTOR_Num = MOTOR_IDs.shape[0]	# number of motors

D_pulley = 6				# [mm]
unit_scale = 4096/(np.pi*D_pulley)	# motor position/DELTAL

# ======================================================================================
#                                   Dynamixel Setup
# ======================================================================================

# Initialize PortHandler
port_handler = PortHandler(DEVICENAME)

# Initialize PacketHandler
packet_handler = Protocol2PacketHandler()

# Open port
if not port_handler.openPort():
    print("Failed to open port")
    sys.exit(1)
print("Port opened successfully")

# Set baudrate
if not port_handler.setBaudRate(BAUDRATE):
    print("Failed to set baudrate")
    sys.exit(1)
print("Baudrate set successfully")

# Define Dynamixel control table addresses
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
ADDR_PROFILE_VELOCITY = 112

# Set operating mode for all motors
for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_OPERATING_MODE, EXPOSITION_MODE, size=1)

# Enable torque for all motors
for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_TORQUE_ENABLE, 1, size=1)

VELOCITY_LIMIT = 100     # velocity limitation

for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_PROFILE_VELOCITY, VELOCITY_LIMIT, size=4)

def read_DELTAL_from_motors():
    def read_position_float(motor_id):
        raw, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        signed = np.array(raw, dtype=np.uint32).view(np.int32)
        return float(signed)

    positions = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)
    DELTAL = positions / unit_scale
    return DELTAL

print("--- Reference positions set to zero ---")

# ======================================================================================
#                                Calibrate The Motors
# ======================================================================================

calibrate_motors()

# ======================================================================================
#                          Baseline
# ======================================================================================

# Initial conditions (rest position)
DELTAL0 = np.array([1e-4, 1e-4, 1e-4])
DxDyDl0 = np.array([1e-4, 1e-4, 1e-4])
coords0 = np.array([1e-4, 1e-4, 140, 1e-4, 1e-4, 1e-4])

# ======================================================================================
#                                   Camera Reading
# ======================================================================================

# Use v4l2-ctl --list-devices for the usb camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
ret, image2 = cap.read()
time.sleep(2)

import threading

latest_frame = None
lock = threading.Lock()

def update_camera():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame

camera_thread = threading.Thread(target=update_camera, daemon=True)
camera_thread.start()

def capture_image():
    with lock:
        return latest_frame.copy() if latest_frame is not None else None

time.sleep(2)

# Capture and show image
print("Saving 1 image (every 2s)...")
for i in range(1):
    image = capture_image()
    if image is not None:
        filename = f"frame_{i+1}.png"
        cv2.imwrite(filename, image)
        print(f"[OK] Saved {filename}")
    else:
        print(f"[ERROR] No image at step {i+1}")
    time.sleep(2)

# Model
class Markers(nn.Module):
    def __init__(self, alpha=0.18):
        super().__init__()
        self.alpha = alpha

    def rgb_to_hsv(self, x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        maxc, _ = x.max(dim=1)
        minc, _ = x.min(dim=1)
        v = maxc
        eps = 1e-8

        deltac = maxc - minc
        s = deltac / (maxc + eps)
        s[maxc == 0] = 0

        rc = (maxc - r) / (deltac + eps)
        gc = (maxc - g) / (deltac + eps)
        bc = (maxc - b) / (deltac + eps)

        h = torch.zeros_like(maxc)
        h[maxc == r] = (bc - gc)[maxc == r]
        h[maxc == g] = 2.0 + (rc - bc)[maxc == g]
        h[maxc == b] = 4.0 + (gc - rc)[maxc == b]
        h = (h / 6.0) % 1.0
        h[deltac == 0] = 0.0

        return torch.stack([h, s, v], dim=1)

    def forward(self, x):
        hsv = self.rgb_to_hsv(x)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

        green = ((h > 0.25) & (h < 0.45) & (s > 0.3) & (v > 0.2))
        black = (v < 0.3)
        mask = (green | black).unsqueeze(1)  # [B, 1, H, W]

        x_out = x * self.alpha
        x_out[mask.expand_as(x_out)] = 1.0

        return x_out

from torchvision.models import mobilenet_v2

class NetCamera(nn.Module):
    def __init__(self):
        super().__init__()
        self.markers = Markers()
        base = mobilenet_v2(pretrained=True)
        self.encoder = base.features  # [B, 1280, H, W]
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 6)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.markers(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.head(x)
        return x
    
class DenormalizedModel(nn.Module):
    def __init__(self, base_model, mean, std):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        out = self.base_model(x)
        return out * self.std + self.mean
    
# Load full model
model = torch.load("helyx_model.pt", map_location=torch.device(device))
model = model.to(device)
model.eval()

def CameraSensor():
    image = capture_image()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = image_rgb.astype(np.float32)
    input = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    input_tensor = input.unsqueeze(0).to(device)
    output = model(input_tensor)
    xyz = output[0, :3].detach().cpu().numpy()
    return xyz

# Current position with CameraSensor()
xyz = CameraSensor()
print(f"Current position = {xyz}")

# Function to read position
def read_position_float(motor_id):
    raw, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
    signed = np.array(raw, dtype=np.uint32).view(np.int32)
    return float(signed)

class WriteMotors:
    def __init__(self):
        # Read the initial position just once
        self.start_positions = np.array(
            [read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32
        )

    def __call__(self, d_DELTAL):
        # Computes the destination (difference with respect to the origin)
        goal_positions = self.start_positions + (unit_scale * d_DELTAL).astype(np.float32)

        # Writes on all the motors
        groupSyncWrite = GroupSyncWrite(port_handler, packet_handler, ADDR_GOAL_POSITION, 4)
        for motor_id, pos in zip(MOTOR_IDs, goal_positions):
            pos_int = int(round(pos))
            param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                     DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
            groupSyncWrite.addParam(motor_id, param)
        groupSyncWrite.txPacket()
        groupSyncWrite.clearParam()

# Controller class
class Controller:
    def __init__(self, freq=10, vmax=5, Kp=0.5):

        # Command input
        self.d_DELTAL_integral = np.zeros(3)

        # Control parameters
        self.dt = 1.0 / freq
        self.vmax = vmax
        self.Kp = Kp

        # Starting and final point
        self.A = None
        self.B = None

        # Time (as an internal clock)
        self.time = 0.0

        # Jacobians pre-definition
        self.Q2COORDINATES = q2coordinates()
        self.Q2TENDON = q2tendon()

    def set_target(self, start, end):
        """Defines motion profile"""
        self.A = np.array(start)
        self.B = np.array(end)
        self.dir = self.B - self.A
        self.dist = np.linalg.norm(self.dir)
        self.dir = self.dir / self.dist
        self.total_time = self.dist / self.vmax

    def desired_position(self, t):
        """Computes desired position and velocity"""

        if t >= self.total_time:                    # time overcome
            return self.B, self.dir * self.vmax
        else:
            x_ref = self.A + self.vmax * t * self.dir
            v_ref = self.vmax * self.dir
            return x_ref, v_ref
    
    def kinematics(self, DELTAL):
        """Takes current DELTAL as a vector and restitutes the coordinates"""
        Dl = np.mean(DELTAL)
        delta = np.array([0, 2*np.pi/3, -2*np.pi/3])
        r = Dl - DELTAL
        Dx = (2/3) * np.sum(r * np.cos(delta))
        Dy = (2/3) * np.sum(r * np.sin(delta))

        return np.array([Dx, Dy, Dl])

    def step(self, x, DELTAL_current):
        """Returns d_DELTAL"""
        
        # Error definition, according to the trajectory
        x_ref, x_dot_ref = self.desired_position(self.time)

        # Sum the first part (according to the control law)
        x_dot_k = self.Kp * (x_ref - x)
        x_dot = x_dot_k + x_dot_ref

        # Jacobian for coordinates
        DxDyDl_current = self.kinematics(DELTAL_current)
        J_coords = self.Q2COORDINATES(*DxDyDl_current)
        J_tendon = self.Q2TENDON(*DxDyDl_current)
        J_xyz = J_coords[0:3, 0:3]
        DELTAL_dot = J_tendon @ np.linalg.pinv(J_xyz) @ x_dot

        # Integral computation
        self.d_DELTAL_integral = self.d_DELTAL_integral + DELTAL_dot * self.dt

        # Time adjusting
        self.time = self.time + self.dt

        return self.d_DELTAL_integral, self.time > self.total_time

# Controller
controller = Controller(freq=5)
MotorWriter = WriteMotors()
B = input('Set coordinates (x, y, z) to reach (comma-separated): \n')
end = np.array([float(v.strip()) for v in B.split(',')])
start = CameraSensor()

# Set the controller target
controller.set_target(start, end)
x_current = start.copy()

# DeltaL: read_DELTAL_from_motors() - DELTAL_reference
DELTAL_initial, _, _ = coordinates2tendon(DELTAL0, DxDyDl0, coords0, start - coords0[:3])

# Motors are already moved by DELTAL_current
DELTAL_reference = read_DELTAL_from_motors() - DELTAL_initial

# Initial position
DELTAL_current = read_DELTAL_from_motors() - DELTAL_reference

print(f"Current DELTAL = {DELTAL_current}")

while True:

    start_t = time.time()

    # Computes variation in tendons
    d_DELTAL, reached = controller.step(x_current, DELTAL_current)

    # Apply command
    MotorWriter(d_DELTAL)

    # Read the new position from the camera
    x_current = CameraSensor()

    # Control the frequency
    real_dt = time.time() - start_t
    
    if (controller.dt - real_dt) > 0:
        time.sleep(controller.dt - real_dt)
    else:
        print("Reduce the frequency! \n")
    
    if reached:
        break

x_now = CameraSensor()
print(f"Current x = {x_now}")

cap.release()
cv2.destroyAllWindows()

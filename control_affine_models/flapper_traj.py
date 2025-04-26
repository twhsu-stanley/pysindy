import json
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

import warnings
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
import pickle
import pysindy as ps

from utils import *

def segment_array(arr, segment_length):
    num_segments = len(arr) // segment_length
    return np.array(arr[:num_segments * segment_length]).reshape(num_segments, segment_length, 1)

# Define the root directory
root_dir = '../qualisys_drone_sdk/examples/flapper/circular_traj'

time_horzn = 5.0

# Trajectory data
# states
x_state_traj, y_state_traj, z_state_traj = [], [], []
vx_state_traj, vy_state_traj, vz_state_traj = [], [], []
ax_state_traj, ay_state_traj, az_state_traj = [], [], []
roll_state_traj, pitch_state_traj, yaw_state_traj = [], [], []
roll_rate_state_traj, pitch_rate_state_traj, yaw_rate_state_traj = [], [], []
# control inputs
x_ctrltarget_traj, y_ctrltarget_traj, z_ctrltarget_traj = [], [], []
vx_ctrltarget_traj, vy_ctrltarget_traj, vz_ctrltarget_traj = [], [], []
roll_controller_traj, pitch_controller_traj, yaw_controller_traj = [], [], []
roll_rate_controller_traj, pitch_rate_controller_traj, yaw_rate_controller_traj = [], [], []

i_start = 150
i_end = 650

# Traverse and process
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.json'):
            file_path = os.path.join(dirpath, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)

            poses = data.get("pose", [])
            controls = data.get("control", [])
            dt = data["save_freq"]

            # dt must be uniform across all trajectories
            if dt != 0.1:
                sys.exit("some error message")

            # Segment
            segment_length = math.floor(time_horzn / dt)
                
            # Load position
            position = data["pos"]
            x_state = np.array([p["stateEstimate.x"] for p in position])
            y_state = np.array([p["stateEstimate.y"] for p in position])
            z_state = np.array([p["stateEstimate.z"] for p in position])
            x_state = segment_array(x_state[i_start:i_end], segment_length)
            y_state = segment_array(y_state[i_start:i_end], segment_length)
            z_state = segment_array(z_state[i_start:i_end], segment_length)
            x_state_traj.extend(x_state)
            y_state_traj.extend(y_state)
            z_state_traj.extend(z_state)
            #
            target_pos = data["target_pos"]
            x_ctrltarget = np.array([c["ctrltarget.x"] for c in target_pos])
            y_ctrltarget = np.array([c["ctrltarget.y"] for c in target_pos])
            z_ctrltarget = np.array([c["ctrltarget.z"] for c in target_pos])
            x_ctrltarget = segment_array(x_ctrltarget[i_start:i_end], segment_length)
            y_ctrltarget = segment_array(y_ctrltarget[i_start:i_end], segment_length)
            z_ctrltarget = segment_array(z_ctrltarget[i_start:i_end], segment_length)
            x_ctrltarget_traj.extend(x_ctrltarget)
            y_ctrltarget_traj.extend(y_ctrltarget)
            z_ctrltarget_traj.extend(z_ctrltarget)
            #
            # Load velocity
            velocity = data["vel"]
            vx_state = np.array([v["stateEstimate.vx"] for v in velocity])
            vy_state = np.array([v["stateEstimate.vy"] for v in velocity])
            vz_state = np.array([v["stateEstimate.vz"] for v in velocity])
            vx_state = segment_array(vx_state[i_start:i_end], segment_length)
            vy_state = segment_array(vy_state[i_start:i_end], segment_length)
            vz_state = segment_array(vz_state[i_start:i_end], segment_length)
            vx_state_traj.extend(vx_state)
            vy_state_traj.extend(vy_state)
            vz_state_traj.extend(vz_state)
            #
            target_vel = data["target_vel"]
            vx_ctrltarget = np.array([c["ctrltarget.vx"] for c in target_vel])
            vy_ctrltarget = np.array([c["ctrltarget.vy"] for c in target_vel])
            vz_ctrltarget = np.array([c["ctrltarget.vz"] for c in target_vel])
            vx_ctrltarget = segment_array(vx_ctrltarget[i_start:i_end], segment_length)
            vy_ctrltarget = segment_array(vy_ctrltarget[i_start:i_end], segment_length)
            vz_ctrltarget = segment_array(vz_ctrltarget[i_start:i_end], segment_length)
            vx_ctrltarget_traj.extend(vx_ctrltarget)
            vy_ctrltarget_traj.extend(vy_ctrltarget)
            vz_ctrltarget_traj.extend(vz_ctrltarget)
            #
            # Load acceleration
            acc = data["acc"]
            ax_state = np.array([a["stateEstimate.ax"] for a in acc])
            ay_state = np.array([a["stateEstimate.ay"] for a in acc])
            az_state = np.array([a["stateEstimate.az"] for a in acc])
            ax_state = segment_array(ax_state[i_start:i_end], segment_length)
            ay_state = segment_array(ay_state[i_start:i_end], segment_length)
            az_state = segment_array(az_state[i_start:i_end], segment_length)
            ax_state_traj.extend(ax_state)
            ay_state_traj.extend(ay_state)
            az_state_traj.extend(az_state)
            #
            # Load attitude
            stabilizer = data["stabilizer"]
            roll_state = np.array([c["stabilizer.roll"] for c in stabilizer])
            pitch_state = np.array([c["stabilizer.pitch"] for c in stabilizer])
            yaw_state = np.array([c["stabilizer.yaw"] for c in stabilizer])
            roll_state = segment_array(roll_state[i_start:i_end], segment_length)
            pitch_state = segment_array(pitch_state[i_start:i_end], segment_length)
            yaw_state = segment_array(yaw_state[i_start:i_end], segment_length)
            roll_state_traj.extend(roll_state)
            pitch_state_traj.extend(pitch_state)
            yaw_state_traj.extend(yaw_state)
            #
            controller_attitude = data["controller_attitude"]
            roll_controller = np.array([c["controller.roll"] for c in controller_attitude])
            pitch_controller = np.array([c["controller.pitch"] for c in controller_attitude])
            yaw_controller = np.array([c["controller.yaw"] for c in controller_attitude])
            roll_controller = segment_array(roll_controller[i_start:i_end], segment_length)
            pitch_controller = segment_array(pitch_controller[i_start:i_end], segment_length)
            yaw_controller = segment_array(yaw_controller[i_start:i_end], segment_length)
            roll_controller_traj.extend(roll_controller)
            pitch_controller_traj.extend(pitch_controller)
            yaw_controller_traj.extend(yaw_controller)
            #
            # Load attitude rate
            attitude_rate = data["attitude_rate"] # [milliradians / sec]
            roll_rate_state = np.array([c["stateEstimateZ.rateRoll"]/1000*180/math.pi for c in attitude_rate])
            pitch_rate_state = np.array([c["stateEstimateZ.ratePitch"]/1000*180/math.pi for c in attitude_rate])
            yaw_rate_state = np.array([c["stateEstimateZ.rateYaw"]/1000*180/math.pi for c in attitude_rate])
            roll_rate_state = segment_array(roll_rate_state[i_start:i_end], segment_length)
            pitch_rate_state = segment_array(pitch_rate_state[i_start:i_end], segment_length)
            yaw_rate_state = segment_array(yaw_rate_state[i_start:i_end], segment_length)
            roll_rate_state_traj.extend(roll_rate_state)
            pitch_rate_state_traj.extend(pitch_rate_state)
            yaw_rate_state_traj.extend(yaw_rate_state)
            #
            controller_attitude_rate = data["controller_attitude_rate"]
            roll_rate_controller = np.array([c["controller.rollRate"] for c in controller_attitude_rate])
            pitch_rate_controller = np.array([c["controller.pitchRate"] for c in controller_attitude_rate])
            yaw_rate_controller = np.array([c["controller.yawRate"] for c in controller_attitude_rate])
            roll_rate_controller = segment_array(roll_rate_controller[i_start:i_end], segment_length)
            pitch_rate_controller = segment_array(pitch_rate_controller[i_start:i_end], segment_length)
            yaw_rate_controller = segment_array(yaw_rate_controller[i_start:i_end], segment_length)
            roll_rate_controller_traj.extend(roll_rate_controller)
            pitch_rate_controller_traj.extend(pitch_rate_controller)
            yaw_rate_controller_traj.extend(yaw_rate_controller)

# Convert to NumPy arrays
x_state_traj = np.array(x_state_traj)
y_state_traj = np.array(y_state_traj)
z_state_traj = np.array(z_state_traj)
vx_state_traj = np.array(vx_state_traj)
vy_state_traj = np.array(vy_state_traj)
vz_state_traj = np.array(vz_state_traj)
ax_state_traj = np.array(ax_state_traj)
ay_state_traj = np.array(ay_state_traj)
az_state_traj = np.array(az_state_traj)
roll_state_traj = np.array(roll_state_traj)
pitch_state_traj = np.array(pitch_state_traj)
yaw_state_traj = np.array(yaw_state_traj)
roll_rate_state_traj = np.array(roll_rate_state_traj)
pitch_rate_state_traj = np.array(pitch_rate_state_traj)
yaw_rate_state_traj = np.array(yaw_rate_state_traj)
x_ctrltarget_traj = np.array(x_ctrltarget_traj)
y_ctrltarget_traj = np.array(y_ctrltarget_traj)
z_ctrltarget_traj = np.array(z_ctrltarget_traj)
vx_ctrltarget_traj = np.array(vx_ctrltarget_traj)
vy_ctrltarget_traj = np.array(vy_ctrltarget_traj)
vz_ctrltarget_traj = np.array(vz_ctrltarget_traj)
roll_controller_traj = np.array(roll_controller_traj)
pitch_controller_traj = np.array(pitch_controller_traj)
yaw_controller_traj = np.array(yaw_controller_traj)
roll_rate_controller_traj = np.array(roll_rate_controller_traj)
pitch_rate_controller_traj = np.array(pitch_rate_controller_traj)
yaw_rate_controller_traj = np.array(yaw_rate_controller_traj)

# Construct the dataset of states and control inputs
x_traj = np.concatenate([x_state_traj, y_state_traj, z_state_traj, vx_state_traj, vy_state_traj, vz_state_traj], axis=-1)
x_dot_traj = np.concatenate([vx_state_traj, vy_state_traj, vz_state_traj, ax_state_traj, ay_state_traj, az_state_traj], axis=-1)
u_traj = np.concatenate([x_ctrltarget_traj, y_ctrltarget_traj, z_ctrltarget_traj], axis=-1)

# Shuffle and split the dataset
perm = np.random.permutation(x_traj.shape[0])
x_traj = x_traj[perm,:,:]
x_dot_traj = x_dot_traj[perm,:,:]
u_traj = u_traj[perm,:,:]

num_traj_train = int(x_traj.shape[0] * 0.8)
num_traj_cal = int(x_traj.shape[0] * 0.17)
num_traj_val = x_traj.shape[0] - num_traj_train - num_traj_cal

# Training set
x_train = x_traj[:num_traj_train, :, :]
u_train = u_traj[:num_traj_train, :, :]
x_dot_train = x_dot_traj[:num_traj_train, :, :]
# reshape the training data for training
x_train = x_train.reshape(-1, x_train.shape[-1])
u_train = u_train.reshape(-1, u_train.shape[-1])
x_dot_train = x_dot_train.reshape(-1, x_dot_train.shape[-1])

# Calibration set
x_cal = x_traj[num_traj_train:(num_traj_train+num_traj_cal), :, :]
u_cal = u_traj[num_traj_train:(num_traj_train+num_traj_cal), :, :]
x_dot_cal = x_dot_traj[num_traj_train:(num_traj_train+num_traj_cal), :, :]

# Validation set
x_val = x_traj[(num_traj_train+num_traj_cal):(num_traj_train+num_traj_cal+num_traj_val), :, :]
u_val = u_traj[(num_traj_train+num_traj_cal):(num_traj_train+num_traj_cal+num_traj_val), :, :]
x_dot_val = x_dot_traj[(num_traj_train+num_traj_cal):(num_traj_train+num_traj_cal+num_traj_val), :, :]

# Instantiate and fit the SINDYc model
# Generalized Library (such that it's control affine)
generalized_library = ps.GeneralizedLibrary(
    [ps.PolynomialLibrary(degree = 1),
     ps.FourierLibrary(n_frequencies = 2),
     #ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     #ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     ps.IdentityLibrary() # for control input
    ],
    #tensor_array = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,0], [1,0,1,0]],
    tensor_array = [[1,0,1], [0,1,1], [1,1,0]],
    inputs_per_library = [[0,1,2,3,4,5], [0,1,2,3,4,5], [6,7,8]]
)

# Unconstrained model
model = ps.SINDy(
    optimizer = ps.STLSQ(threshold = 0.01),
    feature_library = generalized_library,
)
model.fit(x_train, x_dot = x_dot_train, u = u_train)
model.print()
print("Feature names:\n", model.get_feature_names())

# Plot prediction
idx_plot = 14
plot_prediction(model, x_cal[idx_plot,:,:], u_cal[idx_plot,:,:], x_dot_cal[idx_plot,:,:], dt)
#plt.figure
#plt.plot(np.arange(x_dot_cal[idx_plot,:,4].shape[0]) * dt, x_dot_cal[idx_plot,:,4])
#plt.show()

# Compute conformal quantile using the calibration set and test it on the validation set
alpha = 0.1
norm = 2
# sup over trajectories
quantile = get_conformal_traj_quantile(model, 
                                       x_cal, u_cal, x_dot_cal, x_val, u_val, x_dot_val,
                                       alpha, norm = 2)


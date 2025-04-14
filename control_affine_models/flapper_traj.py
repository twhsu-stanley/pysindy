import json
import os
import matplotlib.pyplot as plt
import numpy as np
import math

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

# Define the root directory
root_dir = '../flapper_0406'

time_horzn = 3.0

# Storage for segmented data
x1_traj, x2_traj, x3_traj = [], [], []
x1_dot_traj, x2_dot_traj, x3_dot_traj = [], [], []
x1_ddot_traj, x2_ddot_traj, x3_ddot_traj = [], [], []
u1_traj, u2_traj, u3_traj = [], [], []

def segment_array(arr, segment_length):
    num_segments = len(arr) // segment_length
    return np.array(arr[:num_segments * segment_length]).reshape(num_segments, segment_length, 1)

# Traverse and process
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.json'):
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                poses = data.get("pose", [])
                controls = data.get("control", [])
                dt = data.get('save_freq')

                # dt must be uniform across all trajectories
                if dt != 0.1:
                    continue

                # Extract pose entries
                x1_vals = np.array([p[0] for p in poses])
                x2_vals = np.array([p[1] for p in poses])
                x3_vals = np.array([p[2] for p in poses])
                x1_dot_vals = np.gradient(x1_vals, dt)
                x2_dot_vals = np.gradient(x2_vals, dt)
                x3_dot_vals = np.gradient(x3_vals, dt)
                x1_ddot_vals = np.gradient(x1_dot_vals, dt)
                x2_ddot_vals = np.gradient(x2_dot_vals, dt)
                x3_ddot_vals = np.gradient(x3_dot_vals, dt)
                #plt.figure
                #plt.plot(np.arange(x1_ddot_vals.shape[0]) * dt, x1_ddot_vals)
                #plt.show()

                # Extract control entries (handling None)
                u1_vals = np.array([c[0] for c in controls])
                u2_vals = np.array([c[1] for c in controls])
                u3_vals = np.array([c[2] for c in controls])

                # Segment
                segment_length = math.floor(time_horzn / dt)

                x1_segment = segment_array(x1_vals[2:-2], segment_length)
                x2_segment = segment_array(x2_vals[2:-2], segment_length)
                x3_segment = segment_array(x3_vals[2:-2], segment_length)
                
                x1_dot_segment = segment_array(x1_dot_vals[2:-2], segment_length)
                x2_dot_segment = segment_array(x2_dot_vals[2:-2], segment_length)
                x3_dot_segment = segment_array(x3_dot_vals[2:-2], segment_length)

                x1_ddot_segment = segment_array(x1_ddot_vals[2:-2], segment_length)
                x2_ddot_segment = segment_array(x2_ddot_vals[2:-2], segment_length)
                x3_ddot_segment = segment_array(x3_ddot_vals[2:-2], segment_length)
                
                u1_segment = segment_array(u1_vals[2:-2], segment_length)
                u2_segment = segment_array(u2_vals[2:-2], segment_length)
                u3_segment = segment_array(u3_vals[2:-2], segment_length)
                
                # Append data
                x1_traj.extend(x1_segment)
                x2_traj.extend(x2_segment)
                x3_traj.extend(x3_segment)
                x1_dot_traj.extend(x1_dot_segment)
                x2_dot_traj.extend(x2_dot_segment)
                x3_dot_traj.extend(x3_dot_segment)
                x1_ddot_traj.extend(x1_ddot_segment)
                x2_ddot_traj.extend(x2_ddot_segment)
                x3_ddot_traj.extend(x3_ddot_segment)
                u1_traj.extend(u1_segment)
                u2_traj.extend(u2_segment)
                u3_traj.extend(u3_segment)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Convert to NumPy arrays
x1_traj = np.array(x1_traj)
x2_traj = np.array(x2_traj)
x3_traj = np.array(x3_traj)
x1_dot_traj = np.array(x1_dot_traj)
x2_dot_traj = np.array(x2_dot_traj)
x3_dot_traj = np.array(x3_dot_traj)
x1_ddot_traj = np.array(x1_ddot_traj)
x2_ddot_traj = np.array(x2_ddot_traj)
x3_ddot_traj = np.array(x3_ddot_traj)

x_traj = np.concatenate([x1_traj, x2_traj, x3_traj, x1_dot_traj, x2_dot_traj, x3_dot_traj], axis=-1)
x_dot_traj = np.concatenate([x1_dot_traj, x2_dot_traj, x3_dot_traj, x1_ddot_traj, x2_ddot_traj, x3_ddot_traj], axis=-1)

u1_traj = np.array(u1_traj)
u2_traj = np.array(u2_traj)
u3_traj = np.array(u3_traj)
u_traj = np.concatenate([u1_traj, u2_traj, u3_traj], axis=-1)

# Shuffle and split the dataset
perm = np.random.permutation(x_traj.shape[0])
x_traj = x_traj[perm,:,:]
x_dot_traj = x_dot_traj[perm,:,:]
u_traj = u_traj[perm,:,:]

num_traj_train = int(x_traj.shape[0] * 0.6)
num_traj_cal = int(x_traj.shape[0] * 0.37)
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
     ps.FourierLibrary(n_frequencies = 4),
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
idx_plot = 4
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


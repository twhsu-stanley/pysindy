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

# Seed the random number generators for reproducibility
#np.random.seed(100)

# Set up simulation parameters
time_horzn = 10.0
dt = 0.01
time_steps = int(np.ceil(time_horzn/dt))
ang_ind = [2]
u_max = np.pi
u_fun = lambda t: u_max * np.sin(2 * np.pi * 0.5 * t)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-9
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-9

# System parameter
v = 1.0

def dubins_car_dyn(state, u):
    # x_dot = f(x, u)

    x_, y_, theta_ = state

    x_dot = v * np.cos(theta_)
    y_dot = v * np.sin(theta_)
    theta_dot = u
    
    return [x_dot, y_dot, theta_dot]

def dubins_car(t, state, u_fun):
    u = u_fun(t)
    return dubins_car_dyn(state, u)

# Generate a dataset {x_dot_i , (x_i, u_i)}, i=1,...N
num_traj = 2050 # size of the entire dataset
num_traj_train = 50 # size of training set
num_traj_cal = 1000 # size of calibration set
num_traj_val = 1000 # size of validation set
assert num_traj_train + num_traj_cal + num_traj_val == num_traj

x_max = 10.0
y_max = 10.0
theta_max = np.pi
x_range = np.array([
     [-x_max, x_max],
     [-y_max, y_max],
     [-theta_max, theta_max]
])

# Generate random initial states
x0_traj = generate_samples(x_range, num_traj)
x_traj = np.zeros((num_traj, time_steps, 3))
u_traj = np.zeros((num_traj, time_steps, 1))
x_dot_traj = np.zeros((num_traj, time_steps, 3))
for i in range(num_traj):
    x0 = x0_traj[i,:]
    x_temp, x_dot_temp, u_temp = gen_single_trajectory(dubins_car, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)
    x_traj[i,:,:] = x_temp
    u_traj[i,:,:] = u_temp.reshape(-1,1)
    x_dot_traj[i,:,:] = x_dot_temp

# Split the dataset into the training, calibration, and validation sets
# NOTE: Discard the first and last time sample because x_dots by np.gradient are not accurate
# Training set
x_train = x_traj[:num_traj_train, 1:-1, :]
u_train = u_traj[:num_traj_train, 1:-1, :]
x_dot_train = x_dot_traj[:num_traj_train, 1:-1, :]
# reshape the training data for training
x_train = x_train.reshape(-1, x_train.shape[-1])
u_train = u_train.reshape(-1, u_train.shape[-1])
x_dot_train = x_dot_train.reshape(-1, x_dot_train.shape[-1])

# Calibration set
x_cal = x_traj[num_traj_train:(num_traj_train+num_traj_cal), 1:-1, :]
u_cal = u_traj[num_traj_train:(num_traj_train+num_traj_cal), 1:-1, :]
x_dot_cal = x_dot_traj[num_traj_train:(num_traj_train+num_traj_cal), 1:-1, :]

# Validation set
x_val = x_traj[(num_traj_train+num_traj_cal):(num_traj_train+num_traj_cal+num_traj_val), 1:-1, :]
u_val = u_traj[(num_traj_train+num_traj_cal):(num_traj_train+num_traj_cal+num_traj_val), 1:-1, :]
x_dot_val = x_dot_traj[(num_traj_train+num_traj_cal):(num_traj_train+num_traj_cal+num_traj_val), 1:-1, :]

# Instantiate and fit the SINDYc model
# Generalized Library (such that it's control affine)
generalized_library = ps.GeneralizedLibrary(
    [ps.PolynomialLibrary(degree = 5),
     ps.IdentityLibrary() # for control input
    ],
    tensor_array = [[1,1]],
    inputs_per_library = [[2], [3]]
)

# Unconstrained model
model_uc = ps.SINDy(
    optimizer = ps.STLSQ(threshold = 0.0001),
    feature_library = generalized_library,
)
model_uc.fit(x_train, x_dot = x_dot_train, u = u_train)
model_uc.print()
print("Feature names:\n", model_uc.get_feature_names())

model = model_uc

control_affine = check_control_affine(model)
assert control_affine is True

x0 = [0.0, 0.0, 0.0]
test_model_prediction(dubins_car, model, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)

# Compute conformal quantile using the calibration set and test it on the validation set
alpha = 0.1
norm = 2
# method 1: sup over trajectories
quantile = get_conformal_traj_quantile(model,
                                       x_cal, u_cal, x_dot_cal, x_val, u_val, x_dot_val,
                                       alpha, norm = 2)

# method 2: point-wise data
""""
# reshape the data
x_cal = x_cal.reshape(-1, x_cal.shape[-1])
u_cal = u_cal.reshape(-1, u_cal.shape[-1])
x_dot_cal = x_dot_cal.reshape(-1, x_dot_cal.shape[-1])
x_val = x_val.reshape(-1, x_val.shape[-1])
u_val = u_val.reshape(-1, u_val.shape[-1])
x_dot_val = x_dot_val.reshape(-1, x_dot_val.shape[-1])
quantile_comp = get_conformal_quantile(model,
                                  x_cal, u_cal, x_dot_cal, x_val, u_val, x_dot_val,
                                  alpha, norm = 2)
"""

# Save the quantile and alpha as paramters under the model
model_error = {"alpha": alpha, "quantile": quantile, "norm": norm}

model_saved = {"feature_names": model.get_feature_names(), "coefficients": model.optimizer.coef_, "model_error": model_error}

# Save the model and dataset
with open('./control_affine_models/saved_models/model_dubins_car_traj_sindy', 'wb') as file:
    pickle.dump(model_saved, file)

# Testing
with open('./control_affine_models/saved_models/' + 'model_dubins_car_traj_sindy', 'rb') as file:
	model2 = pickle.load(file)
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
time_horzn = 5.0
dt = 0.01
time_steps = int(np.ceil(time_horzn/dt))
ang_ind = [1]
u_max = 10.0
u_fun = lambda t: u_max * np.sin(2 * np.pi * 1 * t)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-9
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-9

# Model parameters
m = 0.3 # pendulum mass
M = 1 # cart mass
L = 1 # length
b = 0 # friction coeff
g = 9.81

def cartpole_dyn(state, u):
    # Cart-pole (single inverted pendulum on a cart, 2-DOF)
    # x_dot = f(x, u)

    z, theta, z_dot, theta_dot = state
    F = u

    det = M + m * (np.sin(theta)**2)
    z_ddot = (F - b * z_dot - m * L * (theta_dot**2) * np.sin(theta)  + 0.5 * m * g * np.sin(2 * theta)) / det
    theta_ddot = (F * np.cos(theta) - 0.5 * m * L * (theta_dot**2) * np.sin(2 * theta) - b * (z_dot * np.cos(theta))
                + (m + M) * g * np.sin(theta)) / (det * L)
    
    return [z_dot, theta_dot, z_ddot, theta_ddot]

def cartpole(t, state, u_fun):
    # Cart-pole (single inverted pendulum on a cart, 2-DOF)
    u = u_fun(t)
    return cartpole_dyn(state, u)

# Generate a dataset {x_dot_i , (x_i, u_i)}, i=1,...N
num_traj = 1250 # size of the entire dataset
num_traj_train = 200 # size of training set
num_traj_cal = 1000 # size of calibration set
num_traj_val = 50 # size of validation set
assert num_traj_train + num_traj_cal + num_traj_val == num_traj

# These maxima below are used for normalization (Tx_inv)
# Thus, they must be consistent with the maxima used in the neural CLF code
z_max = 1.0
theta_max = np.pi/6
v_max = 1.5
omega_max = 1.0
x_range = np.array([
     [-z_max, z_max],
     [-theta_max, theta_max],
     [-v_max, v_max],
     [-omega_max, omega_max]
])

# Generate random initial states
x0_traj = generate_samples(x_range, num_traj)
x_traj = np.zeros((num_traj, time_steps, 4))
u_traj = np.zeros((num_traj, time_steps, 1))
x_dot_traj = np.zeros((num_traj, time_steps, 4))
for i in range(num_traj):
    x0 = x0_traj[i,:]
    x_temp, x_dot_temp, u_temp = gen_single_trajectory(cartpole, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)
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
    [ps.PolynomialLibrary(degree = 2),
     #ps.FourierLibrary(n_frequencies = 1),
     #ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     ps.IdentityLibrary() # for control input
    ],
    tensor_array = [[0,1,0,1], [0,0,1,1], [1,0,0,1], [1,1,0,0], [1,0,1,0]],
    inputs_per_library = [[2,3], [1], [1], [4]]
)

# Unconstrained model
model_uc = ps.SINDy(
    optimizer = ps.STLSQ(threshold = 0.01),
    feature_library = generalized_library,
)
model_uc.fit(x_train, x_dot = x_dot_train, u = u_train)
model_uc.print()
print("Feature names:\n", model_uc.get_feature_names())

model = model_uc

control_affine = check_control_affine(model)
assert control_affine is True

model = set_derivative_coeff(model, [0,1], [2,3]) #x2 (resp. x3) is the time deriv of x0 (resp. x1)
model.print()

x0 = [0.0, 0.0, 0.0, 0.0]
test_model_prediction(cartpole, model, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)

# Compute conformal prediction quantile
alpha = 0.05
norm = 2
x_norm = [z_max, theta_max, v_max, omega_max]
quantile = get_conformal_traj_quantile(model,
                                       x_cal, u_cal, x_dot_cal, x_val, u_val, x_dot_val,
                                       alpha, norm = 2, normalization = x_norm)

# Save the quantile and alpha as paramters under the model
model_error = {"alpha": alpha, "quantile": quantile, "norm": norm}

model_saved = {"feature_names": model.get_feature_names(), "coefficients": model.optimizer.coef_, "model_error": model_error}

# Save the model and dataset
with open('./control_affine_models/saved_models/model_cartpole_traj_sindy', 'wb') as file:
    pickle.dump(model_saved, file)

# Testing
with open('./control_affine_models/saved_models/' + 'model_cartpole_traj_sindy', 'rb') as file:
	model2 = pickle.load(file)
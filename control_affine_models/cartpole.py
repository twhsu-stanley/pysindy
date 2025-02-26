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

# Model parameters {"M": 1.0, "m": 1.0, "L": 0.5, "Kd": 10.0}
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
num_samples = 500000 # size of the entire dataset
num_samples_train = 498000 # size of training set
num_samples_cal = 1000 # size of calibration set
num_samples_val = 1000 # size of validation set
assert num_samples_train + num_samples_cal + num_samples_val == num_samples

z_max = 10.0
theta_max = np.pi
v_max = 40.0
omega_max = np.pi * 4
u_max = 80.0
x_range = np.array([
     [-z_max, z_max],
     [-theta_max, theta_max],
     [-v_max, v_max],
     [-omega_max, omega_max]
])
u_range = np.array([
     [-u_max, u_max]
])
x_samples = generate_samples(x_range, num_samples)
u_samples = generate_samples(u_range, num_samples)
x_dot_samples = np.zeros((num_samples, 4))
for i in range(num_samples):
    x_dot_samples[i,:] = cartpole_dyn(x_samples[i,:], u_samples[i,0])

# Split the dataset into the training, calibration, and validation sets
# Training set
x_train = x_samples[:num_samples_train, :]
u_train = u_samples[:num_samples_train, :]
x_dot_train = x_dot_samples[:num_samples_train, :]
# Calibration set
x_cal = x_samples[num_samples_train:(num_samples_train+num_samples_cal), :]
u_cal = u_samples[num_samples_train:(num_samples_train+num_samples_cal), :]
x_dot_cal = x_dot_samples[num_samples_train:(num_samples_train+num_samples_cal), :]
# Validation set
x_val = x_samples[(num_samples_train+num_samples_cal):(num_samples_train+num_samples_cal+num_samples_val), :]
u_val = u_samples[(num_samples_train+num_samples_cal):(num_samples_train+num_samples_cal+num_samples_val), :]
x_dot_val = x_dot_samples[(num_samples_train+num_samples_cal):(num_samples_train+num_samples_cal+num_samples_val), :]

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
"""
# This also worked
generalized_library = ps.GeneralizedLibrary(
    [ps.PolynomialLibrary(degree = 2),
     #ps.FourierLibrary(n_frequencies = 1),
     ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     ps.IdentityLibrary() # for control input
    ],
    tensor_array = [[0,1,0,0,1], [0,0,1,0,1], [0,0,0,1,1], [1,1,0,0,0], [1,0,1,0,0], [1,0,0,1,0]],
    inputs_per_library = [[2,3], [1], [1], [1], [4]]
)
"""

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

#  Assess results on a test trajectory
time_horzn = 1.0
dt = 0.01
ang_ind = [1]
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-9
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-9
x0 = [0.0, 0.0, 0.0, 0.0]
u_amp_data = np.random.uniform(0, 100)
u_freq_data = np.random.uniform(0, 5)
u_fun = lambda t: u_amp_data * np.sin(2 * np.pi * u_freq_data * t)
test_model_prediction(cartpole, model, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)

# Compute conformal prediction quantile
alpha = 0.05
norm = 2
# These maxima below are used for normalization (Tx_inv)
# Thus, they must be consistent with the maxima used in the neural CLF code
z_max = 1.0
theta_max = np.pi/6
v_max = 1.5
omega_max = 1.0
x_norm = [z_max, theta_max, v_max, omega_max]
quantile = get_conformal_quantile(model,
                                             x_cal, u_cal, x_dot_cal, x_val, u_val, x_dot_val,
                                             alpha, norm = 2, normalization = x_norm)

# Save the quantile and alpha as paramters under the model
model_error = {"alpha": alpha, "quantile": quantile, "norm": norm, "normalization": x_norm}

model_saved = {"feature_names": model.get_feature_names(), "coefficients": model.optimizer.coef_, "model_error": model_error}

# Save the model and dataset
with open('./control_affine_models/saved_models/model_cartpole_sindy', 'wb') as file:
    pickle.dump(model_saved, file)
 
# Testing
with open('./control_affine_models/saved_models/' + 'model_cartpole_sindy', 'rb') as file:
	model2 = pickle.load(file)
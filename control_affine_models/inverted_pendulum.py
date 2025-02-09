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

# System parameters
m = 1 # pendulum mass
L = 1 # length
b = 0.01 # friction coeff
g = 9.81

def ip_dyn(state, u):
    # Inverted pendulum
    # x_dot = f(x, u)

    theta, theta_dot = state
    theta_ddot = (-b*theta_dot + m*g*L*np.sin(theta)/2 ) / (m*L**2/3) - 1/(m*L**2/3) * u
    
    return [theta_dot, theta_ddot]

def ip(t, state, u_fun):
    # Inverted pendulum
    u = u_fun(t)
    return ip_dyn(state, u)

# Generate a dataset {x_dot_i , (x_i, u_i)}, i=1,...N
num_samples = 100000 # size of the entire dataset
num_samples_train = 90000 # size of training set
num_samples_cal = 5000 # size of calibration set
num_samples_val = 5000 # size of validation set
assert num_samples_train + num_samples_cal + num_samples_val == num_samples

theta_max = np.pi
theta_dot_max = 6.0
u_max = 5.0
x_range = np.array([
     [-theta_max, theta_max],
     [-theta_dot_max, theta_dot_max]
])
u_range = np.array([
     [-u_max, u_max]
])
x_samples = generate_samples(x_range, num_samples)
u_samples = generate_samples(u_range, num_samples)
x_dot_samples = np.zeros((num_samples, 2))
for i in range(num_samples):
    x_dot_samples[i,:] = ip_dyn(x_samples[i,:], u_samples[i,0])

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
    [ps.PolynomialLibrary(degree = 4),
     ps.IdentityLibrary() # for control input
    ],
    tensor_array = [[1,1]],
    inputs_per_library = [[0,1], [2]]
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

model = set_derivative_coeff(model, [0], [1]) #x1 is the time derivative of x0
model.print()

# Compute conformal prediction quantile using the calibration set and test it on the validation set
alpha = 0.1
norm = 2
quantile = get_conformal_prediction_quantile(model,
                                             x_cal, u_cal, x_dot_cal, x_val, u_val, x_dot_val,
                                             alpha, norm = 2)

# Save the quantile and alpha as paramters under the model
model_error = {"alpha": alpha, "quantile": quantile, "norm": norm}

model_saved = {"feature_names": model.get_feature_names(), "coefficients": model.optimizer.coef_, "model_error": model_error}

# Save the model and dataset
with open('./control_affine_models/saved_models/model_inverted_pendulum_sindy', 'wb') as file:
    pickle.dump(model_saved, file)

# Testing
with open('./control_affine_models/saved_models/' + 'model_inverted_pendulum_sindy', 'rb') as file:
	model2 = pickle.load(file)

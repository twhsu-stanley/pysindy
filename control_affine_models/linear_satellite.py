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
import dill
import pysindy as ps

from utils import *

# Seed the random number generators for reproducibility
#np.random.seed(100)

# System parameter
a = 500e3
MU = 3.986e14  # Earth's gravitational parameter
n = np.sqrt(MU / a ** 3)

def linear_satellite_dyn(state, u):
    # state_dot = f(state, u)
    #
    # state: 6 x 1
    # u: 3 x 1

    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [3*n**2, 0, 0, 0, 2*n, 0],
                  [0, 0, 0, -2*n, 0, 0],
                  [0, 0, -n**2, 0, 0, 0]])
    
    B = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    # Compute the state derivative
    state_dot = np.dot(A, state) + np.dot(B, u)
    
    return state_dot

def linear_satellite(t, state, u_fun):
    # u: 3 x 1
    u = u_fun(t)
    return linear_satellite_dyn(state, u)

# Generate a dataset {x_dot_i , (x_i, u_i)}, i=1,...N
num_samples = 6108 # size of the entire dataset
num_samples_train = 108 # size of training set
num_samples_cal = 3000 # size of calibration set
num_samples_val = 3000 # size of validation set
assert num_samples_train + num_samples_cal + num_samples_val == num_samples

x_max = 1.0
y_max = 1.0
z_max = 1.0
v_x_max = 5.0
v_y_max = 5.0
v_z_max = 5.0
x_range = np.array([
     [-x_max, x_max],
     [-y_max, y_max],
     [-z_max, z_max],
     [-v_x_max, v_x_max],
     [-v_y_max, v_y_max],
     [-v_z_max, v_z_max],
])
u_range = np.array([
     [-5.0, 5.0],
     [-5.0, 5.0],
     [-5.0, 5.0],
])
x_samples = generate_samples(x_range, num_samples)
u_samples = generate_samples(u_range, num_samples)
x_dot_samples = np.zeros((num_samples, x_range.shape[0]))
for i in range(num_samples):
    x_dot_samples[i,:] = linear_satellite_dyn(x_samples[i,:], u_samples[i,:])

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
     ps.IdentityLibrary() # for control input
    ],
    tensor_array = [[1,1]],
    inputs_per_library = [[0,1,2,3,4,5], [6,7,8]]
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

# Compute conformal prediction quantile
alpha = 0.05
norm = 2
quantile = get_conformal_prediction_quantile(model,
                                             x_cal, u_cal, x_dot_cal, x_val, u_val, x_dot_val,
                                             alpha, norm = 2)

# Save the quantile and alpha as paramters under the model
model_error = {"alpha": alpha, "quantile": quantile, "norm": norm}

model_saved = {"feature_names": model.get_feature_names(), "coefficients": model.optimizer.coef_, "model_error": model_error}

# Save the model and dataset
with open('./control_affine_models/saved_models/model_linear_satellite_sindy', 'wb') as file:
    dill.dump(model_saved, file)
 
# Testing
with open('./control_affine_models/saved_models/' + 'model_linear_satellite_sindy', 'rb') as file:
	model2 = dill.load(file)
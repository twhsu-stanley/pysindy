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
num_samples = 100000 # size of the entire dataset
num_samples_train = 90000 # size of training set
num_samples_cal = 5000 # size of calibration set
num_samples_val = 5000 # size of validation set
assert num_samples_train + num_samples_cal + num_samples_val == num_samples

x_max = 10.0
y_max = 10.0
theta_max = np.pi
x_range = np.array([
     [-x_max, x_max],
     [-y_max, y_max],
     [-theta_max, theta_max]
])
u_range = np.array([
     [-np.pi, np.pi]
])
x_samples = generate_samples(x_range, num_samples)
u_samples = generate_samples(u_range, num_samples)
x_dot_samples = np.zeros((num_samples, 3))
for i in range(num_samples):
    x_dot_samples[i,:] = dubins_car_dyn(x_samples[i,:], u_samples[i,0])

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

# Compute conformal prediction quantile
alpha = 0.05
norm = 2
quantile = get_conformal_quantile(model,
                                  x_cal, u_cal, x_dot_cal, x_val, u_val, x_dot_val,
                                  alpha, norm = 2)

# Save the quantile and alpha as paramters under the model
model_error = {"alpha": alpha, "quantile": quantile, "norm": norm}

model_saved = {"feature_names": model.get_feature_names(), "coefficients": model.optimizer.coef_, "model_error": model_error}

# Save the model and dataset
with open('./control_affine_models/saved_models/model_dubins_car_sindy', 'wb') as file:
    dill.dump(model_saved, file)
 
# Testing
with open('./control_affine_models/saved_models/' + 'model_dubins_car_sindy', 'rb') as file:
	model2 = dill.load(file)
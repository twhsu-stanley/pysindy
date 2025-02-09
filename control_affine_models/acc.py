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
v0 = 15
m  = 2000
f0 = 0.5
f1 = 5.0
f2 = 1.0

def acc_dyn(state, u):
    # x_dot = f(x, u)

    p, v, z = state

    p_dot = v
    v_dot = -(f0 + f1 * v + f2 * v**2) / m + u / m
    z_dot = v0 - v
    
    return [p_dot, v_dot, z_dot]

def acc(t, state, u_fun):
    u = u_fun(t)
    return acc_dyn(state, u)

# Generate a dataset {x_dot_i , (x_i, u_i)}, i=1,...N
num_samples = 2000 # size of the entire dataset
num_samples_train = 200 # size of training set
num_samples_cal = 900 # size of calibration set
num_samples_val = 900 # size of validation set
assert num_samples_train + num_samples_cal + num_samples_val == num_samples

p_max = 100.0
v_max = 50.0
z_max = 20.0
x_range = np.array([
     [-p_max, p_max],
     [-v_max, v_max],
     [-z_max, z_max]
])
u_range = np.array([
     [-10000, 10000]
])
x_samples = generate_samples(x_range, num_samples)
u_samples = generate_samples(u_range, num_samples)
x_dot_samples = np.zeros((num_samples, 3))
for i in range(num_samples):
    x_dot_samples[i,:] = acc_dyn(x_samples[i,:], u_samples[i,0])

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
    [ps.PolynomialLibrary(degree = 1),
     ps.IdentityLibrary() # for control input
    ],
    tensor_array = [[1,1]],
    inputs_per_library = [[0,1,2], [3]]
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
alpha = 0.1
norm = 2
quantile = get_conformal_prediction_quantile(model,
                                             x_cal, u_cal, x_dot_cal, x_val, u_val, x_dot_val,
                                             alpha, norm = 2)

# Save the quantile and alpha as paramters under the model
model_error = {"alpha": alpha, "quantile": quantile, "norm": norm}

model_saved = {"feature_names": model.get_feature_names(), "coefficients": model.optimizer.coef_, "model_error": model_error}

# Save the model and dataset
with open('./control_affine_models/saved_models/model_acc_sindy', 'wb') as file:
    dill.dump(model_saved, file)
 
# Testing
with open('./control_affine_models/saved_models/' + 'model_acc_sindy', 'rb') as file:
	model2 = dill.load(file)
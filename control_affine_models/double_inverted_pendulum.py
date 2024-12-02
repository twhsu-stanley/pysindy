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
np.random.seed(100)

# Set up simulation parameters
time_horzn = 5
dt = 0.002
ang_ind = [2,4]

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-9
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-9

# Randomized initial condition
def x0_fun(): return [0.0, np.random.uniform(-10, 10), np.random.uniform(-np.pi, np.pi), np.random.uniform(-2, 2), np.random.uniform(-np.pi, np.pi), np.random.uniform(-2, 2)]

# Range of the amplitudes and frequencies of the randomized sine inputs
u_amp_range = [0, 100]
u_freq_range = [0, 6]

# Model parameters
M = 5.0
m1 = 2.0
m2 = 1.5
L1 = 0.5
L2 = 0.25
g = 9.81

def double_inverted_pendulum(t, state, u_fun):
    # Double link inverted pendulum on a cart (3-DOF)
    # A(state) * state_ddot = b(state, state_dot) + dL_dstate + u
    # => A(state) * state_ddot = c(state, state_dot, u)
    # => state_ddot = inv( A(state) ) * c(state, state_dot, u)

    x, x_dot, theta_1, theta_1_dot, theta_2, theta_2_dot = state

    theta_1 = wrapTo2pi(theta_1)
    theta_2 = wrapTo2pi(theta_2)

    u = u_fun(t)

    dL_dx = 0.0
    dL_da = -(m1 + m2) * L1 * theta_1_dot * x_dot * np.sin(theta_1) + (m1 + m2) * g * L1 * np.sin(theta_1) - m2 * L1 * L2 * theta_1_dot * theta_2_dot * np.sin(theta_1 - theta_2)
    dL_db = m2 * L2 * (g * np.sin(theta_2) + L1 * theta_1_dot * theta_2_dot * np.sin(theta_1 - theta_2) - x_dot * theta_2_dot * np.sin(theta_2))

    a11 = M + m1 + m2
    a12 = (m1 + m2) * L1 * np.cos(theta_1)
    a13 = m2 * L2 * np.cos(theta_2)
    b1 = (m1 + m2) * L1 * theta_1_dot ** 2 * np.sin(theta_1) + m2 * L2 * theta_2_dot ** 2 * np.sin(theta_2)

    a21 = (m1 + m2) * L1 * np.cos(theta_1)
    a22 = (m1 + m2) * L1 ** 2
    a23 = m2 * L1 * L2 * np.cos(theta_1 - theta_2)
    b2 = (m1 + m2) * x_dot * theta_1_dot * L1 * np.sin(theta_1) + m2 * L1 * L2 * theta_2_dot * (theta_1_dot - theta_2_dot) * np.sin(theta_1 - theta_2)

    a31 = m2 * L2 * np.cos(theta_2)
    a32 = m2 * L1 * L2 * np.cos(theta_1 - theta_2)
    a33 = m2 * L2 ** 2
    b3 = m2 * x_dot * theta_2_dot * L2 * np.sin(theta_2) + m2 * L1 * L2 * theta_1_dot * (theta_1_dot - theta_2_dot) * np.sin(theta_1 - theta_2)

    A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    c = np.array([b1 + dL_dx + u, b2 + dL_da, b3 + dL_db])
    
    # Compute state_dot = np.linalg.inv(A) @ c
    det_A = np.linalg.det(A)

    Ax = np.copy(A)
    Ax[:, 0] = c
    x_ddot = np.linalg.det(Ax) / det_A

    Aa = np.copy(A)
    Aa[:, 1] = c
    theta_1_ddot = np.linalg.det(Aa) / det_A

    Ab = np.copy(A)
    Ab[:, 2] = c
    theta_2_ddot = np.linalg.det(Ab) / det_A

    return [x_dot, x_ddot, theta_1_dot, theta_1_ddot, theta_2_dot, theta_2_ddot]

## Train a SINDYc model using trajectory data
# Generate the training dataset
t_data = np.arange(0, time_horzn, dt)
t_data_span = (t_data[0], t_data[-1])
n_traj_train = 500

x_train, x_dot_train, u_train = gen_trajectory_dataset(double_inverted_pendulum, x0_fun, n_traj_train, time_horzn, dt, 
                                          u_amp_range, u_freq_range, ang_ind, **integrator_keywords)

#plt.plot(t_data, x_train[0])
#plt.show()
#plt.plot(t_data, u_train[0])
#plt.show()

# Instantiate and fit the SINDYc model
# Generalized Library
#poly_library_x_1 = ps.PolynomialLibrary(degree = 1)
poly_library_x_2 = ps.PolynomialLibrary(degree = 2)
fourier_library_x_4 = ps.FourierLibrary(n_frequencies = 4)
fourier_library_x_2 = ps.FourierLibrary(n_frequencies = 2)
poly_library_u = ps.IdentityLibrary() #ps.PolynomialLibrary(degree = 1) # assume control affine

# Initialize the generalized library such that it's control affine
generalized_library = ps.GeneralizedLibrary(
    [poly_library_x_2,
     fourier_library_x_2,
     fourier_library_x_2 * fourier_library_x_2,
     #fourier_library_x_2 * fourier_library_x_2 * fourier_library_x_2,
     poly_library_u
    ],
    tensor_array = [[0,1,0,1], [0,0,1,1], [1,1,0,0], [1,0,1,0]],
    inputs_per_library = [[1,3,5], [2,4], [2,4], [6]]
)

# Unconstrained model
model_uc = ps.SINDy(
    optimizer = ps.STLSQ(threshold = 0.005),
    feature_library = generalized_library,
)
model_uc.fit(x_train, x_dot = x_dot_train, u = u_train, t = dt)
model_uc.print()
print("Feature names:\n", model_uc.get_feature_names())

# Fit the model with constraints at equilibrium points
n_features = model_uc.n_output_features_
constraint_lhs = np.zeros((6, 6 * n_features))
Theta_0 = model_uc.get_regressor(np.zeros((1,6)), u = np.array([[0.0]]))
constraint_lhs[0, :n_features] = Theta_0
constraint_lhs[1, n_features:2*n_features] = Theta_0
constraint_lhs[2, 2*n_features:3*n_features] = Theta_0
constraint_lhs[3, 3*n_features:4*n_features] = Theta_0
constraint_lhs[4, 4*n_features:5*n_features] = Theta_0
constraint_lhs[5, 5*n_features:] = Theta_0

#Theta_pi = model_uc.get_regressor(np.array([[0.0, 0.0, np.pi, 0.0, np.pi, 0.0]]), u = np.array([[0.0]]))
#constraint_lhs[6, :n_features] = Theta_pi
#constraint_lhs[7, n_features:2*n_features] = Theta_pi
#constraint_lhs[8, 2*n_features:3*n_features] = Theta_pi
#constraint_lhs[9, 3*n_features:4*n_features] = Theta_pi
#constraint_lhs[10, 4*n_features:5*n_features] = Theta_pi
#constraint_lhs[11, 5*n_features:] = Theta_pi

optimizer_cnstr = ps.ConstrainedSR3(
    constraint_rhs = np.zeros((6,)),
    constraint_lhs = constraint_lhs,
    equality_constraints = True
)
model = ps.SINDy(
    optimizer = optimizer_cnstr,
    feature_library = generalized_library,
)
model.fit(x_train, x_dot = x_dot_train, u = u_train, t = dt)
model.print()
print("Feature names:\n", model.get_feature_names())

# Verify the constraints are met
coeff = model.optimizer.coef_
Theta = model.get_regressor(np.zeros((1,6)), u = np.array([[0.0]]))
#assert np.all(abs(Theta @ coeff.T - np.zeros((6,))) < 1e-1)

#Theta = model.get_regressor(np.array([[0.0, 0.0, np.pi, 0.0, np.pi, 0.0]]), u = np.array([[0.0]]))
#assert np.all(abs(Theta @ coeff.T - np.zeros((6,))) < 1e-1)

check_control_affine(model)

## Assess results on a test trajectory
# Evolve the equations in time using a different initial condition
x0 = x0_fun() #[0.0, 0.5, 1.5, 0.0, 0.4, 0.0]
u_amp_data = np.random.uniform(5, 100)
u_freq_data = np.random.uniform(1, 5)
u_fun = lambda t: u_amp_data * np.sin(2 * np.pi * u_freq_data * t)
test_model_prediction(double_inverted_pendulum, model, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)

########################################################################################################
## Point-wise Conformal Prediction
# TODO: make this a function

# Generate the calibration dataset
n_traj_cal = 100
x_cal, x_dot_cal, u_cal = gen_trajectory_dataset(double_inverted_pendulum, x0_fun, n_traj_cal, time_horzn, dt,
                                      u_amp_range, u_freq_range, ang_ind, **integrator_keywords)
n_traj_val = 100
x_val, x_dot_val, u_val = gen_trajectory_dataset(double_inverted_pendulum, x0_fun, n_traj_val, time_horzn, dt,
                                      u_amp_range, u_freq_range, ang_ind, **integrator_keywords)

nc_score = []
for i in range(n_traj_cal):
    #err = (model.predict(x_cal[i], u = u_cal[i]) - model.differentiate(x_cal[i], t = dt))
    err = (model.predict(x_cal[i], u = u_cal[i]) - x_dot_cal[i])
    R = np.linalg.norm(err, 2, axis = 1)
    nc_score.extend(R)

alpha = 0.05
n = len(nc_score)
quantile = np.quantile(nc_score, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")

# Empirical coverage
emp_scores = []
for i in range(n_traj_val):
    #err = (model.predict(x_val[i], u = u_val[i]) - model.differentiate(x_val[i], t = dt))
    err = (model.predict(x_val[i], u = u_val[i]) - x_dot_val[i])
    R = np.linalg.norm(err, 2, axis = 1)
    emp_scores.extend(R)
emp_coverage = sum(i < quantile for i in emp_scores) / len(emp_scores)
print("Quantile for alpha = %5.3f is %5.3f" % (alpha, quantile))
print("Empirical Coverage = %5.3f vs. 1-alpha = %5.3f" % (emp_coverage, 1- alpha))
########################################################################################################

## Save the model and dataset
with open('./control_affine_models/saved_models/model_double_inverted_pendulum_sindy', 'wb') as file:
    dill.dump(model, file)

trajectory_data = {'x_cal': x_cal, 'u_cal': u_cal, 'x_val': x_val, 'u_val': u_val, 'dt': dt}
with open('./control_affine_models/trajectory_data/' + 'traj_double_inverted_pendulum_sindy', 'wb') as file:
	pickle.dump(trajectory_data, file)     

# Testing
with open('./control_affine_models/saved_models/' + 'model_double_inverted_pendulum_sindy', 'rb') as file:
	model2 = dill.load(file)
     
with open('./control_affine_models/trajectory_data/' + 'traj_double_inverted_pendulum_sindy', 'rb') as file:
    trajectory_data2 = pickle.load(file)
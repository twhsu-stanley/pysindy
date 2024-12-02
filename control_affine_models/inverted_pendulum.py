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

import math

from test_picklable import *

t_end_data = 5
t_end_test = 5

# Seed the random number generators for reproducibility
np.random.seed(100)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-9
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-9

# ## SINDy with control (SINDYc)
# Here we learn a inverted penduum model given by:
# $$ \dot{theta} = theta_dot $$
# $$ \dot{theta_dot} = g/L * sin(theta) -b/(m*L^2) * theta_dot + 1/(m*L^2) * u $$
def inverted_pendulum(t, x, u_fun, L = 1, m = 1, b = 0.01):
    g = 9.81
    u = u_fun(t)
    return [
        x[1],
        g/L * np.sin(x[0]) -b/(m*L**2) * x[1] + 1/(m*L**2) * u,
    ]

## Generate the dataset
dt = 0.002
t_data = np.arange(0, t_end_data, dt)
t_data_span = (t_data[0], t_data[-1])
x_data = []
u_data = []
n_traj = 2000
for i in range(n_traj):
    u_amp_data = np.random.uniform(0, 100)
    u_freq_data = np.random.uniform(0, 5)
    u_fun = lambda t: u_amp_data * np.sin(2 * np.pi * u_freq_data * t)
    x0_data = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
    x_traj = solve_ivp(
        inverted_pendulum,
        t_data_span,
        x0_data,
        t_eval = t_data,
        args = (u_fun,),
        **integrator_keywords,
    ).y.T
    x_data.append(x_traj)
    u_data.append(u_fun(t_data))

#plt.plot(t_data, x_data[0])
#plt.show()
#plt.plot(t_data, u_data[0])
#plt.show()

# Split the dataset into training and calibration sets
# TODO: make this a function
D1 = 500 #math.floor(n_traj * 0.02) # Size of the training set
D2 = 1000 #n_traj - D1  # Size of the calibration set
D3 = n_traj - D1 - D2

# Random shuffling
xu_data = list(zip(x_data, u_data))
np.random.shuffle(xu_data)
x_data, u_data = zip(*xu_data)

# Split the data
x_train = x_data[:D1]
u_train = u_data[:D1]
x_cal = x_data[D1:D1+D2]
u_cal = u_data[D1:D1+D2]
x_val = x_data[D1+D2:]
u_val = u_data[D1+D2:]

# TODO: Add noise to training data?
for i in range(D1):
    for j in range(x_train[i].shape[0]):
        x_train[i][j] = x_train[i][j] + np.random.normal(0.0, 0.4, (x_train[i].shape[1]))

# Instantiate and fit the SINDYc model
# Generalized Library
poly_library_x = ps.IdentityLibrary() # ps.PolynomialLibrary(degree = 1)
fourier_library_x = ps.FourierLibrary(n_frequencies = 1)
poly_library_u = ps.IdentityLibrary() # assume control affine

# Initialize the generalized library such that it's control affine
generalized_library = ps.GeneralizedLibrary(
    [poly_library_x, fourier_library_x, poly_library_u],
    tensor_array = [[1,1,0], [1,0,1], [0,1,1]],
    #exclude_libraries = [0,1,2],
    inputs_per_library = [[0,1], [0,1], [2]],
)

# Unconstrained model
model_uc = ps.SINDy(
    optimizer = ps.STLSQ(threshold = 0.005),
    feature_library = generalized_library,
)
model_uc.fit(x_train, u = u_train, t = dt)
model_uc.print()
print("Feature names:\n", model_uc.get_feature_names())

# Fit the model with constraints
constraint_lhs = np.zeros((2, 2 * model_uc.n_output_features_))
Theta_0 = model_uc.get_regressor(np.array([[0.0,0.0]]), u = np.array([[0.0]]))
constraint_lhs[0,:model_uc.n_output_features_] = Theta_0
constraint_lhs[1,model_uc.n_output_features_:] = Theta_0

optimizer_cnstr = ps.ConstrainedSR3(
    constraint_rhs = np.array([0.0, 0.0]), 
    constraint_lhs = constraint_lhs,
    equality_constraints = True
)
model = ps.SINDy(
    optimizer = optimizer_cnstr,
    feature_library = generalized_library,
)
model.fit(x_train, u = u_train, t = dt)
model.print()
print("Feature names:\n", model.get_feature_names())

#unpicklable_attr = remove_unpicklable(model)

# Verify the constraints are met
Theta = model.get_regressor(np.array([[0.0,0.0]]), u = np.array([[0.0]]))
coeff = model.optimizer.coef_
assert np.all(abs(Theta @ coeff.T - [0.0, 0.0]) < 1e-10)

# TESTING: testing the control affine form ################################
#"""
Xt = np.array([[0.0,2.0]])
Ut = np.array([[5.0]])
Theta = model.get_regressor(Xt, u = np.array([[1.0]]))
coeff = model.optimizer.coef_
feature_names = model.get_feature_names()
idx_x = [] # Indices for f(x)
idx_u = [] # Indices for g(x)*u
for i in range(len(feature_names)):
    if 'u0' in feature_names[i]:
        idx_u.append(i)
    else:
        idx_x.append(i)

# Get f(x)
f_of_x = Theta[:,idx_x] @ coeff[:,idx_x].T

# Get g(x)
g_of_x = Theta[:,idx_u] @ coeff[:,idx_u].T

Err = abs(f_of_x + g_of_x * Ut - model.get_regressor(Xt, Ut) @ coeff.T)
for i in range(len(Err[0])):
    if Err[0][i] > 1e-2:
        print("f_of_x + g_of_x * Ut = ", f_of_x + g_of_x * Ut)
        print("Theta @ coeff.T = ", model.get_regressor(Xt, Ut) @ coeff.T)
        raise ValueError("f_of_x + g_of_x * Ut != Theta @ coeff.T; Check if the model is control affine")
#"""
###########################################################################

## Assess results on a test trajectory
# Evolve the equations in time using a different initial condition
t_test = np.arange(0, t_end_test, dt)
t_test_span = (t_test[0], t_test[-1])
u_amp_test = np.random.uniform(10, 100)
u_freq_test = np.random.uniform(0, 5)
u_fun = lambda t: u_amp_test * np.sin(2 * np.pi * u_freq_test * t)
x0_test = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
x_test = solve_ivp(
    inverted_pendulum,
    t_test_span,
    x0_test,
    t_eval = t_test,
    args = (u_fun,),
    **integrator_keywords,
).y.T

u_test = u_fun(t_test)

# Compare SINDy-predicted derivatives with finite difference derivatives
print("Model score: %f" % model.score(x_test, u = u_test, t=dt))

### Predict derivatives with learned model

# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test, u = u_test)

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(x_test, t = dt)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_dot_test_computed[:, i], "k", label="numerical derivative")
    axs[i].plot(t_test, x_dot_test_predicted[:, i], "r--", label="model prediction")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
fig.show()

# Point-wise Conformal Prediction
nc_score = []
for i in range(D2):
    err = (model.predict(x_cal[i], u = u_cal[i]) - model.differentiate(x_cal[i], t = dt))
    R = np.linalg.norm(err, 2, axis = 1)
    nc_score.extend(R)

alpha = 0.05
n = len(nc_score)
quantile = np.quantile(nc_score, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")

# Empirical coverage
emp_scores = []
for i in range(D3):
    err = (model.predict(x_val[i], u = u_val[i]) - model.differentiate(x_val[i], t = dt))
    R = np.linalg.norm(err, 2, axis = 1)
    emp_scores.extend(R)
emp_coverage = sum(i < quantile for i in emp_scores) / len(emp_scores)
print("Empirical Coverage = %5.3f vs. 1-alpha = %5.3f" % (emp_coverage, 1- alpha))

# ### Simulate forward in time (control input function known)
# When working with control inputs `SINDy.simulate` requires a *function* to be passed in for the control inputs, `u`, 
# because the default integrator used in `SINDy.simulate` uses adaptive time-stepping. 
# We show what to do in the case when you do not know the functional form for the control inputs in the example following this one.

# Evolve the new initial condition in time with the SINDy model
x_test_sim = model.simulate(x0_test, t_test, u = u_fun, integrator_kws = integrator_keywords)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], "k", label="true simulation")
    axs[i].plot(t_test, x_test_sim[:, i], "r--", label="model simulation")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel="$x_{}$".format(i))

fig.show()

# ### Simulate forward in time (unknown control input function)
# If you only have a vector of control input values at the times in `t_test` and do not know the functional form for `u`, the `simulate` function will internally form an interpolating function based on the vector of control inputs. As a consequence of this interpolation procedure, `simulate` will not give a state estimate for the last time point in `t_test`. This is because the default integrator, `scipy.integrate.solve_ivp` (with LSODA as the default solver), is adaptive and sometimes attempts to evaluate the interpolant outside the domain of interpolation, causing an error.
u_test = u_fun(t_test)
x_test_sim = model.simulate(x0_test, t_test, u = u_test, integrator_kws = integrator_keywords)

# Note that the output is one example short of the length of t_test
print("Length of t_test:", len(t_test))
print("Length of simulation:", len(x_test_sim))

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test[:-1], x_test[:-1, i], "k", label="true simulation")
    axs[i].plot(t_test[:-1], x_test_sim[:, i], "r--", label="model simulation")
    axs[i].set(xlabel="t", ylabel="$x_{}$".format(i))

plt.show(block=True)
plt.pause(0.001)

# Save the model
with open('./control_affine_models/saved_models/model_inverted_pendulum_sindy', 'wb') as file:
    dill.dump(model, file)

trajectory_data = {'x_cal': x_cal, 'u_cal': u_cal, 'x_val': x_val, 'u_val': u_val, 'dt': dt}
with open('./control_affine_models/trajectory_data/' + 'traj_inverted_pendulum_sindy', 'wb') as file:
	pickle.dump(trajectory_data, file)     

# Testing
with open('./control_affine_models/saved_models/' + 'model_inverted_pendulum_sindy', 'rb') as file:
	model2 = dill.load(file)
     
with open('./control_affine_models/trajectory_data/' + 'traj_inverted_pendulum_sindy', 'rb') as file:
    trajectory_data2 = pickle.load(file)
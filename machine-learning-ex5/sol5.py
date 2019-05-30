from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

import re
import time
import math
import copy

import numpy.fft
import numpy.linalg
import numpy.random

import scipy.fftpack
import scipy.interpolate
import scipy.optimize
import scipy.io

import matplotlib.pyplot as plt
#%%
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex5\ex5')
exp1_data_mat = scipy.io.loadmat('ex5data1.mat')

# Training data set
x_no_x0_arr = exp1_data_mat['X']
y_arr = exp1_data_mat['y']
x_arr = np.ones((x_no_x0_arr.shape[0], x_no_x0_arr.shape[1]+1))
x_arr[:, 1:] = x_no_x0_arr

# Cross-validation data set
y_val_arr = exp1_data_mat['yval']

x_val_no_x0_arr = exp1_data_mat['Xval']
x_val_arr = np.ones((x_val_no_x0_arr.shape[0], x_val_no_x0_arr.shape[1]+1))
x_val_arr[:, 1:] = x_val_no_x0_arr

# Test data
x_test_no_x0_arr = exp1_data_mat['Xtest']
y_test_arr = exp1_data_mat['ytest']

x_test_arr = np.ones((x_test_no_x0_arr.shape[0], x_test_no_x0_arr.shape[1]+1))
x_test_arr[:, 1:] = x_test_no_x0_arr
#%%
def poly_J(theta_arr, x_arr, y_arr, reg_const):
    m = x_arr.shape[0]
    theta_reshaped_arr = theta_arr.reshape(1, theta_arr.shape[0])
    h_theta_arr = x_arr @ theta_reshaped_arr.T
    return np.asscalar(1 / (2*m) * ((h_theta_arr-y_arr).T @ (h_theta_arr-y_arr) + reg_const * theta_reshaped_arr[:, 1:] @ theta_reshaped_arr[:, 1:].T))

def poly_J_grad(theta_arr, x_arr, y_arr, reg_const):
    m = x_arr.shape[0]
    theta_reshaped_arr = theta_arr.reshape(1, theta_arr.shape[0])
    h_theta_arr = x_arr @ theta_reshaped_arr.T
    grad_arr = (1 / m * (h_theta_arr - y_arr).T @ x_arr + reg_const / m * theta_reshaped_arr)
    grad_arr[:, 0] = grad_arr[:, 0] - reg_const / m * theta_reshaped_arr[:, 0]
    return grad_arr.flatten()
#%%
theta_arr = np.array([1, 1])
reg_const = 0
#%%
poly_J(theta_arr, x_arr, y_arr, reg_const)
#%%
poly_J_grad(theta_arr, x_arr, y_arr, reg_const)
#%%
theta_min_arr = scipy.optimize.minimize(fun=poly_J, x0=theta_arr, args=(x_arr, y_arr, reg_const), method='L-BFGS-B', jac=poly_J_grad, options={'disp': None, 'maxiter': 2000})['x']
theta_min_arr
#%%
x_fit_arr = np.ones((x_arr.shape[0] * 10, x_arr.shape[1]))

for i in range(1, x_arr.shape[1]):
    x_fit_arr[:, i] = np.linspace(np.min(x_arr[:, i]), np.max(x_arr[:, i]), x_fit_arr.shape[0])

y_fit_arr = x_fit_arr @ theta_min_arr
#%%
# Plot the data

fig, ax = plt.subplots()
ax.scatter(x_no_x0_arr, y_arr)
ax.plot(x_fit_arr[:, 1:], y_fit_arr, color='red')
plt.show()
#%%
reg_const = 0

min_m = x_arr.shape[1]
m_arr = np.linspace(min_m, x_arr.shape[0], x_arr.shape[0]-min_m).astype(np.int)

tr_err_arr = np.zeros(m_arr.size)
val_err_arr = np.zeros(m_arr.size)

for i in range(m_arr.shape[0]):
    m = m_arr[i]
    x_subset_arr = x_arr[:m]
    y_subset_arr = y_arr[:m]
    theta_min_arr = scipy.optimize.minimize(fun=poly_J, x0=theta_arr, args=(x_subset_arr, y_subset_arr, reg_const), method='L-BFGS-B', jac=poly_J_grad, options={'disp': None})['x']

    tr_err_arr[i] = poly_J(theta_min_arr, x_subset_arr, y_subset_arr, reg_const=0)
    val_err_arr[i] = poly_J(theta_min_arr, x_val_arr, y_val_arr, reg_const=0)
#%%
fig, ax = plt.subplots()
ax.plot(m_arr, tr_err_arr, color='blue')
ax.plot(m_arr, val_err_arr, color='red')

plt.show()
#%%
def norm_feature(x_arr):
    x_part_arr = x_arr[:, 1:]
    return np.column_stack((x_arr[:, 0], (x_part_arr - np.mean(x_part_arr, axis=0)) / np.std(x_part_arr, axis=0, ddof=1))), np.mean(x_part_arr, axis=0), np.std(x_part_arr, axis=0, ddof=1)
#%%
x_norm_arr, x_mean_arr, x_std_arr = norm_feature(x_arr)
#%%
# Add additional features with x**p where p is 0..8
pow_max = 4

x_new_arr = np.zeros((x_arr.shape[0], int(pow_max + 1)))
x_val_new_arr = np.zeros((x_val_arr.shape[0], int(pow_max + 1)))
x_test_new_arr = np.zeros((x_test_arr.shape[0], int(pow_max + 1)))

for i in range(0, pow_max+1):
    x_new_arr[:, i] = x_arr[:, 1]**i
    x_val_new_arr[:, i] = x_val_arr[:, 1]**i
    x_test_new_arr[:, i] = x_test_arr[:, 1]**i

theta_arr = np.ones(x_new_arr[0].shape[0])

x_norm_arr, x_mean_arr, x_std_arr = norm_feature(x_new_arr)

x_val_norm_arr = np.column_stack((x_val_new_arr[:, 0], (x_val_new_arr[:, 1:] - x_mean_arr) / x_std_arr))

x_test_norm_arr = np.column_stack((x_test_new_arr[:, 0], (x_test_new_arr[:, 1:] - x_mean_arr) / x_std_arr))
#%%
reg_const = 1

theta_min_arr = scipy.optimize.minimize(fun=poly_J, x0=theta_arr, args=(x_norm_arr, y_arr, reg_const), method='L-BFGS-B', jac=poly_J_grad, options={'disp': None})['x']

x_fit_arr = np.zeros((x_new_arr.shape[0] * 20, x_new_arr[0].shape[0]))

for i in range(x_fit_arr[0].shape[0]):
    x_fit_arr[:, i] = np.linspace(np.min(x_new_arr[:, 1]), np.max(x_new_arr[:, 1]), x_fit_arr.shape[0])**i

x_fit_norm_arr = np.column_stack((x_fit_arr[:, 0], (x_fit_arr[:, 1:] - x_mean_arr) / x_std_arr))

y_fit_arr = x_fit_norm_arr @ theta_min_arr.reshape(1, theta_min_arr.shape[0]).T
x_denorm_arr = x_fit_norm_arr[:, 1:] * x_std_arr + x_mean_arr

sort_index_arr = np.argsort(x_no_x0_arr[:, 0])
sort_index_fit_arr = np.argsort(x_denorm_arr[:, 0])

fig, ax = plt.subplots()

ax.scatter(x_no_x0_arr[sort_index_arr], y_arr[sort_index_arr], color='blue')
#ax.plot(x_no_x0_arr[sort_index_arr], y_arr[sort_index_arr], color='blue', linestyle='--')
ax.plot(x_denorm_arr[:, 0][sort_index_fit_arr], y_fit_arr[:,0][sort_index_fit_arr], color='red')

ax.set_ylim(np.min(y_fit_arr[:, 0])-5, np.max(y_fit_arr[:, 0])+5)

plt.show()
#%%
reg_const = 3

min_m = x_arr.shape[1]
m_arr = np.linspace(min_m, x_new_arr.shape[0], x_new_arr.shape[0]-min_m).astype(np.int)

tr_err_arr = np.zeros(m_arr.size)
val_err_arr = np.zeros(m_arr.size)

for i in range(m_arr.shape[0]):
    m = m_arr[i]
    x_subset_arr = x_norm_arr[:m]
    y_subset_arr = y_arr[:m]
    theta_min_arr = scipy.optimize.minimize(fun=poly_J, x0=theta_arr, args=(x_subset_arr, y_subset_arr, reg_const), method='L-BFGS-B', jac=poly_J_grad, options={'disp': None})['x']

    tr_err_arr[i] = poly_J(theta_min_arr, x_subset_arr, y_subset_arr, reg_const=0)
    val_err_arr[i] = poly_J(theta_min_arr, x_val_norm_arr, y_val_arr, reg_const=0)
#%%
fig, ax = plt.subplots()
ax.plot(m_arr, tr_err_arr, color='blue')
ax.plot(m_arr, val_err_arr, color='red')

plt.show()
#%%
reg_const_arr = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

tr_err_arr = np.zeros(reg_const_arr.size)
val_err_arr = np.zeros(reg_const_arr.size)

for i in range(reg_const_arr.shape[0]):
    theta_min_arr = scipy.optimize.minimize(fun=poly_J, x0=theta_arr, args=(x_norm_arr, y_arr, reg_const_arr[i]), method='L-BFGS-B', jac=poly_J_grad, options={'disp': None})['x']

    tr_err_arr[i] = poly_J(theta_min_arr, x_norm_arr, y_arr, reg_const=0)
    val_err_arr[i] = poly_J(theta_min_arr, x_val_norm_arr, y_val_arr, reg_const=0)
#%%
fig, ax = plt.subplots()
ax.plot(reg_const_arr, tr_err_arr, color='blue')
ax.plot(reg_const_arr, val_err_arr, color='red')

plt.show()
#%%
val_err_arr
#%%
reg_const = 3

theta_min_arr = scipy.optimize.minimize(fun=poly_J, x0=theta_arr, args=(x_norm_arr, y_arr, reg_const), method='L-BFGS-B', jac=poly_J_grad, options={'disp': None})['x']

print('Average deviation: ' + str(np.sqrt(poly_J(theta_min_arr, x_test_norm_arr, y_test_arr, reg_const))))

#%%
# Training, cross-validation and test sets
fig, ax = plt.subplots()
ax.scatter(x_test_no_x0_arr, y_test_arr)
ax.scatter(x_test_no_x0_arr, x_test_norm_arr @ theta_min_arr.T)

ax.scatter(x_no_x0_arr, y_arr)
ax.scatter(x_no_x0_arr, x_norm_arr @ theta_min_arr.T)

plt.show()
#%%
# Randomly selecting m samples for the training set and the cross-validation set, and determining the corresponding errors.

# Number of times we randomly select a set of m values from the two sets.
num_random_select = 100

reg_const = 1

min_m = 1
m_arr = np.linspace(min_m, x_norm_arr.shape[0], x_norm_arr.shape[0]-min_m).astype(np.int)

tr_err_arr = np.zeros((m_arr.size, num_random_select))
val_err_arr = np.zeros((m_arr.size, num_random_select))

for i in range(m_arr.shape[0]):

    for j in range(num_random_select):

        m = m_arr[i]
        rand_index = np.random.randint(low=0, high=x_norm_arr.shape[0], size=m)
        x_subset_arr = x_norm_arr[rand_index]
        y_subset_arr = y_arr[rand_index]
        theta_min_arr = scipy.optimize.minimize(fun=poly_J, x0=theta_arr, args=(x_subset_arr, y_subset_arr, reg_const), method='L-BFGS-B', jac=poly_J_grad, options={'disp': None})['x']

        tr_err_arr[i, j] = poly_J(theta_min_arr, x_subset_arr, y_subset_arr, reg_const=0)
        val_err_arr[i, j] = poly_J(theta_min_arr, x_val_norm_arr, y_val_arr, reg_const=0)
#%%
fig, ax = plt.subplots()
ax.plot(m_arr, np.mean(tr_err_arr, axis=1), color='blue')
ax.plot(m_arr, np.mean(val_err_arr, axis=1), color='red')

plt.show()

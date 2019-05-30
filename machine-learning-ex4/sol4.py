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
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex4\ex4')
exp1_data_mat = scipy.io.loadmat('ex4data1.mat')
x_no_x0_arr = exp1_data_mat['X']
y_arr = exp1_data_mat['y']

x_arr = np.ones((x_no_x0_arr.shape[0], x_no_x0_arr.shape[1]+1))
x_arr[:, 1:] = x_no_x0_arr
#%%
''' Initial visualization of 100 random training sets
'''
n_img = 100
ncols = int(np.sqrt(n_img))
nrows = int(np.sqrt(n_img))
img_index_arr = np.random.randint(low=0, high=x_no_x0_arr.shape[0], size=n_img).reshape(nrows, ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

fig.set_size_inches(7, 7)

for i in range(nrows):
    for j in range(ncols):
        axes[i, j].imshow(x_no_x0_arr[img_index_arr[i, j]].reshape(20, 20).T * 255, aspect='auto', cmap=plt.cm.binary)
        axes[i, j].set_xticklabels([])
        axes[i, j].set_yticklabels([])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
plt.show()
#%%
def sig_func(z_arr):
    ''' Sigmoid function
    '''
    return 1 / (1 + np.exp(- 1 * z_arr))
#%%
''' Calculation output of the neural network with the weights provided with the assignment
'''
nn_weights_data_mat = scipy.io.loadmat('ex4weights.mat')

theta_1_arr = nn_weights_data_mat['Theta1']
theta_2_arr = nn_weights_data_mat['Theta2']

# Input layer with the bias term already included
a1_arr = x_arr

# Hidden layer
a2_arr = np.ones((a1_arr.shape[0], theta_1_arr.shape[0]+1))
a2_arr[:, 1:] = sig_func(a1_arr @ theta_1_arr.T)

# Output layer
out_arr = sig_func(a2_arr @ theta_2_arr.T)

val2_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Training accuracy with the supplied weights.
training_nn_acc = y_arr.T[0][val2_arr[np.argmax(out_arr, axis=1)] == y_arr.T[0]].shape[0] / y_arr.T[0].shape[0] * 100
training_nn_acc
#%%
''' Functions for neural networks
'''
def sig_nn_out(theta_arr, x_arr, layer_shape_arr, layer_size_arr):
    ''' Calculation of the nn output
    '''
    # Calculate activation of the output layer
    a_prev_arr = x_arr.copy()
    theta_last_index = 0

    # For each layer of the network
    for i in range(layer_shape_arr.shape[0]):
        theta_i_arr = theta_arr[theta_last_index:theta_last_index+layer_size_arr[i]].reshape(layer_shape_arr[i])
        theta_last_index = theta_last_index+layer_size_arr[i]
        a_arr = np.ones((a_prev_arr.shape[0], theta_i_arr.shape[0]+1))
        a_arr[:, 1:] = sig_func(a_prev_arr @ theta_i_arr.T)
        a_prev_arr = a_arr.copy()

    # For the output layer the bias term is not needed
    h_theta_arr = a_arr[:, 1:]
    return h_theta_arr

def sig_J_nn(theta_arr, x_arr, y_arr, layer_shape_arr, layer_size_arr, reg_const):
    ''' Calculation of the nn error function
    '''
    # Calculate activation of the output layer
    a_prev_arr = x_arr.copy()
    theta_last_index = 0
    reg_term = 0
    # For each layer of the network
    for i in range(layer_shape_arr.shape[0]):
        theta_i_arr = theta_arr[theta_last_index:theta_last_index+layer_size_arr[i]].reshape(layer_shape_arr[i])
        theta_last_index = theta_last_index+layer_size_arr[i]
        a_arr = np.ones((a_prev_arr.shape[0], theta_i_arr.shape[0]+1))
        a_arr[:, 1:] = sig_func(a_prev_arr @ theta_i_arr.T)
        a_prev_arr = a_arr.copy()

        reg_term = reg_term + theta_i_arr[:, 1:].flatten() @ theta_i_arr[:, 1:].flatten()
    # For the output layer the bias term is not needed
    h_theta_arr = a_arr[:, 1:]

    cost = 0
    for k in range(y_arr[0].shape[0]):
        h_theta_k_arr = h_theta_arr[:, k]
        y_k_arr = y_arr[:, k]
        cost = cost + np.asscalar(-np.log(h_theta_k_arr) @ y_k_arr - np.log(1-h_theta_k_arr) @ (1-y_k_arr))

    m = x_arr.shape[0]
    cost = 1 / m * cost + reg_const / (2*m) * reg_term

    return cost

def sig_grad(z_arr):
    ''' Derivative of the sigmoid function
    '''
    return sig_func(z_arr) * (1-sig_func(z_arr))

def rand_weights_init(layer_shape_arr, layer_size_arr):
    ''' Random initialization of nn weights
    '''
    theta_init_arr = []
    for k in range(layer_shape_arr.shape[0]):
        if k != layer_shape_arr.shape[0]-1:
            if k == 0:
                eps = np.sqrt(6 / (layer_shape_arr[k, 1] - 1 + layer_shape_arr[k+1, 0] - 1))
            else:
                if (k == layer_shape_arr.shape[0]-1-1) and (k != 0):
                    eps = np.sqrt(6 / (layer_shape_arr[k-1, 0] -1 + layer_shape_arr[k+1, 0]))
                else:
                    eps = np.sqrt(6 / (layer_shape_arr[k-1, 0] -1 + layer_shape_arr[k+1, 0] - 1))

        theta_i_arr = (np.random.rand(layer_size_arr[k]) - 0.5) * 2 * eps
        theta_init_arr = np.concatenate((theta_init_arr, theta_i_arr))
    return theta_init_arr

def sig_J_grad_nn(theta_arr, x_arr, y_arr, layer_shape_arr, layer_size_arr, reg_const):
    ''' Backpropagation algorithm. Not fully vectorized: the calculation is performed for each training set separately.
    '''
    # Number of training sets
    m = x_arr.shape[0]
    # List of weights for the layers  (L-1 arrays, where L is the # number of layers for the network)
    theta_layer_list = []

    # List of activations values for the layers
    z_layer_list = []
    a_layer_list = []

    # Calculate output values for all of the training sets
    a_prev_arr = x_arr.copy()
    theta_last_index = 0
    for l in range(layer_shape_arr.shape[0]):
        theta_l_arr = theta_arr[theta_last_index:theta_last_index+layer_size_arr[l]].reshape(layer_shape_arr[l])
        theta_last_index = theta_last_index+layer_size_arr[l]
        theta_layer_list.append(theta_l_arr)

        a_arr = np.ones((a_prev_arr.shape[0], theta_l_arr.shape[0]+1))
        z_l_arr = np.ones((a_prev_arr.shape[0], theta_l_arr.shape[0]+1))
        z_l_arr[:, 1:] = a_prev_arr @ theta_l_arr.T
        z_layer_list.append(z_l_arr)

        a_arr[:, 1:] = sig_func(z_l_arr[:, 1:])
        a_layer_list.append(a_arr)

        a_prev_arr = a_arr.copy()

    # For the output layer the bias term is not needed
    h_theta_arr = a_arr[:, 1:]

    # The list of activations consists of L-1 arrays, where L is the number of layers of the neural network. For convenience, we would want to have L arrays, such that the first element ([0]) corresponds to the input array
    a_layer_list.insert(0, x_arr)
    # We do the same for z-values, for easier indexing
    z_layer_list.insert(0, 0)
    # print(len(a_layer_list))

    # List of error terms. For convenience, it has L elements, even though the first element, corresponding the error of the first layer is not needed (no error is defined for the first layer). This is done for easier indexing
    delta_list = [0] * (layer_shape_arr.shape[0]+1)

    # List of \Delta terms
    big_delta_list = [0] * (layer_shape_arr.shape[0])

    # We perform calculation for each training set
    for i in range(h_theta_arr.shape[0]):

        # Calculation of the error for the Lth layer
        delta_L_arr = h_theta_arr[i] - y_arr[i, :].T[0]
        delta_list[-1] = delta_L_arr.reshape(delta_L_arr.shape[0], 1)

        # Starting from the L-1 layer, up to the first layer
        for l in reversed(range(layer_shape_arr.shape[0])):

            delta_l_plus_1_arr = delta_list[l+1]
            theta_l_arr = theta_layer_list[l]

            a_l_arr = a_layer_list[l][i].reshape(a_layer_list[l][i].shape[0], 1)

            big_delta_list[l] = big_delta_list[l] + delta_l_plus_1_arr @ a_l_arr.T

            # If we are at the first layer, then there is no point in calculating the error array
            if l != 0:
                z_l_arr = z_layer_list[l][i].reshape(z_layer_list[l][i].shape[0], 1)
                delta_l_arr = theta_l_arr.T @ delta_l_plus_1_arr * sig_grad(z_l_arr)
                # It is important to remove the row in the array, corresponding to the bias term of the previous array.
                delta_l_arr = delta_l_arr[1:]
                delta_list[l] = delta_l_arr

    # At this moment we have the big_delta_arr, which has the same shape as the array of weights for each of the layers. We would want to form one continuous array, however, which is done below. In addition, we also apply the regularization here

    big_delta_arr = np.zeros(theta_arr.size)
    big_delta_index = 0
    for l in range(len(big_delta_list)):

        arr = big_delta_list[l]

        # Regularization is applied to all of the columns, expect the bias column
        arr[:, 1:] = arr[:, 1:] + reg_const * theta_layer_list[l][:, 1:]

        # Flatten the array
        big_delta_arr[big_delta_index:big_delta_index+arr.size] = arr.flatten()

        big_delta_index = big_delta_index + arr.size

    # Calculate the gradients of the sigmoid cost function for the neural network.
    D_arr = big_delta_arr / m

    return D_arr

def sig_J_grad_nn_test(theta_arr, x_arr, y_arr, layer_shape_arr, layer_size_arr, reg_const, eps):
    ''' Slow, but direct way of computing gradients in a neural network

    Should be used for testing only.
    '''
    grad_arr = np.zeros(theta_arr.shape)
    for i in range(theta_arr.shape[0]):
        theta_plus_arr = theta_arr.copy()
        theta_plus_arr[i] = theta_plus_arr[i] + eps
        theta_minus_arr = theta_arr.copy()
        theta_minus_arr[i] = theta_minus_arr[i] - eps

        h_theta_plus_arr = sig_J_nn(theta_plus_arr, x_arr, y_arr, layer_shape_arr, layer_size_arr, reg_const)
        h_theta_minus_arr = sig_J_nn(theta_minus_arr, x_arr, y_arr, layer_shape_arr, layer_size_arr, reg_const)

        grad_arr[i] = (h_theta_plus_arr - h_theta_minus_arr) / (2*eps)

    return grad_arr
#%%
'''Initialization + checking of the backpropagation code
'''
# Convert y-values into column vectors
val_arr = np.sort(pd.Series(y_arr[:, 0]).drop_duplicates().values)

y_new_arr = np.zeros((y_arr.shape[0], y_arr.shape[1] * 10, 1))

for i in range(val_arr.shape[0]):
    val_i_arr = np.zeros((1, val_arr.shape[0]))
    val_i_arr[0, val_arr[i]-1] = 1
    val_i_arr = val_i_arr.T
    y_new_arr[np.nonzero(y_arr.flatten() == val_arr[i])] = val_i_arr

# Shape of the neural network. It should be sufficient to specify these two arrays to optimize a neural network with any number of layers + units.

layer_shape_arr = np.array([theta_1_arr.shape, theta_2_arr.shape])
layer_size_arr = np.array([theta_1_arr.size, theta_2_arr.size])

theta_arr = np.concatenate((theta_1_arr.flatten(), theta_2_arr.flatten()))

print('Cost function test: ' + str(sig_J_nn(theta_arr, x_arr, y_new_arr, layer_shape_arr, layer_size_arr, reg_const=1)))

# Initialize random array of weights
theta_0_arr = rand_weights_init(layer_shape_arr, layer_size_arr)

# Testing of the backpropagation algorithm = gradient checking
min_t_set_index = 100
max_t_set_index = 110

reg_const = 1
sig_j_grad_arr = sig_J_grad_nn(theta_0_arr, x_arr[min_t_set_index:max_t_set_index], y_new_arr[min_t_set_index:max_t_set_index], layer_shape_arr, layer_size_arr, reg_const=reg_const)

sig_j_grad_arr_test = sig_J_grad_nn_test(theta_0_arr, x_arr[min_t_set_index:max_t_set_index], y_new_arr[min_t_set_index:max_t_set_index], layer_shape_arr, layer_size_arr, reg_const=reg_const, eps=10**-2)
print('Max difference: ' + str(np.max(np.abs(sig_j_grad_arr - sig_j_grad_arr_test))))
#%%
''' Minimization procedure. Using L-BFGS-B method. See comments for details
'''

# Regularization parameter
reg_const = 1

# It seems that it takes about 10 min to perform the minimization for reg_const=1. If BFGS method is used, then the code immediately throws an error: memory error. With the L-BFGS-B no memory errors occur. More about this in https://en.wikipedia.org/wiki/Limited-memory_BFGS.
theta_min_arr = scipy.optimize.minimize(fun=sig_J_nn, x0=theta_0_arr, args=(x_arr, y_new_arr, layer_shape_arr, layer_size_arr, reg_const), method='L-BFGS-B', jac=sig_J_grad_nn, options={'disp': None, 'maxiter': 2000})
#%%
theta_0_to_use_arr = theta_min_arr['x']
#%%
# We want to make sure that the cost function is indeed minimized. It is useful to look at the maximum absolute gradient value.
np.max(np.abs(sig_J_grad_nn(theta_0_to_use_arr, x_arr, y_new_arr, layer_shape_arr, layer_size_arr, reg_const=reg_const)))
#%%
# Suprisingly, the array of weights supplied with the assignment is not that good.
np.max(np.abs(sig_J_grad_nn(theta_arr, x_arr, y_new_arr, layer_shape_arr, layer_size_arr, reg_const=reg_const)))
#%%
# This is the value of the cost function at the minimum.
sig_J_nn(theta_0_to_use_arr, x_arr, y_new_arr, layer_shape_arr, layer_size_arr, reg_const=reg_const)
#%%
# Predictions = output of the neural network giving the set of the training sets.
out_arr = sig_nn_out(theta_0_to_use_arr, x_arr, layer_shape_arr, layer_size_arr)
#%%
# Determination of the training accuracy.
val2_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

training_nn_acc = y_arr.T[0][val2_arr[np.argmax(out_arr, axis=1)] == y_arr.T[0]].shape[0] / y_arr.T[0].shape[0] * 100
training_nn_acc
#%%
'''
Visualization of the hidden layer
'''

# Find theta matrices for the neural network
# List of weights for the layers  (L-1 arrays, where L is the # number of layers for the network)
theta_layer_list = []

# Calculate output values for all of the training sets
theta_last_index = 0
for l in range(layer_shape_arr.shape[0]):
    theta_l_arr = theta_0_to_use_arr[theta_last_index:theta_last_index+layer_size_arr[l]].reshape(layer_shape_arr[l])
    theta_last_index = theta_last_index+layer_size_arr[l]
    theta_layer_list.append(theta_l_arr)

# Hidden layer to visualize. The bias unit is discarded (I guess, to make the matrix square)
theta_hidden_arr = theta_layer_list[0][:,1:]

# 25 units in total for the hidden layer
nrows = 5
ncols = 5

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

fig.set_size_inches(7, 7)

for i in range(nrows):
    for j in range(ncols):
        axes[i, j].imshow(theta_hidden_arr[nrows*i + j].reshape(20, 20).T * 255, aspect='auto', cmap=plt.cm.binary)
        axes[i, j].set_xticklabels([])
        axes[i, j].set_yticklabels([])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
plt.show()

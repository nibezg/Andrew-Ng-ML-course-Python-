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
from scipy import sparse


import matplotlib.pyplot as plt

from sklearn import svm

import seaborn as sns

from nltk import PorterStemmer

from collections import Counter

import matplotlib.patches
#%%
# Load and visualize the data
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex8\ex8')
# os.chdir(r'C:\Users\user\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex8\ex8')
exp1_data_df = scipy.io.loadmat('ex8data1.mat')
x_m = exp1_data_df['X']
x_cv_m = exp1_data_df['Xval']
y_cv_v = exp1_data_df['yval'][:, 0]

fig, ax = plt.subplots()

ax.scatter(x_m[:, 0], x_m[:, 1], s=1.5)
plt.show()
#%%
# Determine the prob. density + visualize it
def findProbDens(var_x_v, mean_x_v, x_m):
    ''' Calculate joint probability density, assuming independent features
    '''
    prob_dens_m = 1 / np.sqrt(2 * np.pi * var_x_v) * np.exp(-1 / (2 * var_x_v) * (x_m-mean_x_v)**2)

    prob_dens_indep_v = np.ones(prob_dens_m.shape[0])

    for i in range(prob_dens_m.shape[1]):
        prob_dens_indep_v = prob_dens_indep_v * prob_dens_m[:, i]

    return prob_dens_indep_v

mean_x_v = np.mean(x_m, axis=0)
var_x_v = np.std(x_m, ddof=1, axis=0)**2

n_points = 200

x_range_m = np.linspace(np.min(x_m[:, 0])-5, np.max(x_m[:, 0])+5, n_points)
y_range_m = np.linspace(np.min(x_m[:, 1])-5, np.max(x_m[:, 1])+5, n_points)

xx_arr, yy_arr = np.meshgrid(x_range_m, y_range_m)

x_plot_m = np.concatenate((xx_arr.reshape(1, xx_arr.size).T, yy_arr.reshape(1, yy_arr.size).T), axis=1)

prob_dens_indep_v = findProbDens(var_x_v, mean_x_v, x_plot_m).reshape(xx_arr.shape[0], yy_arr.shape[0])

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax.contour(x_range_m, y_range_m, prob_dens_indep_v, levels=10.0**np.arange(-20, 0, 3)
)
ax.scatter(x_m[:, 0], x_m[:, 1], s=1.5)

ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
#ax.set_ylim(np.min(X_fit[:, 1]), np.max(X_fit[:, 1]))
#ax.set_xlim(np.min(X_fit[:, 0]), np.max(X_fit[:, 0]))

plt.show()
#%%
def getThreshold(mean_x_v, var_x_v, x_cv_m, y_cv_v, eps_db_step):
    ''' Determine the threshold probability value using the probability distribution from the training data (by providing the mean and the variance of the training data), and the cross-validation dataset.
    '''
    prob_dens_cv_v = findProbDens(var_x_v, mean_x_v, x_cv_m)

    # Constructing the array of test threshold values. Note that there is no point in having the threshold values larger and smaller than the largest and the smallest estimates probabilities for the given cross-validation set
    min_eps = np.min(prob_dens_cv_v)
    max_eps = np.max(prob_dens_cv_v)

    # The array values are small and it is convenient to use log scale to specify the step size for the array. We convert it back to usual scale after creating the array, however
    eps_arr = 10.0 ** (np.arange(10 * np.log10(min_eps), 10 * np.log10(max_eps), eps_db_step) / 10)
    #eps_arr = np.linspace(min_eps, max_eps, n_step)

    f_score_arr = np.zeros(eps_arr.shape[0])

    for i in range(eps_arr.shape[0]):
        eps = eps_arr[i]

        # Indeces of false + true positives
        pred_pos = np.argwhere((prob_dens_cv_v < eps) == True)[:, 0]
        # Indeces of false + true negatives
        pred_neg = np.argwhere((prob_dens_cv_v < eps) == False)[:, 0]

        # True positives
        t_p = np.sum(y_cv_v[pred_pos])
        # False positives
        f_p = pred_pos.shape[0] - t_p

        # False negative
        f_n = np.sum(y_cv_v[pred_neg])
        # True negatives
        t_n = pred_neg.shape[0] - f_n

        if t_p + f_p == 0:
            prec = 1E30
        else:
            prec = t_p / (t_p + f_p)

        if t_p + f_n == 0:
            rec = 1E30
        else:
            rec = t_p / (t_p + f_n)

        f_score_arr[i] = 2 * prec * rec / (prec + rec)

    # Sometimes the maximum f_score happens for several threshold values. We would prefer to use the largest threshold. To yse the argmax function, that gives the index of the first (from left to right) index, which is the maximum value of an array, we flip the arrays first
    f_score_arr = np.flip(f_score_arr)
    eps_arr = np.flip(eps_arr)

    max_f_score_index = np.argmax(f_score_arr)
    return f_score_arr[max_f_score_index], eps_arr[max_f_score_index]
#%%
f_score, eps = getThreshold(mean_x_v, var_x_v, x_cv_m, y_cv_v, eps_db_step=0.01)
#%%
# The threshold is about 5 times larger than the value in the assignment. The reason is that there are quite a few threshold values that give the same f score. And, I guess, I am using a different scale for the threshold values + I am selecting the largest threshold value - as the most conservative choice.
print('F1 score: ' + str(f_score) + ', eps: ' + str(eps))
#%%
prob_dens_v = findProbDens(var_x_v, mean_x_v, x_m)
outlier_index_v = np.argwhere(prob_dens_v < eps)[:, 0]
x_outlier_m = x_m[outlier_index_v]
#%%
# Number of outliers
x_outlier_m.shape[0]
#%%
# Visualization of the outliers
fig, ax = plt.subplots()

fig.set_size_inches(8, 8)

for i in range(x_outlier_m.shape[0]):
    circ_patch = matplotlib.patches.Circle(xy=x_outlier_m[i], radius=0.4, fill=False, edgecolor='red', linewidth=4)
    ax.add_patch(circ_patch)

ax.scatter(x_m[:, 0], x_m[:, 1], s=3)
ax.contour(x_range_m, y_range_m, prob_dens_indep_v, levels=10.0**np.arange(-20, 0, 3))

ax.set_xlim(0, 30)
ax.set_ylim(0, 30)

plt.show()
#%%
''' Anomaly detection for a high-dimensional dataset
'''

# Load the data
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex8\ex8')
# os.chdir(r'C:\Users\user\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex8\ex8')
exp2_data_df = scipy.io.loadmat('ex8data2.mat')

x_m = exp2_data_df['X']
x_cv_m = exp2_data_df['Xval']
y_cv_v = exp2_data_df['yval'][:, 0]

mean_x_v = np.mean(x_m, axis=0)
var_x_v = np.std(x_m, ddof=0, axis=0)**2

f_score, eps = getThreshold(mean_x_v, var_x_v, x_cv_m, y_cv_v, eps_db_step=0.01)
#%%
print('F1 score: ' + str(f_score) + ', eps: ' + str(eps))
#%%
prob_dens_v = findProbDens(var_x_v, mean_x_v, x_m)
outlier_index_v = np.argwhere(prob_dens_v < eps)[:, 0]
x_outlier_m = x_m[outlier_index_v]
#%%
# Using the suggested method of selecting the threshold, but with converting the threshold values to dB scale, we get quite different result for anomaly detection - the difference is more than a factor of 10 in the number of samples that are flagged as anomalous. It also shows how the many-dimensional datasets have complicated relationship with the threshold.
x_outlier_m.shape[0]
#%%
'''
Recommender Systems
'''
# Load the data
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex8\ex8')
# os.chdir(r'C:\Users\user\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex8\ex8')
exp3_data_df = scipy.io.loadmat('ex8_movies.mat')
#%%
y_m = exp3_data_df['Y']
r_m = exp3_data_df['R']

# Visualize the data
yy_v, xx_v = np.meshgrid(np.arange(0, y_m.shape[0], 5), np.arange(y_m.shape[1]))

fig, ax = plt.subplots()

ax.contourf(xx_v, yy_v, y_m[::5].T)

plt.show()
#%%
# Average rating for the first movie
movie_index = 0
np.mean(y_m[movie_index][r_m[movie_index] == 1])
#%%
# Collaborative filtering learning algorithm

# Pre-computed X and theta matrices for testing the calculation of the cost function
movie_rating_data_df = scipy.io.loadmat('ex8_movieParams.mat')
x_ex_m = movie_rating_data_df['X']
theta_ex_m = movie_rating_data_df['Theta']
#%%
def cf_J(r_m, y_m, x_m, theta_m, reg_const=0):
    ''' Collaborative filtering cost function (no regularization)
    '''
    c_m = (x_m @ theta_m.T - y_m)**2
    return 1/2 * np.sum(c_m*r_m) + 1/2 * reg_const * (np.sum(theta_m**2)+np.sum(x_m**2))

# Testing the cost function
n_user = 4
n_movie = 5
n_feature = 3
cf_J(r_m[:n_movie, :n_user], y_m[:n_movie, :n_user], x_ex_m[:n_movie, :n_feature], theta_ex_m[:n_user, :n_feature])
#%%
def cf_J_grad(r_m, y_m, x_m, theta_m, reg_const=0):
    ''' Compute the gradient of the collaborative cost function.

    Partial derivates with respect to x_m and theta_m are calculated. The final Jacobian is returned in a flattened form with the partial derivates for x_m being first in the vector.
    '''
    # The gradient of the cost function is the following:
    # For the kth component of that for the jth user:
    # dJ/d theta^(j)_k = sum_{i (r(i,j)=1)} (theta^(j) x^(i)^T - y^(i,j))*x^(i)_k
    jac_J_theta_m = np.zeros(theta_m.shape)

    for j in range(jac_J_theta_m.shape[0]):
        theta_v = theta_m[j]

        idx_v = np.argwhere(r_m[:, j]==1).T[0]

        x_sel_m = x_m[idx_v]
        y_sel_m = y_m[idx_v, j]

        jac_J_theta_m[j] = (theta_v @ x_sel_m.T - y_sel_m) @ x_sel_m + reg_const * theta_v

    jac_J_x_m = np.zeros(x_m.shape)

    for i in range(jac_J_x_m.shape[0]):
        x_v = x_m[i]

        idx_v = np.argwhere(r_m[i]==1).T[0]
        theta_sel_m = theta_m[idx_v]
        y_sel_m = y_m[i, idx_v]

        jac_J_x_m[i] = (theta_sel_m @ x_v.T - y_sel_m) @ theta_sel_m + reg_const * x_v

    jac_v = np.append(jac_J_x_m.flatten(),jac_J_theta_m.flatten())
    return jac_v
#%%
def calc_J_grad_approx(r_m, y_m, x_m, theta_m, reg_const=0, eps:
        '''Compute the approximate gradient of the collaborative cost function by calculating the cost function at, for example, x-eps and x+eps.

        The final partial derivatives is returned in a flattened form with the partial derivates for x_m being first in the vector.
        '''

        jac_J_theta_m = np.zeros(theta_m.shape)

        for i in range(jac_J_theta_m.shape[0]):
            for j in range(jac_J_theta_m.shape[1]):
                theta_m[i, j] = theta_m[i, j] + eps

                J_theta_plus = cf_J(r_m, y_m, x_m, theta_m, reg_const)
                theta_m[i, j] = theta_m[i, j] - 2 * eps
                J_theta_minus = cf_J(r_m, y_m, x_m, theta_m, reg_const)
                theta_m[i, j] = theta_m[i, j] + eps

                jac_J_theta_m[i, j] = (J_theta_plus - J_theta_minus) / (2 * eps)

        jac_J_x_m = np.zeros(x_m.shape)

        for i in range(jac_J_x_m.shape[0]):
            for j in range(jac_J_x_m.shape[1]):
                x_m[i, j] = x_m[i, j] + eps

                J_x_plus = cf_J(r_m, y_m, x_m, theta_m, reg_const)
                x_m[i, j] = x_m[i, j] - 2 * eps
                J_x_minus = cf_J(r_m, y_m, x_m, theta_m, reg_const)
                x_m[i, j] = x_m[i, j] + eps

                jac_J_x_m[i, j] = (J_x_plus - J_x_minus) / (2 * eps)


        approx_jac_v = np.append(jac_J_x_m.flatten(),jac_J_theta_m.flatten())
        return approx_jac_v
#%%
# Testing the gradient algorithm against the simple approximate method

# First a subset of the data is selected - to make the computation faster
n_user = 15
n_movie = 10
n_feature = 5

approx_grad_v = calc_J_grad_approx(r_m[:n_movie, :n_user], y_m[:n_movie, :n_user], x_ex_m[:n_movie, :n_feature], theta_ex_m[:n_user, :n_feature], eps=1E-5)

exact_grad_v = cf_J_grad(r_m[:n_movie, :n_user], y_m[:n_movie, :n_user], x_ex_m[:n_movie, :n_feature], theta_ex_m[:n_user, :n_feature])

print('Test of the algorithm for calculation of the gradient. Maximum |difference between the approximate gradient and the exact gradient| is: ' + str(np.max(np.abs(exact_grad_v - approx_grad_v))))
#%%
# Testing the regularized cost function

n_user = 4
n_movie = 5
n_feature = 3
cf_J(r_m[:n_movie, :n_user], y_m[:n_movie, :n_user], x_ex_m[:n_movie, :n_feature], theta_ex_m[:n_user, :n_feature], reg_const=1.5)
#%%
# Testing the gradient algorithm with regularization against the simple approximate method

# First a subset of the data is selected - to make the computation faster
n_user = 15
n_movie = 10
n_feature = 5

reg_const = 2

approx_grad_v = calc_J_grad_approx(r_m[:n_movie, :n_user], y_m[:n_movie, :n_user], x_ex_m[:n_movie, :n_feature], theta_ex_m[:n_user, :n_feature], reg_const=reg_const, eps=1E-5)

exact_grad_v = cf_J_grad(r_m[:n_movie, :n_user], y_m[:n_movie, :n_user], x_ex_m[:n_movie, :n_feature], theta_ex_m[:n_user, :n_feature], reg_const=reg_const)

print('Test of the algorithm for calculation of the gradient. Maximum |difference between the approximate gradient and the exact gradient| is: ' + str(np.max(np.abs(exact_grad_v - approx_grad_v))))
#%%

# Load the list of movies

os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex8\ex8')

file_obj = open('movie_ids.txt', mode='r')

movie_id_s = file_obj.readlines()

file_obj.close()

movie_id_s

reg_exp = re.compile('\s')

index_arr = np.zeros(len(movie_id_s), dtype=np.int)
movie_arr = np.zeros(len(movie_id_s), dtype=np.object)
for i in range(index_arr.shape[0]):
    split_list = reg_exp.split(movie_id_s[i], maxsplit=1)
    split_list[1] = split_list[1].strip()
    index_arr[i] = split_list[0]
    movie_arr[i] = split_list[1]

movie_s = pd.Series(data=movie_arr, index=index_arr)

#%%
my_rating_arr = np.zeros((y_m.shape[0], 1))

# Populate my ratings with the same ratings as in the execrise
my_rating_arr[1-1] = 4
my_rating_arr[98-1] = 2
my_rating_arr[7-1] = 3
my_rating_arr[12-1] = 5
my_rating_arr[54-1] = 4
my_rating_arr[64-1] = 5
my_rating_arr[66-1] = 3
my_rating_arr[69-1] = 5
my_rating_arr[183-1] = 4
my_rating_arr[226-1] = 5
my_rating_arr[355-1] = 5

my_rating_df = pd.DataFrame(data=np.array([movie_s.values, my_rating_arr[:,0]]).T, index=movie_s.index, columns=['Movie', 'Rating'])

# Movies rated by me
my_rating_df[my_rating_df['Rating']>0]
#%%
# Adding my ratings to the ratings matrix

y_new_m = np.hstack((y_m, my_rating_arr))
r_new_m = np.hstack((r_m, my_rating_arr!=0))

n_movie = y_new_m.shape[0]
n_user = y_new_m.shape[1]
n_feature = 10

# Randomly initialize matrix of features between 0 and 1
x_0_m = np.random.rand(n_movie, n_feature)
theta_0_m = np.random.rand(n_user, n_feature)

param_0_v = np.concatenate((x_0_m, theta_0_m)).flatten()

# Normalize the ratings
y_mean_arr = np.zeros((n_movie, 1))

for i in range(n_movie):
    y_mean_arr[i, 0] = np.mean(y_new_m[i, r_new_m[i] != 0])

y_norm_m = y_new_m - y_mean_arr

# Redefining the cost and gradient functions for the minimization algorithm

def cf_J(param_v, n_feature, r_m, y_m, reg_const=0):
    ''' Collaborative filtering cost function (no regularization)
    '''
    param_m = param_v.reshape(int(param_v.shape[0]/n_feature), n_feature)
    x_m = param_m[:y_m.shape[0]]
    theta_m = param_m[y_m.shape[0]:]

    c_m = (x_m @ theta_m.T - y_m)**2
    return 1/2 * np.sum(c_m*r_m) + 1/2 * reg_const * (np.sum(theta_m**2)+np.sum(x_m**2))


def cf_J_grad(param_v, n_feature, r_m, y_m, reg_const=0):
    ''' Compute the gradient of the collaborative cost function.

    Partial derivates with respect to x_m and theta_m are calculated. The final Jacobian is returned in a flattened form with the partial derivates for x_m being first in the vector.
    '''
    param_m = param_v.reshape(int(param_v.shape[0]/n_feature), n_feature)

    x_m = param_m[:y_m.shape[0]]
    theta_m = param_m[y_m.shape[0]:]

    # The gradient of the cost function is the following:
    # For the kth component of that for the jth user:
    # dJ/d theta^(j)_k = sum_{i (r(i,j)=1)} (theta^(j) x^(i)^T - y^(i,j))*x^(i)_k
    jac_J_theta_m = np.zeros(theta_m.shape)

    for j in range(jac_J_theta_m.shape[0]):
        theta_v = theta_m[j]

        idx_v = np.argwhere(r_m[:, j]==1).T[0]

        x_sel_m = x_m[idx_v]
        y_sel_m = y_m[idx_v, j]

        jac_J_theta_m[j] = (theta_v @ x_sel_m.T - y_sel_m) @ x_sel_m + reg_const * theta_v

    jac_J_x_m = np.zeros(x_m.shape)

    for i in range(jac_J_x_m.shape[0]):
        x_v = x_m[i]

        idx_v = np.argwhere(r_m[i]==1).T[0]
        theta_sel_m = theta_m[idx_v]
        y_sel_m = y_m[i, idx_v]

        jac_J_x_m[i] = (theta_sel_m @ x_v.T - y_sel_m) @ theta_sel_m + reg_const * x_v

    jac_v = np.append(jac_J_x_m.flatten(),jac_J_theta_m.flatten())
    return jac_v
#%%
# Optimize the features and the weights
reg_const = 10

param_opt = scipy.optimize.minimize(fun=cf_J, x0=param_0_v, args=(n_feature, r_new_m, y_norm_m, reg_const), method='L-BFGS-B', jac=cf_J_grad, options={'disp': None, 'maxiter': 2000})
#%%
# Calculate the recommended movies for me
param_opt_v = param_opt['x']
param_opt_m = param_opt_v.reshape(int(param_opt_v.shape[0]/n_feature), n_feature)

x_opt_m = param_opt_m[:y_new_m.shape[0]]
theta_opt_m = param_opt_m[y_new_m.shape[0]:]

rank_arr = theta_opt_m[-1] @ x_opt_m.T + y_mean_arr[:, 0]

my_rating_opt_df = pd.DataFrame(data=np.array([movie_s.values, rank_arr]).T, index=movie_s.index, columns=['Movie', 'Rating'])
#%%
# First 10 most recommended movies
my_rating_opt_df.sort_values(by='Rating', ascending=False).iloc[0:10]
#%%
my_rating_df[my_rating_df['Rating']>0]
#%%
my_rating_opt_df.loc[my_rating_df[my_rating_df['Rating']>0].index]

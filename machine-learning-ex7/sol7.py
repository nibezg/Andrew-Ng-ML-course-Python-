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
#%%
# First part of the exercise. Developing the K-means algorithm

def assignCentroids(x_m, centroid_m):
    dist_m = np.zeros((x_m.shape[0], centroid_m.shape[0]))

    for i in range(centroid_m.shape[0]):
        dist_m[:, i] = np.linalg.norm(x_m - centroid_m[i], axis=1)

    c_vec = np.argmin(dist_m, axis=1)

    return c_vec

def shiftCentroids(x_m, centroid_m, c_vec):
    shifted_centroid_m = centroid_m.copy()
    for i in range(centroid_m.shape[0]):
        shifted_centroid_m[i] = np.mean(x_m[c_vec == i], axis=0)
    return shifted_centroid_m
#%%
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex7\ex7')
x_m = scipy.io.loadmat('ex7data2.mat')['X']
#%%
# Initialize the centroids according to the preselected values
centroid_m = np.array([[3, 3], [6, 2], [8, 5]], dtype=np.float64)
# Or permute the order of the training set and select the first k values as the initial positions of the centroids

# Number of the centroids to use
n_centroids = 3
centroid_m = np.random.permutation(x_m)[:n_centroids]
#%%
# Cluster the data by performing the K-means algorithm
def k_means(x_m, centroid_m, num_iter):

    centroid_iter_arr = np.zeros((num_iter+1, centroid_m.shape[0], centroid_m.shape[1]))

    centroid_iter_arr[0] = centroid_m

    for i in range(num_iter):
        c_vec = assignCentroids(x_m, centroid_m)
        centroid_m = shiftCentroids(x_m, centroid_m, c_vec)
        centroid_iter_arr[i+1] = centroid_m

    return c_vec, centroid_iter_arr
#%%
num_iter = 10

c_vec, centroid_iter_arr = k_means(x_m, centroid_m, num_iter)
#%%
# Visualization of the K-means algorithm

def k_mean_visualize(c_vec, centroid_iter_arr, x_m, ax):
    c_vec = c_vec.astype(np.int16)
    exp1_data_df = pd.DataFrame(np.column_stack((x_m, c_vec)), columns=['x0', 'x1', 'c'])

    sns.scatterplot(x='x0', y='x1', hue='c', data=exp1_data_df, ax=ax, legend=False)


    ax.scatter(centroid_iter_arr[0,:,0], centroid_iter_arr[0,:,1], color='black')
    ax.scatter(centroid_iter_arr[-1,:,0], centroid_iter_arr[-1,:,1], color='black')
    for i in range(1, centroid_iter_arr.shape[0]-1):
        for j in range(centroid_iter_arr[i].shape[0]):
            ax.plot([centroid_iter_arr[i-1, j, 0], centroid_iter_arr[i, j, 0]],  [centroid_iter_arr[i-1, j, 1], centroid_iter_arr[i, j, 1]], color='black')

            ax.scatter(centroid_iter_arr[i, j, 0], centroid_iter_arr[i, j, 1], color='black')
    return ax

fig, ax = plt.subplots()
fig.set_size_inches(12, 10)
ax = k_mean_visualize(c_vec, centroid_iter_arr, x_m, ax)
plt.show()
#%%
# Second part. Image compression. We want to select 16 different clusters from an image

# Python library for working with images. I need it only to download the image and convert it to a numpy array
import PIL

# Load the image
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex7\ex7')
im = PIL.Image.open(r'bird_small.png')

# Convert the image to a matrix with 128 * 128 rows + 3 columns (for the 8-bit RGB colors)
x_m = np.array(im, dtype=np.float64).reshape(im.size[0]*im.size[0], 3)
#%%
# Number of the centroids to use
n_centroids = 16

centroid_m = np.random.permutation(x_m)[:n_centroids]
num_iter = 1000

c_vec, centroid_iter_arr = k_means(x_m, centroid_m, num_iter)

centroid_cluster_m = centroid_iter_arr[-1]
#%%
# Compress the image
x_compress_m = x_m.copy()

for i in range(centroid_cluster_m.shape[0]):
    x_compress_m[c_vec==i] = centroid_cluster_m[i]
#%%
# Original image
PIL.Image.fromarray(x_m.reshape(128, 128, 3).astype(dtype=np.uint8))
#%%
# Compressed image
PIL.Image.fromarray(x_compress_m.reshape(128, 128, 3).astype(dtype=np.uint8))
#%%
''' Principle component analysis
'''
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex7\ex7')
x_m = scipy.io.loadmat('ex7data1.mat')['X']
#%%
# Data visualization
fig, ax = plt.subplots()
exp2_data_df = pd.DataFrame(x_m, columns=['x0', 'x1'])
sns.scatterplot(x='x0', y='x1', data=exp2_data_df, ax=ax, legend=False)
plt.show()
#%%
# The exercise is assuming that we are scaling the data by its standard deviation (including ddof = N-1).
x_norm_m = (x_m - np.mean(x_m, axis=0)) / np.std(x_m, ddof=1, axis=0)

# Calculate the covariance matrix
cov_m = 1 / x_norm_m.shape[0] * x_norm_m.T @ x_norm_m

# Determine the eigenvectors of the covariance matrix (the eigenvectors are normalized)
u_m, s_m, v_m = scipy.linalg.svd(cov_m)
#%%
mean_data_v = np.mean(x_norm_m, axis=0)
# Visuazliation of the eigenvectors

fig, ax = plt.subplots()

# The figure is scaled appropriately to make the eigenvectors visible orthogonal to each other
fig.set_size_inches(8, 8 * np.std(x_m, ddof=1, axis=0)[0]/ np.std(x_m, ddof=1, axis=0)[1])

exp2_data_df = pd.DataFrame(x_norm_m, columns=['x0', 'x1'])
sns.scatterplot(x='x0', y='x1', data=exp2_data_df, ax=ax, legend=False)

# Plot the eigenvectors of the covariance matrix, scaled by the diagonal matrix S
ax.plot([mean_data_v[0], mean_data_v[0]+s_m[0] * u_m[0, 0]], [mean_data_v[1], mean_data_v[1] + s_m[0]*u_m[1, 0]], color='black')

ax.plot([mean_data_v[0], mean_data_v[0]+s_m[1] * u_m[0, 1]], [mean_data_v[1], mean_data_v[1]+s_m[1] * u_m[1, 1]], color='black')
plt.show()
#%%
# Data projection onto the first k eigenvectors
def projectData(x_m, u_m, k):
    return x_m @ u_m[:, :k]

# Recovering the data to
def recoverData(z_m, u_m, k):
    return z_m @ u_m[:, :k].T
#%%
z_m = projectData(x_norm_m, u_m, 1)
z_m[0]
#%%
x_rec_m = recoverData(z_m, u_m, 1)
x_rec_m[0]
#%%
# Visuazliation of the recovered data

fig, ax = plt.subplots()
fig.set_size_inches(8, 8 * np.std(x_m, ddof=1, axis=0)[0]/ np.std(x_m, ddof=1, axis=0)[1])
sns.scatterplot(x='x0', y='x1', data=exp2_data_df, ax=ax, legend=False)

exp2_data_2_df = pd.DataFrame(x_rec_m, columns=['x0', 'x1'])
sns.scatterplot(x='x0', y='x1', data=exp2_data_2_df, ax=ax, legend=False, color='red')

# Plot the eigenvectors of the covariance matrix, scaled by the diagonal matrix S
ax.plot([mean_data_v[0], mean_data_v[0]+s_m[0] * u_m[0, 0]], [mean_data_v[1], mean_data_v[1] + s_m[0]*u_m[1, 0]], color='black')

ax.plot([mean_data_v[0], mean_data_v[0]+s_m[1] * u_m[0, 1]], [mean_data_v[1], mean_data_v[1]+s_m[1] * u_m[1, 1]], color='black')
plt.show()
#%%
''' PCA on images
'''

os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex7\ex7')
x_m = scipy.io.loadmat('ex7faces.mat')['X']
#%%
# Visualize n_img random images
n_img = 100
ncols = int(np.sqrt(n_img))
nrows = int(np.sqrt(n_img))
img_index_arr = np.random.randint(low=0, high=x_m.shape[0], size=n_img).reshape(nrows, ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

fig.set_size_inches(7, 7)

for i in range(nrows):
    for j in range(ncols):
        axes[i, j].imshow(x_m[img_index_arr[i, j]].reshape(32, 32).T * 255.0, aspect='auto', cmap='gray')
        axes[i, j].set_xticklabels([])
        axes[i, j].set_yticklabels([])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
plt.show()
#%%
x_norm_m = (x_m - np.mean(x_m, axis=0))

# Calculate the covariance matrix
cov_m = 1 / x_norm_m.shape[0] * x_norm_m.T @ x_norm_m

# Determine the eigenvectors of the covariance matrix (the eigenvectors are normalized)
u_m, s_m, v_m = scipy.linalg.svd(cov_m)
#%%
# Visualize the first n_eigen eigenvectors
n_eigen = 36
ncols = int(np.sqrt(n_eigen))
nrows = int(np.sqrt(n_eigen))
#img_index_arr = np.random.randint(low=0, high=u_m.shape[1], size=n_img).reshape(nrows, ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

fig.set_size_inches(7, 7)

img_counter = 0
for i in range(nrows):
    for j in range(ncols):
        axes[i, j].imshow(u_m[:, img_counter].reshape(32, 32).T * 255.0, aspect='auto', cmap='gray')
        img_counter = img_counter + 1
        axes[i, j].set_xticklabels([])
        axes[i, j].set_yticklabels([])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
plt.show()
#%%
# Project the data onto the first k dimensions and then recover the data

k=100

z_m = projectData(x_norm_m, u_m, k=k)
z_m[0]

x_rec_m = recoverData(z_m, u_m, k=k) + np.mean(x_m, axis=0)

#%%
# Compare the original images with the images with only first k dimensions

# Visualize n_img random images
n_img = 100
ncols = int(np.sqrt(n_img))
nrows = int(np.sqrt(n_img))
img_index_arr = np.random.randint(low=0, high=x_m.shape[0], size=n_img).reshape(nrows, ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

fig.set_size_inches(9, 9)

for i in range(nrows):
    for j in range(ncols):
        axes[i, j].imshow(x_m[img_index_arr[i, j]].reshape(32, 32).T * 255.0, aspect='auto', cmap='gray')
        axes[i, j].set_xticklabels([])
        axes[i, j].set_yticklabels([])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.show()
#%%
# Visualization of the same images, but with only k dimensions
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

fig.set_size_inches(9, 9)

for i in range(nrows):
    for j in range(ncols):
        axes[i, j].imshow(x_rec_m[img_index_arr[i, j]].reshape(32, 32).T * 255.0, aspect='auto', cmap='gray')
        axes[i, j].set_xticklabels([])
        axes[i, j].set_yticklabels([])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.show()
#%%
''' PCA for Visualization. Projecting the 3D data onto a 2D plane.

Using the image of the parrot from a previous exercise.
'''
from mpl_toolkits.mplot3d import Axes3D

# Load the image
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex7\ex7')
im = PIL.Image.open(r'bird_small.png')

# Convert the image to a matrix with 128 * 128 rows + 3 columns (for the 8-bit RGB colors)
x_m = np.array(im, dtype=np.float64).reshape(im.size[0]*im.size[0], 3)

#%%
# Visualize the clusters in 3D

# Selecting only n_select points for plotting
n_sel = 1000

index_rand_vec = np.random.randint(1, x_m.shape[0], n_sel)
x_sel_m = x_m[index_rand_vec]

fig = plt.figure()
fig.set_size_inches(10, 8)
ax = fig.add_subplot(111, projection='3d')

c_sel_vec = c_vec[index_rand_vec]

for i in range(c_sel_vec.shape[0]):
    x_c_m = x_sel_m[c_sel_vec == i]
    ax.scatter(x_c_m[:, 0], x_c_m[:, 1], x_c_m[:, 2], s=2)

ax.view_init(elev=20., azim=250)

plt.show()
#%%
# We now use PCA to project the data onto the first 2 eigenvectors of the covariance matrix
x_norm_m = (x_m - np.mean(x_m, axis=0)) / np.std(x_m, ddof=1, axis=0)

# Calculate the covariance matrix
cov_m = 1 / x_norm_m.shape[0] * x_norm_m.T @ x_norm_m

# Determine the eigenvectors of the covariance matrix (the eigenvectors are normalized)
u_m, s_m, v_m = scipy.linalg.svd(cov_m)

z_m = projectData(x_norm_m, u_m, k=2)
#%%
# Selecting only n_select points for plotting
n_sel = 10000

index_rand_vec = np.random.randint(1, x_m.shape[0], n_sel)
z_sel_m = z_m[index_rand_vec]
c_sel_vec = c_vec[index_rand_vec]

fig, ax = plt.subplots()

c_sel_vec = c_sel_vec.astype(np.int16)
exp3_data_df = pd.DataFrame(np.column_stack((z_sel_m, c_sel_vec)), columns=['z0', 'z1', 'c'])

sns.scatterplot(x='z0', y='z1', hue='c', data=exp3_data_df, ax=ax, legend=False)

plt.show()
#%%

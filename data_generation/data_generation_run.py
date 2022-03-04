#!/usr/bin/env python3


import os
import numpy as np
from scipy.ndimage import gaussian_filter

import data_generation as dtgn


data_folder_s = '../results/data_smoothed/'
if not os.path.exists(data_folder_s):
    os.makedirs(data_folder_s)
    
data_folder_z = '../results/data_zeros/'
if not os.path.exists(data_folder_z):
    os.makedirs(data_folder_z)

N = 30 # width and length dimension of the RGB image
N_half = np.int((N - 1) / 2)
ratios = [0.7, 0.15, 0.15] # sizes of the objects
colors = [[0.8, 0.8, 0.8], [0.8, 0.2, 0.2], [0.2, 0.2, 0.8]] # colors of the objects, in RGB
bound_val = 0.5 # position bound of the smaller objects
backgrounds = ['smoothed', 'zeros'] # background of the images, either 'zeros' or 'smoothed'.

nx, ny = (N, N)
x = np.linspace(0, N-1, N)
y = np.linspace(0, N-1, N)
x_coord, y_coord = np.meshgrid(x, y)
distance_matrix = np.sqrt((x_coord - N_half) ** 2 + (y_coord - N_half) ** 2) + 1e-32
angle_matrix = np.arccos((x_coord - N_half) / distance_matrix)
angle_matrix[N_half, N_half] = 0
angle_matrix[N_half+1:, :] = 2 * np.pi - angle_matrix[N_half+1:, :]


M = 25 # number of meaningful or random images

background = 'smoothed'
y_label_p = []
for m in range(M):
    img, masks = dtgn.get_an_image(N, ratios, distance_matrix, angle_matrix, bound_val, colors, background, H=1)
    np.save(data_folder_s + f'img_{int(2*m)}', img)
    np.save(data_folder_s + f'masks_{int(2*m)}', masks)
    y_label_p.append([1])
np.save(data_folder_s + 'y_label_p', y_label_p)

background = 'zeros'
y_label_p = []
for m in range(M):
    img, masks = dtgn.get_an_image(N, ratios, distance_matrix, angle_matrix, bound_val, colors, background, H=1)
    np.save(data_folder_z + f'img_{int(2*m)}', img)
    np.save(data_folder_z + f'masks_{int(2*m)}', masks)
    y_label_p.append([1])
np.save(data_folder_z + 'y_label_p', y_label_p)






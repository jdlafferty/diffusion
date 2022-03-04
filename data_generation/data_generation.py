#!/usr/bin/env python3


import numpy as np
from scipy.ndimage import gaussian_filter


def get_a_mask(N, ratio, distance_matrix, angle_matrix):
    N_half = np.int((N - 1) / 2)
    r = N_half * ratio
    H = 10;
    rho = np.random.random(H) * np.logspace(-0.5, -2.5, H)
    phi = np.random.random(H) * 2 * np.pi

    # Accumulate r(t) over t=[0, 2*pi]
    HH = 101
    t = np.linspace(0, 2*np.pi, HH)
    r = np.ones(t.shape) * N_half * ratio

    for h in range(H):
      r = r + rho[h] * np.sin(h*t+phi[h]) * N_half * ratio
    
    mask = np.zeros((N, N))
    for n1 in range(N):
        for n2 in range(N):
            angle_ind = np.int(angle_matrix[n1, n2] / (2 * np. pi) * HH)
            if distance_matrix[n1, n2] < r[angle_ind]:
                mask[n1, n2] = 1
                
    return mask.astype(bool)

def get_uniform_any_interval(small_val, large_val, H):
    random_val = small_val + (large_val - small_val) * np.random.random(H) 
    
    return random_val


def get_an_image(N, ratios, distance_matrix, angle_matrix, bound_val, colors, background, H=1):
    """
    Generate a sample RGB image.
    ___________
    Args:
    N - the dimension of the image 
    ratios - a list of the sizes of the objects of interest in the image
    distance_matrix - distance matrix, with the same dimension of each color channel
    angle_matrix - angle matrix, with the same dimension of each color channel
    bound_val - value for the upper limit of the distance from the center
    background - background pattern, either 'zeros' or 'smoothed'
    colors - a list of the colors of the object, each element is a 3-tuple representing the RGB values.
             len(colors) = len(ratios)
    H - number of random points for sampling in the uniform interval [-bound_val, bound_val], should be 1.
    
    Returns:
    img - one sample RGB image with the shape of (N, N, 3)
    masks - a list of masks, the first one is for the face
    """
    # set the background of the image
    if background == 'zeros':
        img = np.zeros((N, N, 3))
    elif background == 'smoothed':
        img = np.random.random((N, N, 3))
        for c in range(3):
            img[:, :, c] = gaussian_filter(img[:, :, c], 1)
            
    masks = []

    # assign the color to the face
    mask_face = get_a_mask(N, ratios[0], distance_matrix, angle_matrix)
    img[mask_face, 0] = colors[0][0]
    img[mask_face, 1] = colors[0][1]
    img[mask_face, 2] = colors[0][2]
    
    masks.append(mask_face)
    
    # assign the colors to the eyes
    N_half = np.int((N - 1) / 2)
    for ii in range(len(ratios)-1):
        mask_eye = get_a_mask(N, ratios[ii+1], distance_matrix, angle_matrix)
        x_random = np.int(get_uniform_any_interval(-bound_val, bound_val, 1) * N_half)
        y_random = np.int(get_uniform_any_interval(-bound_val, bound_val, 1) * N_half)
        mask_eye = np.roll(mask_eye, (x_random, y_random), (0, 1))
        img[mask_eye, 0] = colors[ii+1][0]
        img[mask_eye, 1] = colors[ii+1][1]
        img[mask_eye, 2] = colors[ii+1][2]
        
        masks.append(mask_eye)

    return img, masks






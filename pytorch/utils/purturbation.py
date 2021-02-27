# Copyright (c) ASU GitHub Project.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

from __future__ import print_function
import math
import os
import random
import copy
import scipy
import string
import numpy as np
import torch
import torch.utils.data
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

from PIL import Image, ImageDraw, ImageFont


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y


def elastic_transform(image):
    alpha = 991
    sigma = 8
    random_state = np.random.RandomState(None)
    image = image[0]
    shape_mrht = np.shape(image)

    dx = gaussian_filter((random_state.rand(*shape_mrht) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape_mrht) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape_mrht[0]), np.arange(shape_mrht[1]), np.arange(shape_mrht[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    transformed_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape_mrht)
    transformed_image = transformed_image[np.newaxis, :, :, :]
    return transformed_image


def nonlinear_transformation(x):
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half chance to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def local_pixel_shuffling(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 5
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        block_noise_size_z = random.randint(1, img_deps // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        noise_z = random.randint(0, img_deps - block_noise_size_z)
        window = orig_image[0, noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y,
                 noise_z:noise_z + block_noise_size_z,
                 ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x,
                                 block_noise_size_y,
                                 block_noise_size_z))
        image_temp[0, noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x


def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
    block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
    block_noise_size_z = random.randint(img_deps // 6, img_deps // 3)
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
    x[:, noise_x:noise_x + block_noise_size_x,
    noise_y:noise_y + block_noise_size_y,
    noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                           block_noise_size_y,
                                                           block_noise_size_z, ) * 1.0
    return x


def image_out_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) * 1.0
    block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
    block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
    block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
    x[:, noise_x:noise_x + block_noise_size_x,
    noise_y:noise_y + block_noise_size_y,
    noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                            noise_y:noise_y + block_noise_size_y,
                                            noise_z:noise_z + block_noise_size_z]
    return x




# Copyright (c) ASU GitHub Project.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

from scipy.misc import comb
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import numpy as np
import copy
import random

def elastic_transform(image):
    alpha = 991
    sigma = 8
    random_state = np.random.RandomState(None)
    shape_mrht = np.shape(image)

    dx = gaussian_filter((random_state.rand(*shape_mrht) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape_mrht) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape_mrht[0]), np.arange(shape_mrht[1]), np.arange(shape_mrht[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    transformed_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape_mrht)
    return transformed_image


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


def local_pixel_shuffling(x):
    shuffled_image = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols, img_deps = x.shape
    num_block = 5
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        block_noise_size_z = random.randint(1, img_deps // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        noise_z = random.randint(0, img_deps - block_noise_size_z)
        window = orig_image[noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y,
                 noise_z:noise_z + block_noise_size_z,
                 ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x,
                                 block_noise_size_y,
                                 block_noise_size_z))
        shuffled_image[noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = window

    return shuffled_image

def non_linear_transformation(x):
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)

    return nonlinear_x

def image_in_painting(x):
    img_rows, img_cols, img_deps = x.shape
    block_noise_size_x, block_noise_size_y, block_noise_size_z = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
    noise_x, noise_y, noise_z = random.randint(3, img_rows - block_noise_size_x - 3), random.randint(3, img_cols - block_noise_size_y - 3), random.randint(3, img_deps - block_noise_size_z - 3)
    x[noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y,noise_z:noise_z + block_noise_size_z] = random.random()

    return x


def image_out_painting(x):
    img_rows, img_cols, img_deps = x.shape
    block_noise_size_x, block_noise_size_y, block_noise_size_z = img_rows - random.randint(10, 20), img_cols - random.randint(10, 20), img_deps - random.randint(10, 20)
    noise_x, noise_y, noise_z = random.randint(3, img_rows - block_noise_size_x - 3), random.randint(3, img_cols - block_noise_size_y - 3), random.randint(3, img_deps - block_noise_size_z - 3)
    image_temp = copy.deepcopy(x)
    x[:, :, :] = random.random()
    x[noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y,
    noise_z:noise_z + block_noise_size_z] = image_temp[noise_x:noise_x + block_noise_size_x,
                                            noise_y:noise_y + block_noise_size_y,
                                            noise_z:noise_z + block_noise_size_z]

    return x


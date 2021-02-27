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
from utils.purturbation import *

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

from PIL import Image, ImageDraw, ImageFont


class SemanticGenesis_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, config):
        self.data = data
        self.label = label
        self.augment = config.data_augmentation
        self.exp_choice = config.exp_choice

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        return self.luna_loader(self.data[index], self.label[index])

    def luna_loader(self, data, label):
        y_img = data
        y_cls = np.squeeze(label)
        x_ori = copy.deepcopy(y_img)
        y_trans = ''

        if self.augment:
            y_img = elastic_transform(y_img)
        r = random.random()
        if r <= 0.25:
            x = local_pixel_shuffling(y_img)
            y_trans = 'local-shuffle'
        elif 0.25 < r <= 0.5:
            x = nonlinear_transformation(y_img)
            y_trans = 'non-linear'
        elif 0.5 < r <= 0.75:
            x = image_in_painting(y_img)
            y_trans = 'in-paint'
        else:
            x = image_out_painting(y_img)
            y_trans = 'out-paint'

        if self.exp_choice == 'en':
            return x, y_cls
        else:
            return x, [x_ori, y_cls, y_trans]


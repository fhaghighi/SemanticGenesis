# Copyright (c) ASU GitHub Project.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################


import warnings

warnings.filterwarnings('ignore')
import os
import keras

print("Keras = {}".format(keras.__version__))
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import pylab
import sys
import math
import SimpleITK as sitk

from matplotlib import offsetbox
import matplotlib.pyplot as plt
import copy
import shutil
from sklearn import metrics

import random
from sklearn.utils import shuffle
from vnet3d import *
from unet3d import *
from keras.callbacks import LambdaCallback, TensorBoard
from glob import glob
from skimage.transform import resize
from optparse import OptionParser

from datetime import datetime
sys.setrecursionlimit(40000)


parser = OptionParser()

parser.add_option("--arch", dest="arch", help="Vnet|Unet", default="Unet", type="string")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=128, type=int)
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=128, type=int)
parser.add_option("--input_depth", dest="input_depth", help="input depth", default=64, type=int)
parser.add_option("--verbose", dest="verbose", help="verbose", default=1, type=int)
parser.add_option("--weights", dest="weights", help="pre-trained weights", default=None, type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=8, type=int)
parser.add_option("--data_dir", dest="data_dir",help=" path to images", default=None)


(options, args) = parser.parse_args()

assert options.data_dir is not None
assert options.weights is not None

input_rows = options.input_rows
input_cols = options.input_cols
input_depth = options.input_depth


if options.arch =="Vnet":
    model = vnet_model_3d((1, options.input_rows, options.input_cols, options.input_depth), batch_normalization=True)
elif options.arch =="Unet":
    model = unet_model_3d((1, options.input_rows, options.input_cols, options.input_depth), batch_normalization=True)

model.load_weights(options.weights)
model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False,clipnorm=1),
              loss="MSE",
              metrics=["MAE", "MSE"])

x=model.get_layer('depth_7_relu').output
x = keras.layers.GlobalAveragePooling3D()(x)
encoder_model = keras.models.Model(inputs=model.input, outputs=x)
encoder_model.summary()


train_images_list=glob(os.path.join(options.data_dir, "train", "*"))
train_images_list.sort()


train_features=np.zeros((len(train_images_list), 512), dtype=np.float32)
count=0
for image in train_images_list:
    x = np.zeros((1, 1, input_rows, input_cols,input_depth), dtype="float")
    img = np.load(image)
    img = resize(img, (input_rows, input_cols, input_depth), preserve_range=True)
    img = np.expand_dims(img, axis=0)
    x[0,:,:,:,:]=img
    feature=encoder_model.predict(x)
    train_features[count, :] = feature
    count +=1
    print(count)

np.save("train_features", train_features)
print("train features has been saved.")



val_images_list=glob(os.path.join(options.data_dir, "validation", "*"))
val_images_list.sort()


val_features=np.zeros((len(val_images_list), 512), dtype=np.float32)
count=0
for image in val_images_list:
    x = np.zeros((1, 1, input_rows, input_cols,input_depth), dtype="float")
    img = np.load(image)
    img = resize(img, (input_rows, input_cols, input_depth), preserve_range=True)
    img = np.expand_dims(img, axis=0)
    x[0,:,:,:,:]=img
    feature=encoder_model.predict(x)
    val_features[count, :] = feature
    count +=1
    print(count)

np.save("validation_features", val_features)
print("validation features has been saved.")


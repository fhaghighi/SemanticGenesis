
# Copyright (c) ASU GitHub Project.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

from __future__ import print_function

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
from scipy.misc import comb

from matplotlib import offsetbox
import matplotlib.pyplot as plt
import copy
import shutil
from sklearn import metrics

import random
from sklearn.utils import shuffle
from vnet3d import *
from keras.callbacks import LambdaCallback, TensorBoard, ReduceLROnPlateau
from glob import glob
from skimage.transform import resize
from optparse import OptionParser

from datetime import datetime
sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--arch", dest="arch", help="Vnet", default="Vnet", type="string")
parser.add_option("--decoder", dest="decoder_block_type", help="transpose | upsampling", default="upsampling",
                  type="string")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=128, type=int)
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=128, type=int)
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=64, type=int)
parser.add_option("--verbose", dest="verbose", help="verbose", default=1, type=int)
parser.add_option("--weights", dest="weights", help="pre-trained weights", default=None, type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=8, type=int)
parser.add_option("--data_dir", dest="data_dir",help="path to data", default=None)


(options, args) = parser.parse_args()

assert options.data_dir is not None


seed = 1
random.seed(seed)
model_path = "Checkpoints/Autoencoder/"
if not os.path.exists(model_path):
    os.makedirs(model_path)



def date_str():
	return datetime.now().__str__().replace("-", "_").replace(" ", "_").replace(":", "_")

class setup_config():
    nb_epoch = 10000
    patience = 50
    lr = 1e-3

    def __init__(self, model="Vnet",
                 backbone="",
                 data_augmentation=True,
                 input_rows=128,
                 input_cols=128,
                 input_deps=64,
                 batch_size=64,
                 decoder_block_type=None,
                 nb_class=1,
                 verbose=1,
                 ):
        self.model = model
        self.backbone = backbone
        self.exp_name = model + "_autoencoder"
        self.input_rows, self.input_cols = input_rows, input_cols
        self.input_deps = input_deps
        self.batch_size = batch_size
        self.verbose = verbose
        self.decoder_block_type = decoder_block_type
        self.nb_class = nb_class
        if nb_class > 1:
            self.activation = "softmax"
        else:
            self.activation = "sigmoid"

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


class DataGenerator(keras.utils.Sequence):
    def __init__(self, directory, batch_size=16, dim=(128, 128, 64)):
        self.directory = directory
        self.images_paths = self.get_list_of_images(directory)
        self.batch_size = batch_size
        self.dim = dim
        self.length = len(self.images_paths)

    def __len__(self):
        return int(np.floor(len(self.images_paths)  / self.batch_size))

    def __getitem__(self, index):

        file_names = self.images_paths[index * batch_size:(index + 1) * batch_size]
        return self.data_loader(file_names)

    def on_epoch_end(self):
        np.random.shuffle(self.images_paths)

    def get_list_of_images(self, path):
        try:
            images=glob(os.path.join(path, "*"))
            return images
        except FileNotFoundError:
            print("Wrong file or file path")

    def data_loader(self, file_list):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        input_depth=self.dim[2]
        x = np.zeros((self.batch_size,1,input_rows, input_cols, input_depth), dtype="float")
        y = np.zeros((self.batch_size,1,input_rows, input_cols, input_depth), dtype="float")

        count = 0
        for i, file in enumerate(file_list):
            img=np.load(file)
            img=resize(img, (input_rows, input_cols,input_depth), preserve_range=True)
            img=np.expand_dims(img, axis=0)
            x[count, :, :, :,:] = img
            y[count, :, :, :,:] = img
            count += 1

        x, y = shuffle(x, y, random_state=0)
        return x, y




config = setup_config(model=options.arch,
                      decoder_block_type=options.decoder_block_type,
                      input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      batch_size=options.batch_size,
                      verbose=options.verbose,

                      )
config.display()



if options.arch =="Vnet":
    model = vnet_model_3d((1, config.input_rows, config.input_cols, config.input_deps), batch_normalization=True)


if options.weights is not None:
    print("Load the pre-trained weights from {}".format(options.weights))
    model.load_weights(options.weights)
model.compile(optimizer=keras.optimizers.Adam(lr=config.lr),
              loss="MSE",
              metrics=["MAE", "MSE"])
model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=config.patience,
                                               verbose=0,
                                               mode='min',
                                               )
check_point = keras.callbacks.ModelCheckpoint(os.path.join(model_path, config.exp_name + ".h5"),
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min',
                                              )

lrate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20,
                                    min_delta=0.0001, min_lr=1e-6, verbose=1)

callbacks = [check_point, early_stopping, lrate_scheduler]


training_generator = DataGenerator(os.path.join(options.data_dir,'train/'),
                                       batch_size=batch_size,dim=(config.input_rows,config.input_cols,config.input_deps))
validation_generator = DataGenerator(os.path.join(options.data_dir,'validation/'),
                                         batch_size=batch_size,dim=(config.input_rows,config.input_cols,config.input_deps))

model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=training_generator.length  // batch_size,
                            validation_steps=validation_generator.length  // batch_size,
                            epochs=config.nb_epoch,
                            max_queue_size=20,
                            workers=7,
                            use_multiprocessing=True,
                            shuffle=True,
                            verbose=config.verbose,
                            callbacks=callbacks,
                            )

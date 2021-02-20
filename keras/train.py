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
import sys
import math
import shutil
import random
from models.ynet3d import *
from keras.callbacks import LambdaCallback, TensorBoard, ReduceLROnPlateau
from optparse import OptionParser
from config import setup_config
from data_loader import DataGenerator
import numpy as np
sys.setrecursionlimit(40000)




parser = OptionParser()


parser.add_option("--decoder", dest="decoder_block_type", help="transpose | upsampling", default="upsampling",
                  type="string")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type=int)
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type=int)
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type=int)
parser.add_option("--verbose", dest="verbose", help="verbose", default=1, type=int)
parser.add_option("--weights", dest="weights", help="pre-trained weights path for resuming of training", default=None, type="string")
parser.add_option("--unet_weights", dest="unet_weights", help="pre-trained unet weights", default=None, type="string")
parser.add_option("--encoder_weights", dest="encoder_weights", help="pre-trained encoder weights", default=None, type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=8, type=int)
parser.add_option("--cls_classes", dest="cls_classes", help="number of classes", default=44,type=int)
parser.add_option("--nb_instances", dest="nb_instances", help="nubmer of samples in each class",type=int, default=200)
parser.add_option("--nb_multires_patch", dest="nb_multires_patch",help="number of multi resolution cubes", type=int, default=3)
parser.add_option("--lambda_rec", dest="lambda_rec",help="reconstruction loss weight", type=int, default=1)
parser.add_option("--lambda_cls", dest="lambda_cls",help="classification loss weight", type=int, default=0.01)
parser.add_option("--data_dir", dest="data_dir",help="data path", default="./Semantic_Genesis_data")


(options, args) = parser.parse_args()

assert options.decoder_block_type in ['transpose',
                                      'upsampling'
                                      ]
assert options.data_dir is not None

seed = 1
random.seed(seed)

config = setup_config(decoder_block_type=options.decoder_block_type,
                      input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      batch_size=options.batch_size,
                      verbose=options.verbose,
                      cls_classes=options.cls_classes,
                      nb_instances=options.nb_instances,
                      nb_multires_patch=options.nb_multires_patch,
                      weights=options.weights,
                      unet_weights=options.unet_weights,
                      encoder_weights=options.encoder_weights,
                      lambda_rec=options.lambda_rec,
                      lambda_cls=options.lambda_cls,
                      DATA_DIR=options.data_dir)


config.display()

model = ynet_model_3d((1, config.input_rows, config.input_cols, config.input_deps), batch_normalization=True,unet_weights=config.unet_weights,encoder_weights=config.encoder_weights,cls_classes=config.cls_classes)

if config.weights is not None:
    print("Load the pre-trained weights from {}".format(config.weights))
    model.load_weights(config.weights)

model.compile(optimizer=keras.optimizers.Adam(lr=config.lr),
              loss={'reconst_output': 'mse', 'cls_output': 'categorical_crossentropy'},
              loss_weights={'reconst_output': config.lambda_rec , 'cls_output': config.lambda_cls},
              metrics={'reconst_output': ['mse', 'mae'], 'cls_output': ['categorical_crossentropy', 'accuracy']})


model.summary()

x_train = []
y_train=[]
for i in range (int(math.ceil(config.cls_classes/50))):
    print("data part:",i)
    for fold in range (config.nb_multires_patch):
        print("fold:",fold)
        s = np.load(os.path.join(config.DATA_DIR, "train_data"+str(fold+1)+"_vwGen_ex_ref_fold"+str(i+1)+".0.npy"))
        l = np.load(os.path.join(config.DATA_DIR, "train_label"+str(fold+1)+"_vwGen_ex_ref_fold"+str(i+1)+".0.npy"))
        if (i==int(math.ceil(config.cls_classes/50))-1) and config.cls_classes % 50 != 0:
            print("select subset of data")
            index=config.cls_classes - i * 50
            s=s[0:config.nb_instances*index,:]
            l=l[0:config.nb_instances*index,:]
        x_train.extend(s)
        y_train.extend(l)
        del s
x_train=np.array(x_train)
y_train=np.array(y_train)


print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print("y_train: {} | {:.2f} ~ {:.2f}".format(y_train.shape, np.min(y_train), np.max(y_train)))

x_valid = []
y_valid=[]
for i in range (int(math.ceil(config.cls_classes/50))):
    print("data part:",i)
    s = np.load(os.path.join(config.DATA_DIR, "val_data1_vwGen_ex_ref_fold"+str(i+1)+".0.npy"))
    l = np.load(os.path.join(config.DATA_DIR, "val_label1_vwGen_ex_ref_fold"+str(i+1)+".0.npy"))
    if (i == int(math.ceil(config.cls_classes / 50)) - 1) and config.cls_classes % 50 != 0:
        print("select subset of data")
        index = config.cls_classes - i * 50
        s = s[0:30 * index, :]
        l = l[0:30 * index, :]
    x_valid.extend(s)
    y_valid.extend(l)
    del s
x_valid=np.array(x_valid)
y_valid=np.array(y_valid)
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))
print("y_valid: {} | {:.2f} ~ {:.2f}".format(y_valid.shape, np.min(y_valid), np.max(y_valid)))



training_generator = DataGenerator(x_train,y_train,batch_size=config.batch_size,dim=(config.input_rows,config.input_cols,config.input_deps),nb_classes=config.cls_classes)
validation_generator = DataGenerator(x_valid,y_valid,
                        batch_size=config.batch_size,dim=(config.input_rows,config.input_cols,config.input_deps),nb_classes=config.cls_classes)

if os.path.exists(os.path.join(config.model_path, config.exp_name+".txt")):
    os.remove(os.path.join(config.model_path, config.exp_name+".txt"))
with open(os.path.join(config.model_path, config.exp_name+".txt"),'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

shutil.rmtree(os.path.join(config.logs_path, config.exp_name), ignore_errors=True)
if not os.path.exists(os.path.join(config.logs_path, config.exp_name)):
    os.makedirs(os.path.join(config.logs_path, config.exp_name))




tbCallBack = TensorBoard(log_dir=os.path.join(config.logs_path, config.exp_name),
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True,
                        )
tbCallBack.set_model(model)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=config.patience,
                                               verbose=0,
                                               mode='min',
                                               )
check_point = keras.callbacks.ModelCheckpoint(os.path.join(config.model_path, config.exp_name + ".h5"),
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min',
                                              )
lrate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20,
                                    min_delta=0.0001, min_lr=1e-6, verbose=1)

callbacks = [check_point, early_stopping,tbCallBack, lrate_scheduler]


while config.batch_size > 1:
    # To find a largest batch size that can be fit into GPU
    try:
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=training_generator.length // config.batch_size,
                            validation_steps=validation_generator.length // config. batch_size,
                            epochs=config.nb_epoch,
                            max_queue_size=20,
                            workers=7,
                            use_multiprocessing=True,
                            shuffle=False,
                            verbose=config.verbose,
                            callbacks=callbacks,
                            )
        break
    except tf.errors.ResourceExhaustedError as e:
        config.batch_size = int(config.batch_size / 2.0)
        print("\n> Batch size = {}".format(config.batch_size))

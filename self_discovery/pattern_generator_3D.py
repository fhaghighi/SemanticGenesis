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

import sys
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from sklearn import metrics
from optparse import OptionParser
from glob import glob
from skimage.transform import resize

sys.setrecursionlimit(40000)

parser = OptionParser()


parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type="int")
parser.add_option("--crop_rows", dest="crop_rows", help="crop rows", default=64, type="int")
parser.add_option("--crop_cols", dest="crop_cols", help="crop cols", default=64, type="int")
parser.add_option("--data_dir", dest="data_dir", help="the directory to pre-training dataset", default=None, type="string")
parser.add_option("--save", dest="save", help="the directory to save processed 3D cubes", default="./Semantic_Genesis_data", type="string")

parser.add_option("--is_normalized", dest="is_normalized", help="True if the images are already normalized", action="store_true", default=False)
parser.add_option("--nb_instances", dest="nb_instances", help="number of instances of each pattern", default=200, type=int)
parser.add_option("--nb_instances_val", dest="nb_instances_val", help="number of instances of each pattern for validation", default=30, type=int)
parser.add_option("--nb_classes", dest="nb_classes", help="number of classes", default=200, type=int)
parser.add_option("--minPatchSize", dest="minPatchSize", help="minimum cube size ", default=50, type=int)
parser.add_option("--maxPatchSize", dest="maxPatchSize", help="maximum cube size", default=80, type=int)
parser.add_option("--multi_res", dest="multi_res", help="True if you want to extract multi-resolution patches", action="store_true", default=False)
parser.add_option("--distance_threshold", dest="distance_threshold", help="minimum distance between points", default=5, type="int")
parser.add_option("--prev_coordinates", dest="prev_coordinates", help="address of previously generated coordinated", default= None)
parser.add_option("--train_features", dest="train_features", help="path to the saved features for training images", default= "train_features.npy")
parser.add_option("--val_features", dest="val_features", help="path to the saved features for validation images", default= "validation_features.npy")


(options, args) = parser.parse_args()

seed = 1
random.seed(seed)

assert options.data_dir is not None
assert options.save is not None
assert options.train_features is not None
assert options.val_features is not None

if not os.path.exists(options.save):
    os.makedirs(options.save)


train_features = np.load(options.train_features)
print("train features:",train_features.shape)
validation_features = np.load(options.val_features)
print("validation features:",validation_features.shape)

target_path_train=os.path.join(options.data_dir,"train")
target_path_val= os.path.join(options.data_dir,"validation")


train_images_list=glob(os.path.join(target_path_train, "*.npy"))
train_images_list.sort()
val_images_list=glob(os.path.join(target_path_val, "*.npy"))
val_images_list.sort()

class setup_config():
    hu_max = 1000.0
    hu_min = -1000.0
    HU_thred = (-150.0 - hu_min) / (hu_max - hu_min)

    def __init__(self,
                 input_rows=None,
                 input_cols=None,
                 input_deps=None,
                 crop_rows=None,
                 crop_cols=None,
                 len_border=None,
                 len_border_z=None,
                 scale=None,
                 DATA_DIR=None,
                 train_fold=[0, 1, 2, 3, 4],
                 valid_fold=[5, 6],
                 test_fold=[7, 8, 9],
                 len_depth=None,
                 lung_min=0.7,
                 lung_max=1.0,
                 is_normalized=False,
                 minPatchSize=50,
                 maxPatchSize=100,
                 multi_res=True,
                 nb_instances=200,
                 nb_instances_val=30,
                 nb_classes=200,
                 save="./"


                 ):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.len_border = len_border
        self.len_border_z = len_border_z
        self.scale = scale
        self.DATA_DIR = DATA_DIR
        self.train_fold = train_fold
        self.valid_fold = valid_fold
        self.test_fold = test_fold
        self.len_depth = len_depth
        self.lung_min = lung_min
        self.lung_max = lung_max
        self.is_normalized=is_normalized
        self.minPatchSize=minPatchSize
        self.maxPatchSize=maxPatchSize
        self.multi_res=multi_res
        self.nb_instances = nb_instances
        self.nb_instances_val = nb_instances_val
        self.nb_classes = nb_classes
        self.save=save

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


config = setup_config(input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      crop_rows=options.crop_rows,
                      crop_cols=options.crop_cols,
                      len_border=100,
                      len_border_z=30,
                      len_depth=3,
                      lung_min=0.7,
                      lung_max=0.15,
                      DATA_DIR=options.data_dir,
                      is_normalized=options.is_normalized,
                     minPatchSize=options.minPatchSize,
                        maxPatchSize=options.maxPatchSize,
                    multi_res=options.multi_res,
                    nb_instances = options.nb_instances,
                     nb_instances_val = options.nb_instances_val,
                     nb_classes = options.nb_classes,
                      save=options.save,
                      )
config.display()

def initialization():
    coordinates = []
    ref_selected = np.zeros((1, train_features.shape[0]))
    visited_labels=[]
    if options.prev_coordinates is not None:
        fileReader = open(options.prev_coordinates, "r")
        cnt=0
        for row in fileReader.readlines():
            if cnt==0:
                cnt+=1
                continue
            line = row.split('\n')[0]
            infos = line.split("#")
            file_name=infos[0]
            coors=infos[1]
            x=int(coors[0])
            y=int(coors[1])
            z=int(coors[2])
            label=int(infos[2])
            if label not in visited_labels:
                visited_labels.append(label)
                coordinates.append(np.array([x, y, z]))
                for i in range (len(train_images_list)):
                    if train_images_list[i]==file_name:
                        ref_selected[0,i]=1
                        break
    print("number of previous coordinated:", len(coordinates))
    print("number of previous selected references:", np.count_nonzero(ref_selected))
    print("previous classes:",visited_labels)

    return coordinates,ref_selected

def get_random_coordinate(config, img_array, coordinates):
    size_x, size_y, size_z = img_array.shape
    if size_z - config.input_deps - config.len_depth - 1 - config.len_border_z < config.len_border_z:
        return None

    if not config.is_normalized:
        print("data is not normalized")
        img_array[img_array < config.hu_min] = config.hu_min
        img_array[img_array > config.hu_max] = config.hu_max
        img_array = 1.0 * (img_array - config.hu_min) / (config.hu_max - config.hu_min)


    repeated=False
    cnt = 0
    while True:
        print("coordinate generator trial number:",cnt)
        cnt += 1
        if cnt > 200:
            return -1,-1,-1

        start_x = random.randint(0 + config.len_border, size_x - config.crop_rows - 1 - config.len_border)
        start_y = random.randint(0 + config.len_border, size_y - config.crop_cols - 1 - config.len_border)
        start_z = random.randint(0 + config.len_border_z,
                                 size_z - config.input_deps - config.len_depth - 1 - config.len_border_z)

        for i in range (coordinates.shape[0]):
                dist = np.linalg.norm(np.array([start_x,start_y,start_z])- coordinates[i,:])
                # print(dist)
                if dist < options.distance_threshold:
                    repeated = True
                    break

        if not repeated:
            crop_window = img_array[start_x: start_x + config.crop_rows,
                          start_y: start_y + config.crop_cols,
                          start_z: start_z + config.input_deps + config.len_depth,
                          ]

            t_img = np.zeros((config.input_rows, config.input_cols, config.input_deps), dtype=float)
            d_img = np.zeros((config.input_rows, config.input_cols, config.input_deps), dtype=float)

            for d in range(config.input_deps):
                for i in range(config.input_rows):
                    for j in range(config.input_cols):
                        for k in range(config.len_depth):
                            if crop_window[i, j, d + k] >= config.HU_thred:
                                t_img[i, j, d] = crop_window[i, j, d + k]
                                d_img[i, j, d] = k
                                break
                            if k == config.len_depth - 1:
                                d_img[i, j, d] = k

            d_img = d_img.astype('float32')
            d_img /= (config.len_depth - 1)
            d_img = 1.0 - d_img

            if np.sum(d_img) > config.lung_max * config.input_rows * config.input_cols * config.input_deps:
                continue
            else:
                return start_x, start_y, start_z




def get_self_learning_data( config):
    coordinates,ref_selected=initialization()
    print("number of coordinated to be generated:", config.nb_classes - len(coordinates))


# three multi-resolution cubes
    train_data1 = np.zeros((config.nb_instances * config.nb_classes, config.input_rows, config.input_cols, config.input_deps))
    train_data2 = np.zeros( (config.nb_instances * config.nb_classes, config.input_rows, config.input_cols, config.input_deps))
    train_data3 = np.zeros((config.nb_instances * config.nb_classes, config.input_rows, config.input_cols, config.input_deps))

    train_label1 = np.zeros((config.nb_instances * config.nb_classes, 1))
    train_label2 = np.zeros((config.nb_instances * config.nb_classes, 1))
    train_label3 = np.zeros((config.nb_instances * config.nb_classes, 1))

    val_data1 = np.zeros((config.nb_instances_val * config.nb_classes, config.input_rows, config.input_cols, config.input_deps))
    val_label1 = np.zeros((config.nb_instances_val * config.nb_classes, 1))


    train_counter = (len(coordinates))*config.nb_instances
    print("train counter:",train_counter)
    val_counter = (len(coordinates))*config.nb_instances_val
    print("validation counter:",val_counter)

    d=10

    train_data1_fileWriter = open("./train_data1_vwGen_exRef.txt", "a")
    train_data2_fileWriter = open("./train_data2_vwGen_exRef.txt", "a")
    train_data3_fileWriter = open("./train_data3_vwGen_exRef.txt", "a")

    val_data1_fileWriter = open("./val_data1_vwGen_exRef.txt", "a")

    for i in range(len(coordinates),config.nb_classes):
        print("class:", i)
        x=-1
        y=-1
        z=-1
        # choose a random reference for each class of vws
        while x==-1 and y==-1 and z==-1:
            while True:
                ref_index = np.random.randint(0, train_features.shape[0])
                if ref_selected[0,ref_index] ==0:
                    ref_selected[0, ref_index] =1
                    break
            ref_feature = train_features[ref_index, :]
            # find similar patients
            distances_train = np.zeros((1, train_features.shape[0]))
            distances_val = np.zeros((1, validation_features.shape[0]))

            for l in range(train_features.shape[0]):
                distances_train[0, l] = np.linalg.norm(ref_feature - train_features[l, :])
            train_sorted_distances_indexes = np.argsort(distances_train)

            for l in range(validation_features.shape[0]):
                distances_val[0, l] = np.linalg.norm(ref_feature - validation_features[l, :])
            val_sorted_distances_indexes = np.argsort(distances_val)

            ref_image=np.load(train_images_list[train_sorted_distances_indexes[0, 0]])

            x,y,z= get_random_coordinate(config,ref_image,np.array(coordinates))
            print("x,y,z",x,y,z)

        coordinates.append(np.array([x,y,z]))

        for j in range(options.nb_instances):
            print("instance:", j)
            img_addr = train_images_list[train_sorted_distances_indexes[0, j]]
            im = np.load(img_addr)

            patch1 = im[x: x + config.crop_rows, y: y + config.crop_cols, z: z + config.input_deps]
            if config.crop_rows != config.input_rows or config.crop_cols != config.input_cols:
                patch1 = resize(patch1, (config.input_rows, config.input_cols, config.input_deps), preserve_range=True)

            train_data1[train_counter, :, :, :] = patch1
            train_label1[train_counter, 0] = i

            train_data1_fileWriter.write(img_addr)
            train_data1_fileWriter.write("#")
            train_data1_fileWriter.write(str(x))
            train_data1_fileWriter.write(",")
            train_data1_fileWriter.write(str(y))
            train_data1_fileWriter.write(",")
            train_data1_fileWriter.write(str(z))
            train_data1_fileWriter.write(",")
            train_data1_fileWriter.write(str(config.input_rows))
            train_data1_fileWriter.write(",")
            train_data1_fileWriter.write(str(config.input_deps))
            train_data1_fileWriter.write("#")
            train_data1_fileWriter.write(str(i))
            train_data1_fileWriter.write("\n")

            if config.multi_res:
                while True:
                    randPatchSize = np.random.randint(config.minPatchSize, config.maxPatchSize)
                    deltaX = np.random.randint(-d, d)
                    deltaY = np.random.randint(-d, d)
                    if ((x + deltaX + randPatchSize <= im.shape[0]) and ( y + deltaY + randPatchSize <= im.shape[1])):
                        patch2 = im[x+deltaX: x +deltaX+ randPatchSize, y+deltaY: y+deltaY + randPatchSize,  z: z + config.input_deps]
                        if randPatchSize != config.input_rows or randPatchSize != config.input_cols:
                            patch2 = resize(patch2, (config.input_rows, config.input_cols, config.input_deps),
                                            preserve_range=True)

                        train_data2[train_counter, :, :, :] = patch2
                        train_label2[train_counter, 0] = i

                        train_data2_fileWriter.write(img_addr)
                        train_data2_fileWriter.write("#")
                        train_data2_fileWriter.write(str(x+deltaX))
                        train_data2_fileWriter.write(",")
                        train_data2_fileWriter.write(str(y+deltaY))
                        train_data2_fileWriter.write(",")
                        train_data2_fileWriter.write(str(z))
                        train_data2_fileWriter.write(",")
                        train_data2_fileWriter.write(str(randPatchSize))
                        train_data2_fileWriter.write(",")
                        train_data2_fileWriter.write(str(config.input_deps))
                        train_data2_fileWriter.write("#")
                        train_data2_fileWriter.write(str(i))
                        train_data2_fileWriter.write("\n")

                        break

                while True:
                    randPatchSize = np.random.randint(config.minPatchSize, config.maxPatchSize)
                    deltaX = np.random.randint(-d, d)
                    deltaY = np.random.randint(-d, d)
                    if (( x + deltaX + randPatchSize <= im.shape[0]) and ( y + deltaY + randPatchSize <= im.shape[1])):
                        patch3 = im[x+deltaX: x +deltaX+ randPatchSize,  y+deltaY: y+deltaY + randPatchSize, z: z + config.input_deps]
                        if randPatchSize != config.input_rows or randPatchSize != config.input_cols:
                            patch3 = resize(patch3, (config.input_rows, config.input_cols, config.input_deps),
                                            preserve_range=True)

                        train_data3[train_counter, :, :, :] = patch3
                        train_label3[train_counter, 0] = i

                        train_data3_fileWriter.write(img_addr)
                        train_data3_fileWriter.write("#")
                        train_data3_fileWriter.write(str(x + deltaX))
                        train_data3_fileWriter.write(",")
                        train_data3_fileWriter.write(str(y + deltaY))
                        train_data3_fileWriter.write(",")
                        train_data3_fileWriter.write(str(z))
                        train_data3_fileWriter.write(",")
                        train_data3_fileWriter.write(str(randPatchSize))
                        train_data3_fileWriter.write(",")
                        train_data3_fileWriter.write(str(config.input_deps))
                        train_data3_fileWriter.write("#")
                        train_data3_fileWriter.write(str(i))
                        train_data3_fileWriter.write("\n")

                        break


            train_counter += 1

        for j in range(config.nb_instances_val):
            img_addr = val_images_list[val_sorted_distances_indexes[0, j]]
            im = np.load(img_addr)

            patch1 = im[x: x + config.crop_rows, y: y + config.crop_cols,  z: z + config.input_deps]
            if config.crop_rows != config.input_rows or config.crop_cols != config.input_cols:
                patch1 = resize(patch1, (config.input_rows, config.input_cols, config.input_deps),
                                preserve_range=True)

            val_data1[val_counter, :, :, :] = patch1
            val_label1[val_counter, 0] = i
            val_counter += 1

        if (i+1)%50==0:
            np.save(os.path.join(config.save, "train_data1_vwGen_ex_ref_fold" + str((i+1)/50)), train_data1[(i-49)*config.nb_instances:(i+1)*config.nb_instances-1,:,:,:])
            np.save(os.path.join(config.save, "train_data2_vwGen_ex_ref_fold" + str((i+1)/50)), train_data2[(i-49)*config.nb_instances:(i+1)*config.nb_instances-1,:,:,:])
            np.save(os.path.join(config.save, "train_data3_vwGen_ex_ref_fold" + str((i+1)/50)), train_data3[(i-49)*config.nb_instances:(i+1)*config.nb_instances-1,:,:,:])

            np.save(os.path.join(config.save, "train_label1_vwGen_ex_ref_fold"+ str((i+1)/50)), train_label1[(i-49)*config.nb_instances:(i+1)*config.nb_instances-1,:])
            np.save(os.path.join(config.save, "train_label2_vwGen_ex_ref_fold"+ str((i+1)/50)), train_label2[(i-49)*config.nb_instances:(i+1)*config.nb_instances-1,:])
            np.save(os.path.join(config.save, "train_label3_vwGen_ex_ref_fold"+ str((i+1)/50)), train_label3[(i-49)*config.nb_instances:(i+1)*config.nb_instances-1,:])

            np.save(os.path.join(config.save, "val_data1_vwGen_ex_ref_fold"+ str((i+1)/50)), val_data1[(i-49)*config.nb_instances_val:(i+1)*config.nb_instances_val-1,:,:,:])
            np.save(os.path.join(config.save, "val_label1_vwGen_ex_ref_fold"+ str((i+1)/50)), val_label1[(i-49)*config.nb_instances_val:(i+1)*config.nb_instances_val-1,:])
            print("data saved!")
    print("number of selected references:",np.count_nonzero(ref_selected))



get_self_learning_data(config)
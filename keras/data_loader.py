# Copyright (c) ASU GitHub Project.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################


import keras
from sklearn.utils import shuffle
from utils.purturbation import *



class DataGenerator(keras.utils.Sequence):
    def __init__(self, images, labels, batch_size=16, dim=(64, 64, 32), nb_classes=44, augmentation=True):
        self.images = images
        print(self.images.shape)
        self.labels=labels
        print(self.labels.shape)
        self.batch_size = batch_size
        self.dim = dim
        self.length = self.images.shape[0]
        self.nb_classes=nb_classes
        self.augmentation=augmentation

    def __len__(self):
        return int(np.floor(self.labels.shape[0] / self.batch_size))

    def __getitem__(self, index):
        no_of_images_per_batch = int(self.batch_size)

        data = self.images[index * no_of_images_per_batch:(index + 1) * no_of_images_per_batch, :]
        labels = self.labels[index * no_of_images_per_batch:(index + 1) * no_of_images_per_batch, :]
        return self.luna_loader(data, labels)

    def on_epoch_end(self):
        self.images, self.labels = shuffle(self.images, self.labels)

    def luna_loader(self, data,labels):
        nb_classes = self.nb_classes

        input_rows = self.dim[0]
        input_cols = self.dim[1]
        input_depth = self.dim[2]
        x = np.zeros((self.batch_size, 1, input_rows, input_cols, input_depth), dtype="float")
        y_rec = np.zeros((self.batch_size, 1, input_rows, input_cols, input_depth), dtype="float")
        y_cls = np.zeros((self.batch_size, nb_classes), dtype="int32")

        for i in range(data.shape[0]):
            patch = data[i,:,:,:]
            label=int(labels[i, 0])

            y_rec[i, :, :, :,:] =np.expand_dims(patch, axis=0)
            y_cls[i, :] = keras.utils.to_categorical(label, self.nb_classes)

            patch_el = copy.deepcopy(patch)
            if self.augmentation:
                patch_el = elastic_transform(patch_el)

            r=random.random()

            if r<= 0.25:
                purturbed_cube=local_pixel_shuffling(patch_el)
            elif 0.25 <r<=0.5:
                purturbed_cube= non_linear_transformation(patch_el)
            elif 0.5 < r <= 0.75:
                purturbed_cube= image_in_painting(patch_el)
            else:
                purturbed_cube= image_out_painting(patch_el)

            x[i,:, :,:,:] = np.expand_dims(purturbed_cube, axis=0)

        x, y_rec,y_cls = shuffle(x,y_rec, y_cls, random_state=0)

        return x, [y_rec,y_cls]

# Copyright (c) ASU GitHub Project.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################


import os
import shutil
import sys


class setup_config:

    optimizer = "Adam"
    nb_epoch = 10000
    patience = 50
    workers = 10
    max_queue_size = workers * 4

    def __init__(self,
                 data_augmentation=True,
                 input_rows=64,
                 input_cols=64,
                 input_deps=32,
                 batch_size=12,
                 nb_class=1,
                 verbose=1,
                 cls_classes=44,
                 nb_instances=200,
                 nb_val_instances=30,
                 lr=0.001,
                 nb_multires_patch=3,
                 lambda_rec=1,
                 lambda_cls=0.01,
                DATA_DIR=None,
                 exp_choice ='en_de'
                 ):
        self.data_augmentation=data_augmentation
        self.exp_name = "semantic_genesis_chest_ct"
        self.input_rows, self.input_cols = input_rows, input_cols
        self.input_deps = input_deps
        self.batch_size = batch_size
        self.verbose = verbose
        self.nb_class = nb_class
        self.cls_classes=cls_classes
        self.nb_instances = nb_instances
        self.nb_val_instances = nb_val_instances

        self.lr=lr
        if nb_class > 1:
            self.activation = "softmax"
        else:
            self.activation = "sigmoid"
        self.nb_multires_patch=nb_multires_patch
        self.exp_choice = exp_choice
        self.model_path = os.path.join("Checkpoints/",self.exp_choice)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.logs_path = os.path.join(self.model_path, "Logs")
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        self.lambda_rec = lambda_rec
        self. lambda_cls = lambda_cls
        self.DATA_DIR=DATA_DIR

        if os.path.exists(os.path.join(self.logs_path, "log.txt")):
            self.log_writter = open(os.path.join(self.logs_path, "log.txt"), 'a')
        else:
            self.log_writter = open(os.path.join(self.logs_path, "log.txt"), 'w')


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:",file=self.log_writter)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)),file=self.log_writter)
        print("\n",file=self.log_writter)

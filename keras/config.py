
# Copyright (c) ASU GitHub Project.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################


import os
from datetime import datetime


class setup_config:
    optimizer = "Adam"
    nb_epoch = 10000
    patience = 50

    def __init__(self,
                 data_augmentation=True,
                 input_rows=64,
                 input_cols=64,
                 input_deps=32,
                 batch_size=12,
                 decoder_block_type=None,
                 nb_class=1,
                 verbose=1,
                 cls_classes=44,
                 nb_instances=200,
                 lr=0.001,
                 nb_multires_patch=3,
                 elastic_prob=1,
                 weights=None,
                 unet_weights=None,
                 encoder_weights=None,
                 lambda_rec=1,
                 lambda_cls=0.01,
                DATA_DIR=None
                 ):
        self.data_augmentation=data_augmentation
        self.exp_name = "semantic_genesis_chest_ct"
        self.input_rows, self.input_cols = input_rows, input_cols
        self.input_deps = input_deps
        self.batch_size = batch_size
        self.verbose = verbose
        self.decoder_block_type = decoder_block_type
        self.nb_class = nb_class
        self.cls_classes=cls_classes
        self.nb_instances = nb_instances
        self.lr=lr
        if nb_class > 1:
            self.activation = "softmax"
        else:
            self.activation = "sigmoid"
        self.nb_multires_patch=nb_multires_patch

        self.model_path = "Checkpoints/"
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.logs_path = os.path.join(self.model_path, "Logs")
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        self.weights = weights
        self.unet_weights = unet_weights
        self.encoder_weights = encoder_weights
        self.lambda_rec = lambda_rec
        self. lambda_cls = lambda_cls
        self.DATA_DIR=DATA_DIR


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


    def date_str(self):
            return datetime.now().__str__().replace("-", "_").replace(" ", "_").replace(":", "_")


import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Dense,GlobalAveragePooling3D,Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D,Concatenate

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def ynet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, deconvolution=False,
                  depth=4, n_base_filters=32, batch_normalization=False, activation_name="sigmoid",
                  unet_weights=None,encoder_weights=None,cls_classes=44):
    """
    Builds the 3D YNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the YNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D YNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()
    num_layer = 0

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, layer_depth=num_layer)
        num_layer += 1
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization, layer_depth=num_layer)
        num_layer += 1
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])


    if encoder_weights is not None:
        x = GlobalAveragePooling3D()(current_layer)
        o = Dense(cls_classes, activation="softmax")(x)
        encoder_model = Model(inputs=inputs, outputs=o)
        encoder_model.load_weights(encoder_weights)
        encoder_model.summary()
        current_layer = encoder_model.get_layer('depth_7_relu').output

    for layer_depth in range(depth-2, -1, -1):

        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], layer_depth=num_layer,
                                                 input_layer=concat, batch_normalization=batch_normalization)
        num_layer += 1
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], layer_depth=num_layer,
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
        num_layer += 1

    final_convolution = Conv3D(n_labels, (1, 1, 1),name="final_conv")(current_layer)
    act = Activation(activation_name,name="reconst_output")(final_convolution)
    unet_model = Model(inputs=inputs, outputs=act)


    if unet_weights is not None:
        unet_model.load_weights(unet_weights)

    cls_input = unet_model.get_layer('depth_7_relu').output
    layer1 = create_convolution_block(input_layer=cls_input, n_filters=512,
                                      batch_normalization=batch_normalization, layer_depth=num_layer)

    cls_conv2=GlobalAveragePooling3D()(layer1)
    cls_fc = Dense(1024, activation="relu", name='cls_fc')(cls_conv2)
    cls_output = Dense(cls_classes, activation="softmax", name='cls_output')(cls_fc)
    model = Model(inputs, [unet_model.output,cls_output])
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False, layer_depth=None):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, name="depth_"+str(layer_depth)+"_conv")(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1, name="depth_"+str(layer_depth)+"_bn")(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1, name="depth_"+str(layer_depth)+"_in")(layer)
    if activation is None:
        return Activation('relu', name="depth_"+str(layer_depth)+"_relu")(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

if __name__ == '__main__':
    model=ynet_model_3d((1, 64, 64, 32),batch_normalization=True,encoder_weights=None)
    model.summary()
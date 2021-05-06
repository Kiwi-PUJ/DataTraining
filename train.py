## @package Training_app
# Training code developed with Tensorflow Keras. Content: Unet, Unet++ and FCN
# 
#  @version 1 
#
# Pontificia Universidad Javeriana
# 
# Electronic Enginnering
# 
# Developed by:
# - Andrea Juliana Ruiz Gomez
#       Mail: <andrea_ruiz@javeriana.edu.co>
#       GitHub: andrearuizg
# - Pedro Eli Ruiz Zarate
#       Mail: <pedro.ruiz@javeriana.edu.co>
#       GitHub: PedroRuizCode
#  
# With support of:
# - Francisco Carlos Calderon Bocanegra
#       Mail: <calderonf@javeriana.edu.co>
#       GitHub: calderonf
# - John Alberto Betancout Gonzalez
#       Mail: <john@kiwibot.com>
#       GitHub: JohnBetaCode

import os
from time import time
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
ReduceLROnPlateau, CSVLogger, TensorBoard)


## Load data
# Load the data
# @param path Path of the image
def load_data(path):
    images_train = sorted(glob(os.path.join(path, "images/train/*")))
    masks_train = sorted(glob(os.path.join(path, "masks/train/*")))
    images_valid = sorted(glob(os.path.join(path, "images/valid/*")))
    masks_valid = sorted(glob(os.path.join(path, "masks/valid/*")))

    train_x, valid_x = images_train, images_valid
    train_y, valid_y = masks_train, masks_valid

    return (train_x, train_y), (valid_x, valid_y)


## Read image
# Read the images
# @param path Path of the image
def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x


## Read mask
# Read the mask of the images
# @param path Path of the mask
def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x / 1.0
    x = np.expand_dims(x, axis=-1)
    return x


## Parse
# Read images and masks and convert to TensorFlow dataformat
# @param x Images
# @param y Masks
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y


## Dataset
# Read images and masks and convert to TensorFlow format
# @param x Images
# @param y Masks
# @param batch Batch size
def tf_dataset(x, y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF)
    dataset = dataset.with_options(options)
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


## Down sample function
# Make the down sample of the layer
# @param x Input
# @param filters The dimensionality of the output space
# @param kernel_size Height and width of the 2D convolution window
# @param padding Padding
# @param strides Strides of the convolution along the height and width
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, 
                activation="relu")(x)
    c = BatchNormalization()(c)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, 
                activation="relu")(c)
    c = BatchNormalization()(c)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p


## Up sample function
# Make the up sample of the layer
# @param x Input
# @param skip The skip connection is made to avoid the loss of accuracy
# in the downsampling layers. In case the image becomes so small that
# it has no information, the weights are calculated with the skip layer.
# @param filters The dimensionality of the output space
# @param kernel_size Height and width of the 2D convolution window
# @param padding Padding
# @param strides Strides of the convolution along the height and width
def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, 
                activation="relu")(concat)
    c = BatchNormalization()(c)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, 
                activation="relu")(c)
    c = BatchNormalization()(c)
    return c


## Bottleneck function
# Added to reduce the number of feature maps in the network
# @param x Input
# @param filters The dimensionality of the output space
# @param kernel_size Height and width of the 2D convolution window
# @param padding Padding
# @param strides Strides of the convolution along the height and width
def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, 
                activation="relu")(x)
    c = BatchNormalization()(c)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, 
                activation="relu")(c)
    c = BatchNormalization()(c)
    return c


## Unet 1
# Unet implementation
# @param f Filters dimensionality
def UNet_1(f):
    inputs = Input((256, 256, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])   # 256 -> 128
    c2, p2 = down_block(p1, f[1])   # 128 -> 64
    c3, p3 = down_block(p2, f[2])   # 64 -> 32
    c4, p4 = down_block(p3, f[3])   # 32 -> 16

    bn = bottleneck(p4, f[4])

    u3 = up_block(bn, c4, f[3])     # 16 -> 32
    u4 = up_block(u3, c3, f[2])     # 32 -> 64
    u5 = up_block(u4, c2, f[1])     # 64 -> 128
    u6 = up_block(u5, c1, f[0])     # 128 -> 256

    # Classifying layer

    outputs = Dropout(0.1)(u6)
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(outputs)

    model = Model(inputs, outputs)
    return model


## Unet 2
# Unet implementation
# @param f Filters dimensionality
def UNet_2(f):
    inputs = Input((256, 256, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])   # 256 -> 128
    c2, p2 = down_block(p1, f[1])   # 128 -> 64
    c3, p3 = down_block(p2, f[2])   # 64 -> 32
    c4, p4 = down_block(p3, f[3])   # 32 -> 16
    c5, p5 = down_block(p4, f[4])   # 16 -> 8
    c6, p6 = down_block(p5, f[5])   # 8 -> 4

    bn = bottleneck(p6, f[6])

    u1 = up_block(bn, c6, f[5])     # 4 -> 8
    u2 = up_block(u1, c5, f[4])     # 8 -> 16
    u3 = up_block(u2, c4, f[3])     # 16 -> 32
    u4 = up_block(u3, c3, f[2])     # 32 -> 64
    u5 = up_block(u4, c2, f[1])     # 64 -> 128
    u6 = up_block(u5, c1, f[0])     # 128 -> 256

    # Classifying layer

    outputs = Dropout(0.1)(u6)
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(outputs)

    model = Model(inputs, outputs)
    return model


## Unet++ 1
# Unet++ implementation
# @param f Filters dimensionality
def UNetpp_1(f):
    inputs = Input((256, 256, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0])   # 256 -> 128
    c2, p2 = down_block(p1, f[1])   # 128 -> 64
    c3, p3 = down_block(p2, f[2])   # 64 -> 32
    c4, p4 = down_block(p3, f[3])   # 32 -> 16

    u11 = up_block(c2, c1, f[0])    # 128 -> 256
    u21 = up_block(c3, c2, f[1])    # 64 -> 128
    u31 = up_block(c4, c3, f[2])    # 32 -> 64

    u21_1 = Concatenate()([c2, u21])
    u22 = up_block(u31, u21_1, f[1])  # 128 -> 256

    u11_1 = Concatenate()([c1, u11])
    u12 = up_block(u21, u11_1, f[0])  # 64 -> 128

    u12_1 = Concatenate()([u11_1, u12])
    u13 = up_block(u22, u12_1, f[0])  # 128 -> 256
    
    bn = bottleneck(p4, f[4])
    
    u3 = up_block(bn, c4, f[3])     # 16 -> 32

    u31_1 = Concatenate()([c3, u31])
    u4 = up_block(u3, u31_1, f[2])    # 32 -> 64

    u22_1 = Concatenate()([u21_1, u22])
    u5 = up_block(u4, u22_1, f[1])    # 64 -> 128

    u13_1 = Concatenate()([u12_1, u13])
    u6 = up_block(u5, u13_1, f[0])    # 128 -> 256

    # Classifying layer

    outputs = Dropout(0.1)(u6)
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(outputs)

    model = Model(inputs, outputs)
    return model


## Unet++ 2
# Unet++ implementation
# @param f Filters dimensionality
def UNetpp_2(f):
    inputs = Input((256, 256, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0])   # 256 -> 128
    c2, p2 = down_block(p1, f[1])   # 128 -> 64
    c3, p3 = down_block(p2, f[2])   # 64 -> 32
    c4, p4 = down_block(p3, f[3])   # 32 -> 16
    c5, p5 = down_block(p4, f[4])   # 16 -> 8
    c6, p6 = down_block(p5, f[5])   # 8 -> 4

    u11 = up_block(c2, c1, f[0])    # 128 -> 256
    u21 = up_block(c3, c2, f[1])    # 64 -> 128
    u31 = up_block(c4, c3, f[2])    # 32 -> 64
    u41 = up_block(c5, c4, f[3])    # 16 -> 32
    u51 = up_block(c6, c5, f[4])    # 8 -> 16

    u11_1 = Concatenate()([c1, u11])
    u12 = up_block(u21, u11_1, f[0])  # 128 -> 256

    u21_1 = Concatenate()([c2, u21])
    u22 = up_block(u31, u21_1, f[1])  # 64 -> 128

    u31_1 = Concatenate()([c3, u31])
    u32 = up_block(u41, u31_1, f[2])  # 32 -> 64

    u41_1 = Concatenate()([c4, u41])
    u42 = up_block(u51, u41_1, f[3])  # 16 -> 32

    u12_1 = Concatenate()([u11_1, u12])
    u13 = up_block(u22, u12_1, f[0])  # 128 -> 256

    u22_1 = Concatenate()([u21_1, u22])
    u23 = up_block(u32, u22_1, f[1])  # 64 -> 128

    u32_1 = Concatenate()([u31_1, u32])
    u33 = up_block(u42, u32_1, f[2])  # 32 -> 64

    u13_1 = Concatenate()([u12_1, u13])
    u14 = up_block(u23, u13_1, f[0])  # 128 -> 256

    u23_1 = Concatenate()([u22_1, u23])
    u24 = up_block(u33, u23_1, f[1])  # 64 -> 128

    u14_1 = Concatenate()([u13_1, u14])
    u15 = up_block(u24, u14_1, f[0])  # 128 -> 256
    
    bn = bottleneck(p6, f[6])
    
    u1 = up_block(bn, c6, f[5])     # 4 -> 8

    u51_1 = Concatenate()([c5, u51])
    u2 = up_block(u1, u51_1, f[4]) 	# 8 -> 16

    u42_1 = Concatenate()([u41_1, u42])
    u3 = up_block(u2, u42_1, f[3]) 	# 16 -> 32

    u33_1 = Concatenate()([u32_1, u33])
    u4 = up_block(u3, u33_1, f[2]) 	# 32 -> 64

    u24_1 = Concatenate()([u23_1, u24])
    u5 = up_block(u4, u24_1, f[1]) 	# 64 -> 128

    u15_1 = Concatenate()([u14_1, u15])
    u6 = up_block(u5, u15_1, f[0]) 	# 128 -> 256

    # Classifying layer

    outputs = Dropout(0.1)(u6)
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(outputs)

    model = Model(inputs, outputs)
    return model


## FCN 1
# Fully Convolutional Network implementation
# @param f Filters dimensionality
def FCN_1(f):
    inputs = Input((256, 256, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])   # 256 -> 128
    c2, p2 = down_block(p1, f[1])   # 128 -> 64
    c3, p3 = down_block(p2, f[2])   # 64 -> 32
    c4, p4 = down_block(p3, f[3])   # 32 -> 16

    bn = bottleneck(p4, f[4])

    pr1 = Conv2D(1, (4, 4), activation='relu', padding='same', strides=1)(bn)
    pr2 = Conv2D(1, (8, 8), activation='relu', padding='same', strides=1)(p3)
    pr3 = Conv2D(1, (16, 16), activation='relu', padding='same', strides=1)(p2)

    us1 = UpSampling2D((2, 2))(pr1)
    add1 = Add()([us1, pr2])
    us2 = UpSampling2D((2, 2))(add1)
    add2 = Add()([us2, pr3])
    us3 = UpSampling2D((4, 4))(add2)

    # Classifying layer

    outputs = Dropout(0.1)(us3)
    outputs = Conv2D(1, (32, 32), activation='sigmoid', padding='same')(outputs)

    model = Model(inputs, outputs)
    return model


## FCN 2
# Fully Convolutional Network implementation
# @param f Filters dimensionality
def FCN_2(f):
    inputs = Input((256, 256, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])   # 256 -> 128
    c2, p2 = down_block(p1, f[1])   # 128 -> 64
    c3, p3 = down_block(p2, f[2])   # 64 -> 32
    c4, p4 = down_block(p3, f[3])   # 32 -> 16
    c5, p5 = down_block(p4, f[4])   # 16 -> 8
    c6, p6 = down_block(p5, f[5])   # 8 -> 4

    bn = bottleneck(p6, f[6])

    pr1 = Conv2D(1, (1, 1), activation='relu', padding='same', strides=1)(bn)
    pr2 = Conv2D(1, (2, 2), activation='relu', padding='same', strides=1)(p5)
    pr3 = Conv2D(1, (4, 4), activation='relu', padding='same', strides=1)(p4)
    pr4 = Conv2D(1, (8, 8), activation='relu', padding='same', strides=1)(p3)
    pr5 = Conv2D(1, (16, 16), activation='relu', padding='same', strides=1)(p2)

    us1 = UpSampling2D((2, 2))(pr1)		
    add1 = Add()([us1, pr2])
    us2 = UpSampling2D((2, 2))(add1)	
    add2 = Add()([us2, pr3])
    us3 = UpSampling2D((2, 2))(add2)	
    add3 = Add()([us3, pr4])
    us4 = UpSampling2D((2, 2))(add3)	
    add4 = Add()([us4, pr5])
    us5 = UpSampling2D((4, 4))(add4)

    # Classifying layer

    outputs = Dropout(0.1)(us5)
    outputs = Conv2D(1, (32, 32), activation='sigmoid', padding='same')(outputs)

    model = Model(inputs, outputs)
    return model    


## Training
# CNN training
def training(m_name):
    ## Dataset
    path = "media/"
    (train_x, train_y), (valid_x, valid_y) = load_data(path)

    ## Hyperparameters
    batch = 15
    epochs = 190

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    ## Time
    t0 = time()

    ## Filters
    f = [16, 32, 64, 128, 256, 512, 1024]

    if m_name == "unetv1":
        model = UNet_1(f)
    elif m_name == "unetv2":
        model = UNet_2(f)
    elif m_name == "unetppv1":
        model = UNetpp_1(f)
    elif m_name == "unetppv2":
        model = UNetpp_2(f)
    elif m_name == "fcnv1":
        model = FCN_1(f)
    else:
        model = FCN_2(f)

    m_sum = 'files/model_summary_%s_BN.txt' % m_name
    m_log = 'logs/%s_BN/scalars/' % m_name
    m_h5 = 'files/model_%s_BN.h5' % m_name
    m_data = 'files/data_%s_BN.csv' % m_name
    m_time = 'files/time_%s_BN.txt' % m_name

    model.compile(optimizer="adam", loss="binary_crossentropy", 
                    metrics=["acc", Precision(), Recall()])
    with open(m_sum, 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    train_steps = len(train_x) // batch
    valid_steps = len(valid_x) // batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    logdir = m_log
    tensorboard_callback = TensorBoard(log_dir=logdir)

    callbacks = [
        ModelCheckpoint(m_h5),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
        CSVLogger(m_data),
        tensorboard_callback,
        EarlyStopping(monitor='val_loss', patience=33, 
        restore_best_weights=False)
    ]

    model.fit(train_dataset, validation_data=valid_dataset, 
                steps_per_epoch=train_steps, 
                validation_steps=valid_steps, epochs=epochs,
                callbacks=callbacks)
    time_tr = open(m_time, 'w')
    time_tr.write(str(time() - t0))


if __name__ == "__main__":

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    model_l = ["unetv1", "unetv2", "unetppv1", "unetppv2", "fcnv1", "fcnv2"]

    for model in model_l:
        with strategy.scope():
            print("\n\n\n\n\n Training", model, "model\n\n\n\n\n")
            training(model)

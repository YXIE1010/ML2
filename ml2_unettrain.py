#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to train the U-Net model based on tuned hyperparameters
Tools provided by WandB facilitates the tracking and real-time
monitoring of the training process
Weight and Biases: The AI developer platform (https://wandb.ai/site)

@author: Yan Xie (yanxieyx@umich.edu)
"""

# load the modules
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from keras.callbacks import History
import numpy as np
import wandb       
from wandb.integration.keras import WandbMetricsLogger    


# Define the custom metrics 
def iou(y_true, y_pred, smooth=1):
    """
    Calculate intersection over union (IoU) between images.
    Input shape : Batch x Height x Width x #Classes (BxHxWxN), N = 1 in our binary classification case
    Input value : 0 - w/o melting layer or 1 - with melting layer
    Using mean as reduction type for batch values.
    """    
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
            
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
    union = union - intersection
    ## calculate the intersection over union index    
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        
    return iou


def dice_coef(y_true, y_pred, smooth=1.e-9):
    """
    Calculate dice coefficient: evaluate the similarity between two datasets
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """    
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice_score = K.mean((2 * intersection + smooth) / (union + smooth), axis=0)
    
    return dice_score


# Define the custom loss function
def dice_loss(y_true, y_pred,smooth=1.e-9):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    dice = ( numerator + smooth) / (denominator + smooth)
    return 1 - dice


def dice_bce_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, axis=[1,2,3])
    dice = dice_loss(y_true, y_pred)
    return bce + dice

        
# Define the Unet Model in modules
def enblock(x, filters, dropout):
    x = keras.layers.Conv2D(filters=filters, kernel_size=(3,3), padding="same", strides=2)(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters=filters, kernel_size=(3,3), padding="same")(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(dropout)(x, training=False)
    return x

def deblock(x, filters, eskipconnet, dropout):
    x = keras.layers.Conv2DTranspose(filters=filters, kernel_size=(3,3), padding="same")(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2DTranspose(filters=filters, kernel_size=(3,3), padding="same", strides=2)(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)    
    # apply skip connection
    x = keras.layers.Concatenate()([x, eskipconnet])
    x = keras.layers.Dropout(dropout)(x, training=False)
    return x

def unet(input_size, output_channels, depth=4):
    # start with the input layer
    input_layer = keras.Input(shape=input_size, name="input_layer")
    
    # define the encoder layers (Downward)
    encoders = []
    for i in range(0, depth):
        if i == 0:
            e = enblock(input_layer, filters*(2**i), DROP_RATE)
        else:
            e = enblock(encoders[i-1], filters*(2**i), DROP_RATE)
        encoders.append(e)

    # defind the decoder layers (Upward)
    decoders = []
    for ii in reversed(range(0, depth-1)):
        if ii == depth-2:
            d = deblock(encoders[ii+1], filters*(2**(ii+1)), encoders[ii], DROP_RATE)
        else:
            d = deblock(decoders[depth-3-ii], filters*(2**(ii+1)), encoders[ii], DROP_RATE)
        decoders.append(d)
    
    # final part
    d1 = decoders[len(decoders)-1]
    y = keras.layers.Conv2DTranspose(filters, kernel_size=(3,3), padding="same")(d1)
    y = keras.layers.BatchNormalization(axis=-1)(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Conv2DTranspose(filters, kernel_size=(3,3), padding="same", strides=2)(y)
    y = keras.layers.BatchNormalization(axis=-1)(y)
    y = keras.layers.Activation('relu')(y)
    
    #Output layer
    output_layer = keras.layers.Conv2D(output_channels, kernel_size=(3,3), padding="same", activation="sigmoid")(y)
    
    #create the model
    model = keras.Model(input_layer, output_layer)
    
    return model


#################### Prepare model input ####################
# load the preprocessed datasets
fpathin = '/your/input/filepath/for/model-input-data/'
fpathout = '/your/output/filepath/for/hyperparameter-tuning-output/'

# access the data files with memory-map mode
data_train0 = np.load(fpathin + 'training.npy', mmap_mode='r' )
data_val0 = np.load(fpathin + 'validation.npy', mmap_mode='r' )
data_test0 = np.load(fpathin + 'testing.npy', mmap_mode='r' )

idml_train = np.where( np.sum(data_train0[:,:,:,3], axis=(1,2))>0 )[0]
idml_val = np.where( np.sum(data_val0[:,:,:,3], axis=(1,2))>0 )[0]
idml_test = np.where( np.sum(data_test0[:,:,:,3], axis=(1,2))>0 )[0]

# find the index of cases w/o melting layer
idno_train = np.setdiff1d( np.arange(0, data_train0.shape[0]), idml_train )
idno_val = np.setdiff1d( np.arange(0, data_val0.shape[0]), idml_val )
idno_test = np.setdiff1d( np.arange(0, data_test0.shape[0]), idml_test )

# set a fixed random seed
np.random.seed(41)
np.random.shuffle(idno_train)
np.random.shuffle(idno_val)
np.random.shuffle(idno_test)

# combine the melting layer cases with no melting layer cases
# same size for both category to generate a balanced dataset
id_train = np.concatenate( (idml_train, idno_train[0:idml_train.shape[0]]), axis=0 )
id_val = np.concatenate( (idml_val, idno_val[0:idml_val.shape[0]]), axis=0 )
id_test = np.concatenate( (idml_test, idno_test[0:idml_test.shape[0]]), axis=0)


train_input0 = data_train0[id_train, :, :, 0:3]
train_target0 = data_train0[id_train, :, :, 3]

# flippling the training dataset for data augmentation
train_input = np.concatenate((train_input0, np.flip(train_input0, axis=2)), axis=0)
train_target = np.concatenate((train_target0, np.flip(train_target0, axis=2)), axis=0)

val_input = data_val0[id_val, :, :, 0:3]
val_target = data_val0[id_val, :, :, 3]

test_input = data_test0[id_test, :, :, 0:3]
test_target = data_test0[id_test, :, :, 3]

# make the dimension 1 stands out
train_target = train_target[:, :, :, np.newaxis]
val_target = val_target[:, :, :, np.newaxis]
test_target = test_target[:, :, :, np.newaxis]

# choose random samples for sanity check of the model outputs
sample_input = val_input[[10,15,20,25,50,55,60,65], :, :, :]
sample_target = val_target[[10,15,20,25,50,55,60,65], :, :, :]

del data_train0, data_val0, data_test0
#################### End of preparation ####################


################### Config hyperparameters #################### 
# get the input size
input_dim = train_input[0].shape

# u-net model depth 
DEPTH = 4

# batch size
N_BATCH = 16

# learning rate of optimizer
LR = 5e-3

# number of filters
filters = 64

# Dropout rate
DROP_RATE = 0.2

# maximum number of epochs
N_EPOCHS = 80


###################  End of hyperparameters ################### 

# implement the UNet model
model_unet = unet(input_dim, 1, depth=DEPTH)
model_unet.summary()

# compile the UNet model
model_unet.compile(optimizer=keras.optimizers.RMSprop(learning_rate=LR),
              loss = dice_loss,
              metrics=[keras.metrics.BinaryAccuracy(name='biacc'),
                       dice_coef,
                       iou,
                       ])

####### start a new wandb run to track this script #######
wandb.init(
    # set the wandb project where this run will be logged
    project = "tuned-runs"
    )
##########################################################

history = History()
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint(filepath=fpathout+'bestmodel.keras', save_best_only=True, mode='auto')
callbacks_unet = [earlyStop, checkpoint, history, WandbMetricsLogger()]

model_unet.fit(train_input, train_target, epochs=N_EPOCHS, batch_size=N_BATCH, \
                 validation_data=(val_input, val_target), shuffle=True, callbacks=callbacks_unet)


# finish the wandb run, necessary in notebooks
wandb.finish()   
    
# save training history 
with open(fpathout +'history_metrics.npy', 'wb') as fout:
    np.save(fout, history.history, allow_pickle=True)


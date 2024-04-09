#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:48:49 2024

@author: yanxieyx
"""

#%% load the modules
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from keras.callbacks import History
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import wandb       
from wandb.keras import WandbMetricsLogger    
# Weight and Biases: The AI developer platform (https://wandb.ai/site)

#%% load the preprocessed datasets
fpathin = '/Users/yanxieyx/Documents/Research/MLdetection/Output/DataPre/'
fpathout = '/Users/yanxieyx/Documents/Research/MLdetection/Output/'
# access the data files with memory-map mode
data_train0 = np.load(fpathin + 'training.npy', mmap_mode='r' )
data_val0 = np.load(fpathin + 'validation.npy', mmap_mode='r' )
data_test0 = np.load(fpathin + 'testing.npy', mmap_mode='r' )

idml_train = np.where( np.sum(data_train0[:,:,:,3], axis=(1,2)) )[0]
idml_val = np.where( np.sum(data_val0[:,:,:,3], axis=(1,2)) )[0]

#%% start with partial data
# part of the training samples
train_input = data_train0[idml_train[0:500], :, :, 0:3]
train_target = data_train0[idml_train[0:500], :, :, 3]
# part of the validation samples
val_input = data_val0[idml_val[0:50], :, :, 0:3]
val_target = data_val0[idml_val[0:50], :, :, 3]  # index 10 and 20 has melting layers

# make the dimension 1 stands out
train_target = train_target[:, :, :, np.newaxis]
val_target = val_target[:, :, :, np.newaxis]

# choose the samples for visualization: index 10, 15, 20, 25
sample_input = val_input[[10,15,20,25], :, :, :]
sample_target = val_target[[10,15,20,25], :, :, :]

del data_train0, data_val0

# get the input size
input_dim = train_input[0].shape
# number of epochs 
N_EPOCHS = 28
# batch size
N_BATCH = 50

#%% Define the custom metrics 
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
    
    #y_true = y_true.astype(np.int8)
    #y_pred = y_pred.astype(np.int8)   
    #intersection = np.sum(np.abs(y_true * y_pred), axis=(1, 2, 3))
    #union = np.sum(y_true, (1, 2, 3)) + np.sum(y_pred, (1, 2, 3))
    #iou = np.mean((intersection + smooth) / (union + smooth), axis=0)
        
    return iou


def iou_weighted(y_true, y_pred, smooth=1):
    """
    Calculate intersection over union (IoU) between images.
    Input shape : Batch x Height x Width x #Classes (BxHxWxN), N = 1 in our binary classification case
    Input value : 0 - w/o melting layer or 1 - with melting layer
    Using weighted mean as reduction type for batch values
    with weightings determined by the existence of melting layer
    """
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
    union = union - intersection
    iou_seq = (intersection + smooth) / (union + smooth)  # shape: B * 1
    #idtrue  = K.sum(y_true, [1, 2, 3])
    weight = tf.where(K.sum(y_true, [1, 2, 3])>0, 100*tf.ones_like(K.sum(y_true, [1, 2, 3])), tf.ones_like(K.sum(y_true, [1, 2, 3])))
    weight = weight / K.sum(weight, axis=0)
    # calculate the intersection over union index
    iou_weighted = K.sum(iou_seq * weight, axis=0)
    
    
    #y_true = y_true.astype(np.int8)
    #y_pred = y_pred.astype(np.int8)    
    #intersection = np.sum(np.abs(y_true * y_pred), axis=(1, 2, 3))
    #union = np.sum(y_true, (1, 2, 3)) + np.sum(y_pred, (1, 2, 3))
    #union = union - intersection
    #iou_seq = (intersection + smooth) / (union + smooth)  # shape: B * 1
    #weight = np.ones(iou_seq.shape)
    #weight[np.sum(y_true, (1, 2, 3))> 0] = 10
    #weight = weight / np.sum(weight, axis=0)
    ## calculate the intersection over union index
    #iou_weighted = np.sum(iou_seq * weight, axis=0)
        
    return iou_weighted



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
    dice_score = K.mean((2 * intersection ) / (union + smooth), axis=0)
    
    
    #y_true = y_true.astype(np.int8)
    #y_pred = y_pred.astype(np.int8)
    #intersection = np.sum(y_true * y_pred, axis=(1, 2, 3))
    #union = np.sum(y_true, axis=(1, 2, 3)) + np.sum(y_pred, axis=(1, 2, 3))
    #dice_score = np.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    
    return dice_score



def true_positive_rate(y_true, y_pred, smooth=1.e-9):
    """
    Calculate the true positive rate: TP / (TP + FP)
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    
    tp_rate = K.sum( K.flatten(y_true) * K.flatten(y_pred) ) / ( K.sum(y_true) + smooth)
        
    return tp_rate


#%% Define the custom loss function
def bce_dice(y_true, y_pred):
    """
    Calculate the custom loss function
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Use sum as reduction type for batch values
    """
    #y_true = tf.convert_to_tensor(y_true, tf.float32)
    #y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    #y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    
    bce = keras.losses.binary_focal_crossentropy(y_true, y_pred, apply_class_balancing = True, alpha = 0.75, gamma = 0, from_logits=False)
    bce_dice = bce - K.log(dice_coef(y_true, y_pred))
    
    return bce_dice

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    dice = numerator / (denominator + tf.keras.backend.epsilon())
    return 1 - dice



#%% Define a function to create masks based on predicted data
def create_mask(data_predict):
    pred_mask = np.zeros(data_predict.shape)
    pred_mask[data_predict>0.5] = 1   # Probability larger than 0.5
    
    return pred_mask
    
#%% Define a function for visualization
def display(iepoch, data_input, data_target, data_predict, filepath, modelstr):
    data_mask = create_mask(data_predict)
    for icase in range(data_input.shape[0]):
        plt.figure(figsize=(16,3.5), dpi=200)
        plt.suptitle('Sample #' + '{:02d}'.format(icase+1) + ' @ Epoch #'+ '{:02d}'.format(iepoch), fontsize=8)   
        
        plt.subplot(1,4,1)
        pc1 = plt.pcolormesh(np.arange(0,256), np.arange(0,256)/256*8, data_input[icase,:,:,2],cmap = cm.RdBu_r, vmin=-0.5, vmax=0.5)
        plt.xlabel('Time (seconds)', fontsize=8)
        plt.ylabel('Height AGL (km)', fontsize=8)
        plt.title('Input standardized dVd', fontsize=8)
        cb1 = plt.colorbar(pc1)
        cb1.ax.tick_params(labelsize=7)
        
        plt.subplot(1,4,2)
        pc2 = plt.pcolormesh(np.arange(0,256), np.arange(0,256)/256*8, data_target[icase,:,:,0], vmin=0, vmax=1)
        plt.xlabel('Time (seconds)', fontsize=8)
        #plt.ylabel('Height AGL (km)', fontsize=8)
        plt.title('True Mask', fontsize=8)
        cb2 = plt.colorbar(pc2)
        cb2.ax.tick_params(labelsize=7)
        
        plt.subplot(1,4,3)
        pc3 = plt.pcolormesh(np.arange(0,256), np.arange(0,256)/256*8, data_predict[icase,:,:,0], vmin=0.3, vmax=0.7)
        plt.xlabel('Time (seconds)', fontsize=8)
        #plt.ylabel('Height AGL (km)', fontsize=8)
        plt.title('Predicted Prob', fontsize=8)
        cb3 = plt.colorbar(pc3)
        cb3.ax.tick_params(labelsize=7)
        
        plt.subplot(1,4,4)
        pc4 = plt.pcolormesh(np.arange(0,256), np.arange(0,256)/256*8, data_mask[icase,:,:,0], vmin=0, vmax=1)
        plt.xlabel('Time (seconds)', fontsize=8)
        #plt.ylabel('Height AGL (km)', fontsize=8)
        plt.title('Predicted Mask', fontsize=8)
        cb4 = plt.colorbar(pc4)
        cb4.ax.tick_params(labelsize=7)
        
        plt.savefig(filepath+modelstr+'_epoch'+'{:02d}'.format(iepoch)+'_samp'+'{:02d}'.format(icase+1)+'.png')
        
        plt.close()
        
        
#%% Define the UNet Model
def unet(input_size):

    # define the input layer
    input_layer = keras.Input(shape = input_size)
    
    # define encoder layers (Downward)
    e1 = keras.layers.Conv2D(64, 3, padding="same", activation="relu", strides=2)(input_layer)
    e1 = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(e1)
    e2 = keras.layers.Conv2D(128, 3, padding="same", activation="relu", strides=2)(e1)
    e2 = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(e2)
    e3 = keras.layers.Conv2D(256, 3, padding="same", activation="relu", strides=2)(e2)
    e3 = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(e3)
    e4 = keras.layers.Conv2D(512, 3, padding="same", activation="relu", strides=2)(e3)
    e4 = keras.layers.Conv2D(512, 3, padding="same", activation="relu")(e4)
    #x = keras.layers.Dropput(0.01)(x)
    
    # define decoder layers (Upward)
    d4 = keras.layers.Conv2DTranspose(512, 3, padding="same", activation="relu")(e4)
    d3 = keras.layers.Conv2DTranspose(512, 3, padding="same", activation="relu", strides=2)(d4)
    # apply skip connection
    d3 = keras.layers.Concatenate()([d3, e3])
    d3 = keras.layers.Conv2DTranspose(256, 3, padding="same", activation="relu")(d3)
    d2 = keras.layers.Conv2DTranspose(256, 3, padding="same", activation="relu", strides=2)(d3)
    # apply skip connection
    d2 = keras.layers.Concatenate()([d2, e2])
    d2 = keras.layers.Conv2DTranspose(128, 3, padding="same", activation="relu")(d2)
    d1 = keras.layers.Conv2DTranspose(128, 3, padding="same", activation="relu", strides=2)(d2)
    # apply skip connection
    d1 = keras.layers.Concatenate()([d1, e1])
    y = keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu")(d1)
    y = keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu", strides=2)(y)
    #y = keras.layers.Dropout(0.01)(y)
    
    #define the output layer
    #activation function really important: softmax -> sigmoid significantly improve the model performance
    output_layer = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(y)
    
    #create the model
    model = keras.Model(input_layer, output_layer)
    
    return model



#%% implement the UNet model
history = History()
model_unet = unet(input_dim)
model_unet.summary()

# compile the UNet model
model_unet.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              #loss = keras.losses.BinaryCrossentropy(),
              #loss = keras.losses.BinaryFocalCrossentropy(
              #    apply_class_balancing = True,
              #    alpha = 0.75,
              #    gamma = 2,
              #    from_logits=False
              #    ),
              loss = dice_loss,
              metrics=[keras.metrics.BinaryAccuracy(name='biacc'),
                       #keras.metrics.TruePositives(name='tp'),
                       true_positive_rate,
                       dice_coef,
                       iou,
                       #iou_weighted
                       ])

# check the model performance before training
sample_predict = model_unet.predict(sample_input)
display(iepoch=0, data_input=sample_input, data_target=sample_target, data_predict=sample_predict, filepath=fpathout, modelstr='unet')


#%%
####### start a new wandb run to track this script #######
wandb.init(
    # set the wandb project where this run will be logged
    project = "test-project"
    )
##########################################################

trainloss_unet = []
trainacc_unet = []
valloss_unet = []
valacc_unet = []


for epoch in range(N_EPOCHS):
#for epoch in range(1):
    # define callback
    #callbacks_unet = [keras.callbacks.ModelCheckpoint(fpathout+'model_unet_epoch'+'{:02d}'.format(epoch+1)+'.keras'), WandbMetricsLogger(log_freq=1)]
    callbacks_unet = [history, WandbMetricsLogger(log_freq=1)]
    
    # train the model
    #history_unet = model_unet.fit(train_input, train_target, epochs=1, batch_size=N_BATCH, \
    #                    validation_data=(val_input, val_target),  shuffle=True, callbacks=callbacks_unet)
    model_unet.fit(train_input, train_target, epochs=1, batch_size=N_BATCH, \
                        validation_data=(val_input, val_target),  shuffle=True, callbacks=callbacks_unet)
    # concatenate the loss and accuray for training and valiation data
    trainloss_unet.append(history.history['loss'][0])
    trainacc_unet.append(history.history['biacc'][0])
    valloss_unet.append(history.history['val_loss'][0])
    valacc_unet.append(history.history['val_biacc'][0])
       
    # check/visualize the model performance at the end of epochs with certain frequency
    if (epoch+1)%2 == 0:
        # save the model
        model_unet.save(fpathout+'model_unet_epoch'+'{:02d}'.format(epoch+1)+'.keras')
        # visualize model performance
        sample_predict = model_unet.predict(sample_input)
        display(iepoch=epoch+1, data_input=sample_input, data_target=sample_target, data_predict=sample_predict, filepath=fpathout, modelstr='unet')
        

# finish the wandb run, necessary in notebooks
wandb.finish()

# save the model history output
with open(fpathout +'model_unet_history.npy', 'wb') as fout:
    np.save(fout, np.array(trainloss_unet), allow_pickle=True)
    np.save(fout, np.array(trainacc_unet), allow_pickle=True)
    np.save(fout, np.array(valloss_unet), allow_pickle=True)
    np.save(fout, np.array(valacc_unet), allow_pickle=True) 
    

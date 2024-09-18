#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:51:26 2024

Code to implement Monte Carlo Dropout (MC Dropout)
for additional information of U-Net model uncertainty

Note the dropout channels of the U-Net layers are in
inference mode during the training process, and only
actively drop input units in the testing dataset to 
implement the ensemble predictions.

@author: Yan Xie (yanxieyx@umich.edu)
"""

# load the modules
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np


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


def iou_test(y_true, y_pred, smooth=1.e-9):
    """
    Calculate intersection over union (IoU) between images for testing data.
    Input shape : Batch x Height x Width x #Classes (BxHxWxN), N = 1 in our binary classification case
    Input value : 0 - w/o melting layer or 1 - with melting layer
    Output shape: Batch of testing dataset x #Classes
    Output value: iou value for the testing data
    """
    y_true = y_true.astype(np.int8)
    y_pred = y_pred.astype(np.int8)   
    intersection = np.sum(np.abs(y_true * y_pred), axis=(1, 2, 3))
    union = np.sum(y_true, (1, 2, 3)) + np.sum(y_pred, (1, 2, 3))
    iou_test = (intersection+ smooth ) / (union + smooth)

    return iou_test


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



def dice_coef_test(y_true, y_pred, smooth=1.e-9):
    """
    Calculate dice coefficient: evaluate the similarity between two images for testing data
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Output shape: Batch x #Classes
    """    
    
    y_true = y_true.astype(np.int8)
    y_pred = y_pred.astype(np.int8)
    intersection = np.sum(y_true * y_pred, axis=(1, 2, 3))
    union = np.sum(y_true, axis=(1, 2, 3)) + np.sum(y_pred, axis=(1, 2, 3))
    dice_test = (2. * intersection + smooth) / (union + smooth)
    
    return dice_test



def contingency_table(yy_true, yy_pred):
    """
    Calculate the contingency table
    Input shape: Batch x #Classes; value: 1-0 index indicating w.-w/o melting layer
    Output shape: 4 x #Classes; value: a, b, c, d
    a: number of cases when the event is predicted and observed (True Positive)
    b: number of cases when the event is predicted but not observed (False Positive)
    c: number of cases when the event is not predicted but observed (False Negative)
    d: number of cases when the event is not predicted and not observed (True Negative)
    n: total case number, should equal to a+b+c+d
    """
    yy_true = yy_true.astype(np.int8)
    yy_pred = yy_pred.astype(np.int8)
    
    one = np.ones(yy_true.shape)
    
    a = np.sum( yy_true * yy_pred )
    b = np.sum( (one - yy_true) * yy_pred )
    c = np.sum( yy_true * (one - yy_pred) )
    d = np.sum( (one - yy_true) * (one - yy_pred) )
    
    n = yy_true.shape[0]
    if n == (a+b+c+d):
        return np.array([a,b,c,d])
    else:
        return np.array([-1,-1,-1,-1])
    
      

def success_ratio(y_true, y_pred, smooth=1.e-9):
    """
    Calculate the sucess ratio, or the true positive rat = TP / (TP + FP)
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    
    #SR = K.sum( K.flatten(y_true) * K.flatten(y_pred) ) / ( K.sum(y_true) + smooth)
    TP = K.sum( y_true * y_pred, axis=[1,2,3] )
    FP = K.sum( tf.subtract(1., y_true) * y_pred, axis=[1,2,3])
    
    SR = K.mean( TP/(TP+FP+smooth), axis=0 )
        
    return SR


def prob_of_detection(y_true, y_pred, smooth=1.e-9):
    """
    Calculate the probability of detection: TP / (TP + FN)
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    
    TP = K.sum( y_true * y_pred, axis=[1,2,3] )
    FN = K.sum( y_true * tf.subtract(1., y_pred), axis=[1,2,3])
    
    POD = K.mean( TP/(TP+FN+smooth), axis=0)
        
    return POD


def critical_success_index(y_true, y_pred, smooth=1.e-9):
    """
    Calculate the critical success index: TP / (TP + FP + FN)
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    
    TP = K.sum( y_true * y_pred, axis=[1,2,3] )
    FN = K.sum( y_true * tf.subtract(1., y_pred), axis=[1,2,3])
    FP = K.sum( tf.subtract(1., y_true) * y_pred, axis=[1,2,3])
    
    CSI = K.mean( TP/(TP+FN+FP+smooth), axis=0)
        
    return CSI


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


# Define a function to create masks based on predicted data
def create_mask(data_predict):
    pred_mask = np.zeros(data_predict.shape)
    pred_mask[data_predict>0.5] = 1   # Probability larger than 0.5
    
    return pred_mask


# Define the Unet Model in modules
def enblock(x, filters, dropout):
    x = keras.layers.Conv2D(filters=filters, kernel_size=(3,3), padding="same", strides=2)(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters=filters, kernel_size=(3,3), padding="same")(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(dropout)(x, training=True)
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
    x = keras.layers.Dropout(dropout)(x, training=True)
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

# learning rate of optimizer
LR = 5e-3

# number of filters
filters = 64

# Dropout rate
DROP_RATE = 0.2


###################  End of hyperparameters ################### 


# implement the UNet model
model_unet = unet(input_dim, 1, depth=DEPTH)
model_unet.summary()

# compile the UNet model
model_unet.compile(optimizer=keras.optimizers.RMSprop(learning_rate=LR),
              loss = dice_bce_loss,
              metrics=[keras.metrics.BinaryAccuracy(name='biacc'),
                       dice_coef,
                       iou,
                       ])

# load weights of the trained model
model_unet.load_weights(fpathin+'bestmodel.keras')


# Ensemble predictions from performing predictions for 50 times
N_ensem = 50
id_flag = 0    

for ii in range(0, N_ensem):
    ypre = model_unet.predict(test_input)
    if id_flag == 0:
        ypre_ensem = ypre[np.newaxis, :, :, :, :]
        id_flag = 1
    elif id_flag == 1:
        ypre_ensem = np.concatenate((ypre_ensem, ypre[np.newaxis, :, :, :, :]), axis=0)


# calculate the mean and standard error
ypre_mean = np.mean(ypre_ensem, axis=0)
ypre_std = np.std(ypre_ensem, axis=0)
# standard error SE = STD / sqrt(50)
ypre_se = ypre_std / np.sqrt(N_ensem)

# evaluation metrics based on the mean of predicitions
# binarize probability into mask
test_predict = create_mask(ypre_mean)

# transform each figure into an index: 1-with melting layer; 0-w/o melting layer
idx_predict = np.zeros((test_predict.shape[0],1))
idx_target = np.zeros((test_target.shape[0],1))

# as long as one pixel w. melting layer exists, the target figure is deemed as w. melting layer
nml_target = np.sum(test_target, axis=(1,2))
idx_target[ np.where(nml_target>0)[0] ] = 1 

# for prediction figure, it needs to have more than 40 pixels detected with
# melting layer occurrence to be considered a reliable positive detection
nml_predict = np.sum(test_predict, axis=(1,2))
idx_predict[ np.where(nml_predict>40)[0] ] = 1

# now compute metrics
testing_iou = iou_test(test_target, test_predict)
testing_dice = dice_coef_test(test_target, test_predict)
testing_table = contingency_table(idx_target, idx_predict)

if np.sum(testing_table) > 0:
    a = testing_table[0]   # a; true positive
    b = testing_table[1]   # b; false positive
    c = testing_table[2]   # c; false negative
    d = testing_table[3]   # d; true negative
    
    # success ratio for testing data - TP / (TP + FP)
    testing_SR = a / (a + b + 1.e-9)
    # probability of detection for testing data - TP / (TP + FN)
    testing_PD = a / (a + c + 1.e-9)
    # critical success index - TP / (TP + FP + FN)
    testing_CSI = a / (a + b + c + 1.e-9)
    # heidke skill score - 2*(a*d - b*c) / ( (a+c)*(c+d) + (a+b)*(b+d) )  
    n = a + b + c + d
    temp = ( (a+b)*(a+c) + (b+d)*(c+d) ) / (n**2)
    testing_HSS = ( (a+d)/n - temp ) / ( 1 - temp + 1.e-9)
    
    print("\n a (True Positive) Count: " + '{:.0f}'.format(a))
    print("\n b (False Positive) Count: " + '{:.0f}'.format(b))
    print("\n c (False Negative) Count: " + '{:.0f}'.format(c))
    print("\n d (True Negative) Count: " + '{:.0f}'.format(d))
    print("\n Success Ratio: " + '{:.4f}'.format(testing_SR))
    print("\n Probability of Detection: " + '{:.4f}'.format(testing_PD))
    print("\n Critical Success Index: " + '{:.4f}'.format(testing_CSI))
    print("\n Heidke Skill Score: " + '{:.4f}'.format(testing_HSS))
    print("\n Mean IOU: " + '{:.4f}'.format(np.mean(testing_iou)))
    print("\n Mean Dice Coef.: " + '{:.4f}'.format(np.mean(testing_dice)))



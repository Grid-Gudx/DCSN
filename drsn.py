# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:32:06 2021

@author: gdx
"""

"""
refer from:

Created on Sat Dec 28 23:24:05 2019
Implemented using TensorFlow 1.0.1 and Keras 2.2.1
 
M. Zhao, S. Zhong, X. Fu, et al., Deep Residual Shrinkage Networks for Fault Diagnosis, 
IEEE Transactions on Industrial Informatics, 2019, DOI: 10.1109/TII.2019.2943898
There might be some problems in the Keras code. The weights in custom layers of models created using the Keras functional API may not be optimized.
https://www.reddit.com/r/MachineLearning/comments/hrawam/d_theres_a_flawbug_in_tensorflow_thats_preventing/

https://cloud.tencent.com/developer/news/661458
The TFLearn code is recommended for usage.
https://github.com/zhao62/Deep-Residual-Shrinkage-Networks/blob/master/DRSN_TFLearn.py
@author: super_9527
"""

#from __future__ import print_function
import tensorflow.keras as keras
#import keras
import numpy as np
from numpy.random import seed 
seed(13)
import tensorflow as tf
tf.random.set_seed(14)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda

def abs_backend(inputs):
    return K.abs(inputs)

def expand_dim_backend(inputs):
    return K.expand_dims(K.expand_dims(inputs,1),1)

def sign_backend(inputs):
    return K.sign(inputs)

def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels)//2
    inputs = K.expand_dims(inputs,-1)
    inputs = K.spatial_3d_padding(inputs, ((0,0),(0,0),(pad_dim,pad_dim)), 'channels_last')
    return K.squeeze(inputs, -1)

def cnn_block(x, out_channels):
     x = Conv2D(out_channels, 3, padding='same', kernel_initializer='he_normal',activation='relu')(x)     
     x = Conv2D(out_channels, 3, padding='same', kernel_initializer='he_normal',activation='relu')(x)     
     x = MaxPooling2D(pool_size=(2, 2))(x)
     return x

# channel-wise thresholds
def cw_threshold(inputs):
    # x W*H*C
    [w, h, channels] = inputs.get_shape().as_list()[1:]
    
    abs_x = Lambda(abs_backend)(inputs) # W*H*C
    abs_mean = GlobalAveragePooling2D()(abs_x) # C    
    x = Dense(channels, use_bias=False)(abs_mean)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(channels, activation='sigmoid', use_bias=False)(x)    
    x = keras.layers.multiply([abs_mean, x]) # C
    
    thres = Lambda(expand_dim_backend)(x)  
    thres = K.tile(thres, (1,w,h,1)) # W*H*C
    
    # Soft thresholding
    sub = keras.layers.subtract([abs_x, thres])  
    zeros = keras.layers.subtract([sub, sub])
    n_sub = keras.layers.maximum([sub, zeros]) 
    out = keras.layers.multiply([Lambda(sign_backend)(inputs), n_sub])
    return out

# channel-shared thresholds
def cs_threshold(inputs):
    # x W*H*C
    [w, h, channels] = inputs.get_shape().as_list()[1:]
    
    abs_x = Lambda(abs_backend)(inputs) # W*H*C
    abs_mean = GlobalAveragePooling2D()(abs_x) # C 
    x = Dense(channels, use_bias=False)(abs_mean)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1, activation='sigmoid', use_bias=False)(x)  # 1
    abs_mean_x = K.expand_dims(K.mean(abs_mean,axis=1), 1)
    thres = keras.layers.multiply([abs_mean_x, x]) # 1
    
    thres = Lambda(expand_dim_backend)(x) #1*1*1
    thres = K.tile(thres, (1,w,h,channels)) # W*H*C
    
    # Soft thresholding   
    sub = keras.layers.subtract([abs_x, thres])  
    zeros = keras.layers.subtract([sub, sub])
    n_sub = keras.layers.maximum([sub, zeros]) 
    out = keras.layers.multiply([Lambda(sign_backend)(inputs), n_sub])
    return out


def creat_model(input_shape, class_num, model_type):
    inputs = Input(shape=input_shape)
    x = inputs
    
    if model_type=='cnn':
        # deep convolution 
        x = cnn_block(x, 8)
        x = cnn_block(x, 16)
        x = cnn_block(x, 32)
    elif model_type=='dcsn_cs':   
        # deep convolution shrinkage networks with channel-shared thresholds: DCSN-CS
        x = cnn_block(x, 8)
        x = cs_threshold(x)
        x = cnn_block(x, 16)
        x = cs_threshold(x)
        x = cnn_block(x, 32)
        x = cs_threshold(x)
    else:    
        # deep convolution shrinkage networks with channel-wise thresholds: DCSN-CW
        x = cnn_block(x, 8)
        x = cw_threshold(x)
        x = cnn_block(x, 16)
        x = cw_threshold(x)
        x = cnn_block(x, 32)
        x = cw_threshold(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(class_num,activation='softmax')(x)   
    model = Model(inputs=inputs, outputs=x)    
    return model

def data_load(noise_add=False):
    # Input image dimensions
    img_rows, img_cols = 28, 28
    
    # The data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    if noise_add:
        # Noised data
        x_train = x_train.astype('float32') / 255. + 0.5*np.random.random(x_train.shape)
        x_test = x_test.astype('float32') / 255. + 0.5*np.random.random(x_test.shape)
    
    # # Visualize the samples
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
    # ax = ax.flatten()
    # for i in range(4):
    #     img = x_train[i]
    #     ax[i].imshow(img,cmap='Greys')
    #     #ax[i].imshow(img)
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # plt.tight_layout()
    # plt.show()
    
    # input_shape convert   
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
        
    return input_shape, x_train, y_train, x_test, y_test

def train(input_shape, x_train, y_train, batch_size, epoch, model_type, model_path):
    K.set_learning_phase(1) #train
    model = creat_model(input_shape, class_num=10, model_type=model_type)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])   
    model.summary()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epoch,
              validation_split=0.2)
    model.save(model_path)

def test(x_test, y_test, model_path):   
    K.set_learning_phase(0) #test
    model = load_model(model_path)
    DRSN_train_score = model.evaluate(x_test, y_test, batch_size=500, verbose=0)
    print('Test loss:', DRSN_train_score[0])
    print('Test accuracy:', DRSN_train_score[1])

if __name__ == '__main__':
    input_shape, x_train, y_train, x_test, y_test = data_load(False)
    
    train(input_shape, x_train, y_train,
          batch_size=400,
          epoch=5,
          model_type='cnn',
          model_path='./model/dcsn_cs0.h5')
    
    # test(x_test, y_test,model_path='./model/dcsn_cw.h5')
    
    
# model = creat_model(input_shape, class_num=10)
# model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])   
# model.summary()
# model.fit(x_train, y_train,
#           batch_size=400,
#           epochs=5,
#           validation_split=0.2)
# model.save('./model/dcsn_cs.h5')

# K.set_learning_phase(0) #test
# DRSN_train_score = model.evaluate(x_test, y_test, batch_size=500, verbose=0)
# print('Train loss:', DRSN_train_score[0])
# print('Train accuracy:', DRSN_train_score[1])

# xx = tf.convert_to_tensor(np.random.rand(100,28,28,5))
# # x = cnn_block(xx, 8)
# # x = cnn_block(x, 16)
# # x = GlobalAveragePooling2D()(x)
# # x = Dense(10,activation='softmax')(x)
# x = cs_threshold(xx)
# print(x.get_shape())

'''
# Residual Shrinakge Block
def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    
    for i in range(nb_blocks):
        
        identity = residual
        
        if not downsample:
            downsample_strides = 1
        
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, strides=(downsample_strides, downsample_strides), 
                          padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
        
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
        
        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling2D()(residual_abs)
        
        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal', 
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)
        
        # Calculate thresholds
        thres = keras.layers.multiply([abs_mean, scales])
        
        # Soft thresholding
        sub = keras.layers.subtract([residual_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])
        
        # Downsampling using the pooL-size of (1, 1)
        if downsample_strides > 1:
            identity = AveragePooling2D(pool_size=(1,1), strides=(2,2))(identity)
            
        # Zero_padding to match channels
        if in_channels != out_channels:
            identity = Lambda(pad_backend, arguments={'in_channels':in_channels,'out_channels':out_channels})(identity)
        
        residual = keras.layers.add([residual, identity])
    
    return residual


# define and train a model
inputs = Input(shape=input_shape)
net = Conv2D(8, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
net = residual_shrinkage_block(net, 1, 8, downsample=True)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = GlobalAveragePooling2D()(net)
outputs = Dense(10, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(net)
model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# model.fit(x_train, y_train, 
#           batch_size=200, 
#           epochs=5, verbose=1, 
#           validation_split=0.2,
#           shuffle=False)

#model.save('./drsn_struct.h5')

# K.set_learning_phase(0) #test
# DRSN_train_score = model.evaluate(x_train, y_train, batch_size=200, verbose=0)
# print('Train loss:', DRSN_train_score[0])
# print('Train accuracy:', DRSN_train_score[1])

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1, validation_data=(x_test, y_test))

# get results
K.set_learning_phase(0) #test
DRSN_train_score = model.evaluate(x_train, y_train, batch_size=100, verbose=0)
print('Train loss:', DRSN_train_score[0])
print('Train accuracy:', DRSN_train_score[1])
DRSN_test_score = model.evaluate(x_test, y_test, batch_size=100, verbose=0)
print('Test loss:', DRSN_test_score[0])
print('Test accuracy:', DRSN_test_score[1])
'''
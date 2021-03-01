import numpy as np 
import tensorflow as tf
from tensorflow.python.keras import utils
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import (Dense, Dropout, Flatten, Conv2D, MaxPooling2D,
                                            BatchNormalization, Input, concatenate, GlobalAveragePooling2D,
                                           Convolution2D,Activation,AveragePooling2D, concatenate, GRU, Permute, Reshape)

LOSS = 'categorical_crossentropy'
OPTIMIZER = 'adam'
METRICS = ['accuracy']

def get_model(model_name,input_shape,num_classes):
    """
    Will load a compiled model. Names and details described below. Input shape should be (x,y,depth) tuple
    
    multi_scale_level_cnn_model ---  exact code from https://github.com/CaifengLiu/music-genre-classification/blob/master/GTZAN_2048_1/model/my_model.ipynb
    
    bottom_up_broadcast_model  ---   my interpretation of the above model

    bottom_up_broadcast_crnn_model   ---  a combination of bototm up broadcast model and crnn

    simple_cnn   --    a simple/vanilla cnn

    """

    model_dict = {"multi_scale_level_cnn_model":multi_scale_level_cnn_model,
                  "bottom_up_broadcast_model":bottom_up_broadcast_model,
                  "bottom_up_broadcast_crnn_model":bottom_up_broadcast_crnn_model,
                  "simple_cnn":simple_cnn
                  }
    model_func = model_dict[model_name]
    model = model_func(input_shape=input_shape, num_classes= num_classes)
    
    return model

########################################################################################################
"""
Support Functions
"""
########################################################################################################


#Support Function for multi_scale_level_cnn
def base_conv_block(num_conv_filters, kernel_size):
    def f(input_):
        x = BatchNormalization()(input_)
        x = Activation('relu')(x)
        out = Convolution2D(num_conv_filters, kernel_size, padding='same')(x)
        return out
    return f

#Support Function for multi_scale_level_cnn
def multi_scale_block(num_conv_filters):
    def f(input_):
        branch1x1 = base_conv_block(num_conv_filters, 1)(input_)
        
        branch3x3 = base_conv_block(num_conv_filters, 1)(input_)  
        branch3x3 = base_conv_block(num_conv_filters, 3)(branch3x3)  
  
        branch5x5 = base_conv_block(num_conv_filters, 1)(input_)  
        branch5x5 = base_conv_block(num_conv_filters, 5)(branch5x5) 
  
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input_)  
        branchpool = base_conv_block(num_conv_filters, 1)(branchpool) 
        
        out = concatenate([branch1x1,branch3x3,branch5x5,branchpool], axis=-1)
        return out
    return f

#Support Function for multi_scale_level_cnn
def dense_block(num_dense_blocks, num_conv_filters):
    def f(input_):
        x = input_
        for _ in range(num_dense_blocks):
            out = multi_scale_block(num_conv_filters)(x)
            x = concatenate([x, out], axis=-1)
        return x
    return f

#Support Function for multi_scale_level_cnn
def transition_block(num_conv_filters):
    def f(input_):
        x = BatchNormalization()(input_)
        x = Activation('relu')(x)
        x = Convolution2D(num_conv_filters, 1)(x)
        out = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return out
    return f

#Support Function for broadcast_module and rcnn_module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out, n):
    # 1x1 conv
    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu',name=f'1x1_conv__{n}')(BatchNormalization(axis=-1)(layer_in))
    # 3x3 conv
    conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu',)(BatchNormalization(axis=-1)(layer_in))
    conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu',name=f'3x3_conv__{n}')(BatchNormalization(axis=-1)(conv3))
    # 5x5 conv
    conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(BatchNormalization(axis=-1)(layer_in))
    conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu',name=f'5x5_conv__{n}')(BatchNormalization(axis=-1)(conv5))
    # 3x3 max pooling
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(BatchNormalization(axis=-1)(pool))
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out
 
def broadcast_module(input_tensor,input_shape):
    """
    Will run input tensor through broadcast module as described in:
    
    """
    paddings = tf.constant([[0, 0],[1,1],[1,1],[0,0]])
    padded_input = tf.pad(input_tensor,paddings,"CONSTANT")
    x = Conv2D(32,(3,3),activation='relu',input_shape=input_shape)(padded_input)
    x = BatchNormalization(axis=-1)(x)
    incept_1_input = MaxPooling2D(pool_size=(1,4),name='incept_1_input')(x)

    f1 = 32
    f2_in = 64
    f2_out = 32
    f3_in = 16
    f3_out = 32
    f4_out = 32

    #Inception Module
    incept_1 = inception_module(incept_1_input, f1, f2_in, f2_out, f3_in, f3_out, f4_out,1)

    incept_2_input = concatenate([incept_1,incept_1_input],name='incept_2_input')
    incept_2 = inception_module(incept_2_input, f1, f2_in, f2_out, f3_in, f3_out, f4_out,2)

    incept_3_input = concatenate([incept_2_input,incept_2],name='incept_3_input')
    incept_3 = inception_module(incept_3_input, f1, f2_in, f2_out, f3_in, f3_out, f4_out,3)

    incept_3_output = concatenate([incept_3,incept_3_input],name='incept_3_output')

    #Transition Layers
    x = BatchNormalization(axis=-1)(Conv2D(32,(1,1))(incept_3_output))
    x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
    x = GlobalAveragePooling2D()(x)

    return x   

def crnn_module(input_tensor):
    """
    will return rcnn module as seen in:
    https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/music_tagger_crnn.py
    """
    nb_layers = 4  # number of convolutional layers
    nb_filters = [64, 128, 128, 128]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area


    input_shape = input_tensor.shape
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3


    bn_1 = BatchNormalization(axis=frequency_axis, input_shape=input_shape)(input_tensor)
    conv_1 = Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape,activation='elu')(bn_1)

    bn_2 = BatchNormalization(axis=channel_axis)(conv_1)
    mp_1 = MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0])(bn_2)
    drop_n = Dropout(0.1)(mp_1)

    for layer in range(nb_layers - 1):
        conv_n = Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same', activation = 'elu')(drop_n)
        bn_n = BatchNormalization(axis=channel_axis)(conv_n)
        mp_n = MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1])(bn_n)
        drop_n = Dropout(0.1)(mp_n)

    perm = Permute((time_axis, frequency_axis, channel_axis))(drop_n)
    resize_shape = drop_n.shape[1] * drop_n.shape[3]
    reshaped = Reshape((drop_n.shape[2], resize_shape))(perm)

    gru_1 = GRU(32, return_sequences=True)(reshaped)
    gru_2 = GRU(32, return_sequences=False)(gru_1)
    output = Dropout(0.3)(gru_2)
    
    return output


########################################################################################################
"""
Models
"""
########################################################################################################

def multi_scale_level_cnn_model(input_shape, num_classes, num_dense_blocks=3, num_conv_filters=32):
    model_input = Input(shape=input_shape)
    
    x = Convolution2D(num_conv_filters, 3, padding='same')(model_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x) #changed from 4,1 to 1,4 
    
    x = dense_block(num_dense_blocks, num_conv_filters)(x)
    x = transition_block(num_conv_filters)(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    
    model_output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=model_input, outputs=model_output)
    
    model.compile(loss=LOSS,
                    optimizer=OPTIMIZER,
                    metrics=METRICS)

    return model

def bottom_up_broadcast_model(input_shape,num_classes):
    visible = Input(shape=input_shape)
    output = broadcast_module(visible,input_shape)
    dense_out = Dense(num_classes, activation = 'softmax')(output)

    model = Model(inputs=visible, outputs = dense_out)
    model.compile(loss=LOSS,
                    optimizer=OPTIMIZER,
                    metrics=METRICS)

    return model

def bottom_up_broadcast_crnn_model(input_shape,num_classes):

    visible = Input(shape=input_shape)
    rcnn_res = crnn_module(visible)
    broadcast_res = broadcast_module(visible, input_shape)
    both_modules = concatenate([rcnn_res,broadcast_res])
    out = Dense(num_classes, activation="softmax")(both_modules)

    model = Model(inputs=visible, outputs=out)

    model.compile(loss=LOSS,
                optimizer=OPTIMIZER,
                metrics=METRICS)

    return model

def simple_cnn(input_shape,num_classes):

    cnn_model = Sequential(name='cnn_simple')
    # Adding convolutional layer
    cnn_model.add(Conv2D(filters=16,
                        kernel_size=(3,3),
                        activation='relu',
                        input_shape=input_shape))

    # Adding max pooling layer
    cnn_model.add(MaxPooling2D(pool_size=(2,4)))

    # Adding convolutional layer
    cnn_model.add(Conv2D(filters=32,
                        kernel_size=(3,3),
                        activation='relu'))

    # Adding max pooling layer
    cnn_model.add(MaxPooling2D(pool_size=(2,4)))

    # Adding a flattened layer to input our image data
    cnn_model.add(Flatten())

    # Adding a dropout layer for regularization
    cnn_model.add(Dropout(0.5))

    # Adding a dense layer with 64 neurons
    cnn_model.add(Dense(64, activation='relu'))

    # Adding an output layer
    cnn_model.add(Dense(num_classes, activation='softmax'))

    # Compiling our neural network
    cnn_model.compile(loss=LOSS,
                    optimizer=OPTIMIZER,
                    metrics=METRICS)

    return cnn_model
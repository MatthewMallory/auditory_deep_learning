import os
import s3fs
import json
import numpy as np 
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (Dense, Dropout, Flatten, Conv2D, MaxPooling2D,
                                            BatchNormalization, Input, concatenate, GlobalAveragePooling2D,
                                           Convolution2D,Activation,AveragePooling2D)
from tensorflow.python.keras.layers import Input, Dense, Flatten, Lambda, Dropout, Activation, LSTM, GRU, \
        TimeDistributed, Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D, \
        BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, \
        ZeroPadding2D, Reshape, merge, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D


from tensorflow.python.keras.layers.local import LocallyConnected1D
from tensorflow.python.keras.layers.advanced_activations import ELU
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model  


from tensorflow.python.keras import utils
import json
import tensorflow as tf
import s3fs

LOCATION = "aws"
MODEL = "multi_scale_level_cnn_model"
BATCH_SIZE = 8
EPOCHS = 20
INPUT_IMAGE_SHAPE = (647,128, 1)
DATASET = "gtzan_raw" # choose from ["fma_all_augmentations",    "gtzan_raw"      , "gtzan_augmented_without_dropout"     , "gtzan_augmented_with_dropout"   ]

def main():
    print("Loading Data From {}...".format(LOCATION))
    if LOCATION == 'local':
        data = np.load(f"/home/matt/audio_deep_learning/Data/{DATASET}/Data.npy")
        labels = np.load(f"/home/matt/audio_deep_learning/Data/{DATASET}/Labels.npy")

        pth = "/home/matt/audio_deep_learning/Data/All_DataSets_Genre_Labels.json"
        with open(pth, "r") as l:
            numerical_labels = json.load(l)[DATASET]


    else:
        fs = s3fs.S3FileSystem() 
        with fs.open(f"spectrogramdatabucket/{DATASET}/Labels.npy") as f:
            labels = np.load(f)
        with fs.open(f"spectrogramdatabucket/{DATASET}/Data.npy") as d:
            data = np.load(d)

        pth = "spectrogramdatabucket/All_DataSets_Genre_Labels.json" 
        with fs.open(pth,"r") as l:
            numerical_labels = json.load(l)[DATASET]


    num_classes = len(numerical_labels.keys())
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1

    epochs = 100
    batch_size = 8
    lr = 0.01
    k_fold = 10


    if not os.path.exists("Outdir"):
        os.mkdir("Outdir")

    odir = "Outdir/{}_{}".format(MODEL,DATASET)
    if not os.path.exists(odir):
        os.mkdir(odir)

    for i in range(k_fold):
            
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(data, labels, train_size=train_size, 
                                                                        val_size=val_size, test_size=test_size)

        print("Training Shape: {} ... {}".format(X_train.shape,y_train.shape))
        print("Testing Shape: {} ... {}".format(X_test.shape,y_test.shape))
        print("Validation Shape: {} ... {}".format(X_val.shape,y_val.shape))

        loss_acc_plot_path = os.path.join(odir,"{}_kfold_{}_accuracy_loss_plots.pdf".format(MODEL,i))
        confusion_mat_plot_outpath = os.path.join(odir,"{}_kfold_{}_validation_confusion_matrix_heatmap.pdf".format(MODEL,i))
        confusion_mat_outpath = os.path.join(odir,"{}_kfold_{}_validation_confusion_matrix.npy".format(MODEL,i))

        file_name = os.path.join(odir,"{}_{}_kfold_{}.hdf5".format(MODEL,DATASET,i))

        lr_change = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=0.000)
        early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=10, mode='min')
        model_checkpoint = ModelCheckpoint(file_name, monitor='val_accuracy', save_best_only=True, mode='max')

        callbacks =[lr_change, model_checkpoint, early_stopping]
        opt = Adam(lr=lr)
        model = multi_scale_level_cnn(input_shape=(data.shape[1], data.shape[2], data.shape[3]), 
                                num_dense_blocks=3, num_conv_filters=32, num_classes=num_classes)
        model.compile(
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    optimizer=opt)

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                validation_data=(X_val, y_val), verbose=1,
                callbacks=callbacks)
        model_best = load_model(file_name)
        train_loss, train_acc = model_best.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
        val_loss, val_acc = model_best.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
        test_loss, test_acc = model_best.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)


        print("Final Results:")
        print(f"Train Loss, Train Acc = {train_loss}, {train_acc}")
        print(f"Test Loss, Test Acc = {test_loss}, {test_acc}")
        print(f"Val Loss, Val Acc = {val_loss}, {val_acc}")


        # #Save model
        # tf.keras.models.save_model(model_best,
        #                             "{}/{}_model_output_".format(odir,MODEL), 
        #                             overwrite=True,
        #                             include_optimizer=True)

        #Save plots
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_x = range(1, len(acc) + 1)
        figy,axe = plt.subplots(2,1)
        axe[0].plot(epochs_x, acc, 'bo', label='Training acc')
        axe[0].plot(epochs_x, val_acc, 'b', label='Test acc')
        axe[0].set_title('Training and testing accuracy')
        axe[0].legend()
        axe[1].plot(epochs_x, loss, 'bo', label='Training loss')
        axe[1].plot(epochs_x, val_loss, 'b', label='Testing loss')
        axe[1].set_title('Training and testing loss ')
        axe[1].legend()
        axe[1].set_xlabel("Epoch")
        figy.set_size_inches(7,10)
        figy.savefig(loss_acc_plot_path,dpi=300,bbox_inches="tight")
        plt.clf()

        #Save confmat
        predictions_arr = model.predict(X_val, verbose=1)
        conf_matrix = confusion_matrix(np.argmax(y_val, 1), np.argmax(predictions_arr, 1),normalize='true')
        acc = accuracy_score(np.argmax(y_val, 1), np.argmax(predictions_arr, 1))
        print("Validation Accuracy = {}".format(acc))
        f,a = plt.gcf(),plt.gca()
        sns.heatmap(conf_matrix,vmax=1,annot=True)
        a.set_xticklabels(list(numerical_labels.keys()),rotation=90)
        a.set_yticklabels(list(numerical_labels.keys()),rotation=0)
        a.set_title("Validation Accuracy  = {}".format(round(100*acc,4)))
        f.set_size_inches(6,5)
        f.savefig(confusion_mat_plot_outpath,bbox_inches='tight',dpi=300)
        np.save(confusion_mat_outpath,conf_matrix)


def train_val_test_split(X, y, train_size, val_size, test_size):
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_size/(test_size + val_size), stratify=y_val_test)
    return X_train, y_train, X_val, y_val, X_test, y_test



def base_conv_block(num_conv_filters, kernel_size):
    def f(input_):
        x = BatchNormalization()(input_)
        x = Activation('relu')(x)
        out = Convolution2D(num_conv_filters, kernel_size, padding='same')(x)
        return out
    return f

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
#         out = base_conv_block(num_conv_filters, 1)(out)
        return out
    return f


def dense_block(num_dense_blocks, num_conv_filters):
    def f(input_):
        x = input_
        for _ in range(num_dense_blocks):
            out = multi_scale_block(num_conv_filters)(x)
            x = concatenate([x, out], axis=-1)
        return x
    return f


def transition_block(num_conv_filters):
    def f(input_):
        x = BatchNormalization()(input_)
        x = Activation('relu')(x)
        x = Convolution2D(num_conv_filters, 1)(x)
        out = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return out
    return f



def multi_scale_level_cnn(input_shape, num_dense_blocks, num_conv_filters, num_classes):
    model_input = Input(shape=input_shape)
    
    x = Convolution2D(num_conv_filters, 3, padding='same')(model_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)
    
    x = dense_block(num_dense_blocks, num_conv_filters)(x)
    x = transition_block(num_conv_filters)(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    
    model_output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=model_input, outputs=model_output)
    
    return model


if __name__ == "__main__":
    main()
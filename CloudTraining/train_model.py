# import s3fs
import numpy as np 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns

import json
import tensorflow as tf

from models.models import get_model

"""
model_options:

    multi_scale_level_cnn_model ---  exact code from https://github.com/CaifengLiu/music-genre-classification/blob/master/GTZAN_2048_1/model/my_model.ipynb
    
    bottom_up_broadcast_model  ---   my interpretation of the above model

    bottom_up_broadcast_crnn_model   ---  a combination of bototm up broadcast model and crnn

    simple_cnn   --    a simple/vanilla cnn

"""

MODEL = "simple_cnn"
BATCH_SIZE = 32
EPOCHS = 2

def load_data():
    print("Loading Data...")
    # fs = s3fs.S3FileSystem() 
    # with fs.open("spectrogramdatabucket/Spectrogram_Data_Labels.npy") as f:
    #     labels = np.load(f)
    # with fs.open("spectrogramdatabucket/Spectrogram_Data.npy") as d:
    #     data = np.load(d)


    data = np.load("/home/matt/audio_deep_learning/Data/FMA_Small_Spectrogram_Data.npy")
    labels = np.load("/home/matt/audio_deep_learning/Data/FMA_Small_Spectrogram_Data_Labels.npy")

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, shuffle=True)
    X_train = X_train.reshape(X_train.shape[0], 128, 647, 1)
    X_test = X_test.reshape(X_test.shape[0], 128, 647, 1)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.35, shuffle=True)

    print("Training Shape: {} ... {}".format(X_train.shape,y_train.shape))
    print("Testing Shape: {} ... {}".format(X_test.shape,y_test.shape))
    print("Validation Shape: {} ... {}".format(X_val.shape,y_val.shape))

    return X_train, y_train, X_test, y_test, X_val, y_val

def load_genre_label_dict():
    # fs = s3fs.S3FileSystem() 
    # pth = "spectrogramdatabucket/Genre_Track_Id_Dict.json" 
    # with fs.open(pth,"rb") as l:
    #     x = json.load(l)

    pth = "/home/matt/audio_deep_learning/Genre_Track_Id_Dict.json"
    with open(pth, "rb") as l:
        x = json.load(l)
    
    return x 


def main():

    odir = MODEL
    if not os.path.exists(odir):
        os.mkdir(odir)

    loss_acc_plot_path = os.path.join(odir,"{}_accuracy_loss_plots.pdf".format(MODEL))
    confusion_mat_plot_outpath = os.path.join(odir,"{}_validation_confusion_matrix_heatmap.pdf".format(MODEL))
    confusion_mat_outpath = os.path.join(odir,"{}_validation_confusion_matrix.npy".format(MODEL))

    numerical_labels = load_genre_label_dict()
    X_train, y_train, X_test, y_test, X_val, y_val = load_data()


    # X_train = X_train[0:1000,:,:,:]
    # y_train = y_train[0:1000]
    model = get_model(model_name = MODEL, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) )


    history = model.fit(X_train,
                        y_train, 
                        batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS)
    
    #Save model
    tf.keras.models.save_model(model,
                                "{}/{}_model_output".format(MODEL,MODEL), 
                                overwrite=True,
                                include_optimizer=True)

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

if __name__ == "__main__":
    main()
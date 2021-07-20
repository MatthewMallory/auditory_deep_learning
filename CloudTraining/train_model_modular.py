import os
import s3fs
import json
import numpy as np 
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score


from models.models import get_model
"""
model_options:

    multi_scale_level_cnn_model ---  exact code from https://github.com/CaifengLiu/music-genre-classification/blob/master/GTZAN_2048_1/model/my_model.ipynb
    
    bottom_up_broadcast_model  ---   my interpretation of the above model

    bottom_up_broadcast_crnn_model   ---  a combination of bototm up broadcast model and crnn

    simple_cnn   --    a simple/vanilla cnn

"""

LOCATION = "local"
MODEL = "multi_scale_level_cnn_model"
BATCH_SIZE = 8
EPOCHS = 20
INPUT_IMAGE_SHAPE = (647,128, 1)
DATASET = "gtzan_raw" # choose from ["fma_all_augmentations",    "gtzan_raw"      , "gtzan_augmented_without_dropout"     , "gtzan_augmented_with_dropout"   ]

def load_data(location,DATASET):
    print("Loading Data From {}...".format(location))
    if location == 'local':
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


    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, shuffle=True)
    X_train = X_train.reshape(X_train.shape[0], INPUT_IMAGE_SHAPE[0], INPUT_IMAGE_SHAPE[1],  INPUT_IMAGE_SHAPE[2])
    X_test = X_test.reshape(X_test.shape[0], INPUT_IMAGE_SHAPE[0],  INPUT_IMAGE_SHAPE[1], INPUT_IMAGE_SHAPE[2])

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.35, shuffle=True)

    print("Training Shape: {} ... {}".format(X_train.shape,y_train.shape))
    print("Testing Shape: {} ... {}".format(X_test.shape,y_test.shape))
    print("Validation Shape: {} ... {}".format(X_val.shape,y_val.shape))

    return X_train, y_train, X_test, y_test, X_val, y_val, numerical_labels


def main():

    if not os.path.exists("Outdir"):
        os.mkdir("Outdir")

    odir = "Outdir/{}_{}".format(MODEL,DATASET)
    if not os.path.exists(odir):
        os.mkdir(odir)

    loss_acc_plot_path = os.path.join(odir,"{}_accuracy_loss_plots.pdf".format(MODEL))
    confusion_mat_plot_outpath = os.path.join(odir,"{}_validation_confusion_matrix_heatmap.pdf".format(MODEL))
    confusion_mat_outpath = os.path.join(odir,"{}_validation_confusion_matrix.npy".format(MODEL))

    X_train, y_train, X_test, y_test, X_val, y_val,numerical_labels = load_data(LOCATION,DATASET)

    num_labels = len(numerical_labels.keys())
    model = get_model(model_name = MODEL, input_shape = INPUT_IMAGE_SHAPE, num_classes = num_labels)


    history = model.fit(X_train,
                        y_train, 
                        batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS)
    
    #Save model
    tf.keras.models.save_model(model,
                                "{}/{}_model_output".format(odir,MODEL), 
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
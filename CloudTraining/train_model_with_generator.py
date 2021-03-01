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
from utils.data_gen import groovy_data_generator
"""
model_options:

    multi_scale_level_cnn_model ---  exact code from https://github.com/CaifengLiu/music-genre-classification/blob/master/GTZAN_2048_1/model/my_model.ipynb
    
    bottom_up_broadcast_model  ---   my interpretation of the above model

    bottom_up_broadcast_crnn_model   ---  a combination of bototm up broadcast model and crnn

    simple_cnn   --    a simple/vanilla cnn

"""
TRAIN_LOCATION = "local"
MODEL = "multi_scale_level_cnn_model"
DATASET = "gtzan_raw" # choose from ["fma_all_augmentations",    "gtzan_raw"      , "gtzan_augmented_without_dropout"     , "gtzan_augmented_with_dropout"   ]
BATCH_SIZE = 32
INPUT_IMAGE_SHAPE = (128,647,1)
EPOCHS = 30


def main():
    if not os.path.exists("Outdir"):
        os.mkdir("Outdir")
    odir = "Outdir/{}_{}".format(MODEL,DATASET)
    #Create Outfile Variables
    if not os.path.exists(odir):
        os.mkdir(odir)

    loss_acc_plot_path = os.path.join(odir,"{}_accuracy_loss_plots.pdf".format(MODEL))
    confusion_mat_plot_outpath = os.path.join(odir,"{}_validation_confusion_matrix_heatmap.pdf".format(MODEL))
    confusion_mat_outpath = os.path.join(odir,"{}_validation_confusion_matrix.npy".format(MODEL))


    if TRAIN_LOCATION == "local":
        train_dir = "/home/matt/audio_deep_learning/Data/{}/TrainDir".format(DATASET)
        test_dir = "/home/matt/audio_deep_learning/Data/{}/TestDir".format(DATASET)
        val_dir = "/home/matt/audio_deep_learning/Data/{}/ValDir".format(DATASET)
        num_val_samples = len(os.listdir(val_dir))

        pth = "/home/matt/audio_deep_learning/Data/All_DataSets_Genre_Labels.json"
        with open(pth, "r") as l:
            numerical_labels = json.load(l)[DATASET]

    else:
        fs = s3fs.S3FileSystem() 
        train_dir = "spectrogramdatabucket/{}/TrainDir".format(DATASET)
        test_dir = "spectrogramdatabucket/{}/TestDir".format(DATASET)
        val_dir =  "spectrogramdatabucket/{}/ValDir".format(DATASET)
        num_val_samples = len(fs.ls(val_dir))

        pth = "spectrogramdatabucket/All_DataSets_Genre_Labels.json" 
        with fs.open(pth,"r") as l:
            numerical_labels = json.load(l)[DATASET]


    num_labels = len(numerical_labels.keys())
    train_generator = groovy_data_generator(image_dir=train_dir, batch_size=BATCH_SIZE, image_size = INPUT_IMAGE_SHAPE, num_unique_labels = num_labels, train_location=TRAIN_LOCATION)
    test_generator = groovy_data_generator(image_dir=test_dir, batch_size=BATCH_SIZE, image_size = INPUT_IMAGE_SHAPE, num_unique_labels = num_labels, train_location=TRAIN_LOCATION)

    model = get_model(model_name = MODEL, input_shape = INPUT_IMAGE_SHAPE, num_classes = num_labels)

    if TRAIN_LOCATION == 'local':
        history = model.fit(train_generator, 
                            batch_size=BATCH_SIZE,
                            validation_data=test_generator,
                            epochs=EPOCHS)
    else:
        #when using aws ML image, the TF version complains if you have a batch size argument while using a data generator.
        history = model.fit(train_generator,
                    validation_data=test_generator,
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
    X_val, y_val = groovy_data_generator(image_dir=val_dir, batch_size=num_val_samples, image_size = INPUT_IMAGE_SHAPE, num_unique_labels = num_labels, train_location=TRAIN_LOCATION).__getitem__(0)
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
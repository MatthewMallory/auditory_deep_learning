This repository aims to present a novel solution to music genre classification with deep learning. Through transfer learning we can use this neural network to organize a users' saved music library into discrete clusters. From here, it will be possible to use this method as a content based music recomendation system as compared to collaborative filtering methods. 

The neural network architecture developed here was influenced strongly by the work done by [Liu et al](https://link.springer.com/article/10.1007/s11042-020-09643-6). In this work a convolution network architecture combining [dense connectivity](https://arxiv.org/abs/1608.06993) and [inception blocks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf) was presented as a bottom up broadcast neural network. This network architecture takes multi-scale time-frequency information into consideration which creates semantic features for the decision layer to descriminate genre of an unknown music clip. The architecture network is shown below (Liu et al.)

![network architecture (c.)](https://github.com/MatthewMallory/auditory_deep_learning/blob/main/Figures/model.png)

The GTZAN dataset was used for model training and evaluation. This dataset consists of 1000 clips of 30 seconds of audio evenly distributed across 10 genres (classic, jazz, blues, metal, pop, rock, country, disco, hiphop and reggae). The audio clips were transformed into mel-spectrogram by applying a logarithmic scale to the frequency axis of the short time fourier transform. Librosa was used to ectract mel-spectrotgrams with 128 mel filters (bands) covering the frequency range 0-22050Hz, with a frame length of 2048 and hop size of 1024. The result is a 647 x 128 image.

With a considerably small dataset, data augmentation was used as described in [by Le et al.](https://arxiv.org/pdf/1904.08779.pdf) which directly modify the mel-spectrogram rather than audio modification approaches such as pitch shiftin. The figure below shows an example of these augmentation results. The notebook [AudioAugmentation](https://github.com/MatthewMallory/auditory_deep_learning/blob/main/Notebooks/AudioAugmentation.ipynb) in this repository hosts code for these augmentations and contains audio playback widgets to hear the effects of mel-spectrogram augmentation.

![augmentations](https://github.com/MatthewMallory/auditory_deep_learning/blob/main/Figures/augmentation.png)

The model was trained to minimize categorical cross-entropy between the predictions and truthful genre labels using ADAM optimizer. A batch size of 8 was used for 100 epochs in the training process. The initial learning rate was set to 0.01 and automatically decrease by a factor of 0.5 when the loss has stopped improving after 3 epochs. Training, testing and validation sets were randomly partitioned following 8-1-1 proportions. Data was hosted on AWS and training was carried out on an ec2 cluster. The figures below show the training and validation loss curves and validation confusion matrix. 

![training and loss curves](https://github.com/MatthewMallory/auditory_deep_learning/blob/main/Figures/acc_loss_plots.pdf)

![validation confusion matrix](https://github.com/MatthewMallory/auditory_deep_learning/blob/main/Figures/genre_conf_mat.pdf)



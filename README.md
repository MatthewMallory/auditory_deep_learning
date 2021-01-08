While our favorite song plays on the radio, the sound waves traveling through the air penetrate our external auditory canal and so begins the marvelous auditory processing pathway. At the begining of this display of incredible physiological complexities, we find hair cells within the cochlear responding to vibrations in the basilar membrane. Variations in hair cell stiffness and size enable them to respond to frequencies in our detectable range, 20-20,000 Hz. These variations are spatially organized, so, thick and short hair cells at the base are sensitive to high frequencies whereas thin and longer hair cells at the apex are sensitive to low frequencies. An activated hair cell begins the cascading of action potentials via type 1 spiral ganglion neurons taking both ipsilateral and contraleteral routes to the primary auditory cortex (A1) which, similar to our hair cells, is spatially organized by frequency. 

The auditory pathway, albeit complex and comprising of hundreds and thousands of cells working in unison, is tangible. We are able to study it. We understand the components and mechanisms driving this process. The same cannot be said for music preferences. Yet. 

This repository aims to explore deep learning in audio analysis, more specifically first to solve genre classification problems with neural network architectures. The next goal is to use the neural network's ability to seperate genres and apply this to my saved tracks and create unuspervised clusters. This will be one step closer to creating advanced, content based music recommendaiton systems. 

The broad goal is to create a music recommendation system that goes beyond collaborative filtering methods and introduces information learned directly from the audio. Acoustic features exist (seen in spotify kaggle set), but again this is an indirect representation of the audio in ambiguous feature space. I will use the fma_dataset which has 8 balanced genres and 1000 .mp3 files per genre. 



Outline Of Files in the Repository:
[1. Exploratory Data Analysis on Spotify Kaggle DataSet (separate from fma_dataset)](https://github.com/MatthewMallory/auditory_deep_learning/blob/main/Notebooks/Spotify_Exploritory_Data_Analysis.ipynb)

[2. Python Script to generate mel spectrogram from fma_dataset](https://github.com/MatthewMallory/auditory_deep_learning/blob/main/generate_spectrograms.py)

[3. Augmenting Audio Files and Mel spectrograms](https://github.com/MatthewMallory/auditory_deep_learning/blob/main/Notebooks/AudioAugmentation.ipynb)

[4. Baseline CNN without augmenting data](https://github.com/MatthewMallory/auditory_deep_learning/blob/main/Notebooks/Basic_CNN_small_dataset.ipynb)

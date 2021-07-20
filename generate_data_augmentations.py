from multiprocessing import Pool
import pandas as pd
import librosa
import tifffile
import numpy as np
from data_augmentation.augmentation import *
import os
import ntpath 

def create_spectrogram_multiprocessing(track_path, mel_db_path):

    if True:#try:
        audio_augmented = augment_audio(track_path)
        
        mel_db = tifffile.imread(mel_db_path)
        spectrogram_augmented = augment_spectrogram(mel_db)

        for aug_dict in [audio_augmented,spectrogram_augmented]:
            for aug_label,aug_mel_db in aug_dict.items():
                out_file = "{}_{}.tiff".format(mel_db_path[0:-5],aug_label)
                tifffile.imwrite(out_file, aug_mel_db)
                print(out_file)
    else:
        print("FAILED")
        return track_path

def main():
    print(librosa.__version__)
    parallel_inputs = []
    for genre in os.listdir("/home/matt/audio_deep_learning/Data/gtzan_augmented/WAV_Files/"):
        genre_dir = os.path.join("/home/matt/audio_deep_learning/Data/gtzan_augmented/WAV_Files", genre)
        

        outdir = "/home/matt/audio_deep_learning/Data/gtzan_augmented/Spectrograms/"+genre
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        if os.path.isdir(genre_dir):
            for track_id in [f for f in os.listdir(genre_dir) if ".wav" in f]:
                track_path = os.path.join(genre_dir, track_id)

                mel_db_filename = track_id.split(".wav")[0] + '_mel_spect_db.tiff'
                mel_db_path = os.path.join(outdir, mel_db_filename)

                if os.path.exists(mel_db_path):
                    parallel_inputs.append((track_path,mel_db_path))
    
    p = Pool()
    print("There are {} tracks to analyze".format(len(parallel_inputs)))

    failed_tracks = p.starmap(create_spectrogram_multiprocessing,parallel_inputs)
    if failed_tracks != []:
        pd.DataFrame({'tracks':failed_tracks}).to_csv("Failed Spectrogram.csv")
    else:
        print("Flawless")

if __name__ == "__main__":
    main()
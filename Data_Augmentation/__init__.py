from multiprocessing import Pool
import pandas as pd
import librosa
import tifffile
import numpy as np
import os

def create_spectrogram_multiprocessing(track_path, mel_path, mel_db_path):

    try:
        y, sr = librosa.load(track_path)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,
                                             hop_length=1024,
                                             win_length=None,
                                             window='hann',
                                             center=True, pad_mode='reflect',
                                             power=2.0) #,
        mel_db = librosa.power_to_db(mel, ref=np.max)
        tifffile.imwrite(mel_path, mel)
        tifffile.imwrite(mel_db_path, mel_db)
    except:
        print("FAILED")
        return track_path

def main():
    print(librosa.__version__)
    parallel_inputs = []
    for genre in os.listdir("mp3_files/"):
        genre_dir = os.path.join("mp3_files", genre)
        if os.path.isdir(genre_dir):
            for track_id in [f for f in os.listdir(genre_dir) if ".mp3" in f]:
                track_path = os.path.join(genre_dir, track_id)
                mel_filename = track_id.split(".")[0] + '_mel_spect.tiff'
                mel_db_filename = track_id.split(".")[0] + '_mel_spect_db.tiff'

                mel_path = os.path.join(genre_dir, mel_filename)
                mel_db_path = os.path.join(genre_dir, mel_db_filename)
                if (not os.path.exists(mel_path)) or (not os.path.exists(mel_db_path)):
                    parallel_inputs.append((track_path,mel_path,mel_db_path))
    
    p = Pool()
    print("There are {} tracks to analyze".format(len(parallel_inputs)))
    failed_tracks = p.starmap(create_spectrogram_multiprocessing,parallel_inputs)
    pd.DataFrame({'tracks':failed_tracks}).to_csv("Failed Spectrogram.csv")

if __name__ == "__main__":
    main()
from multiprocessing import Pool
import pandas as pd
import numpy as np
import tifffile
import librosa
import os

def create_spectrogram_multiprocessing(track_path, mel_db_path):

    try:
        y, sr = librosa.load(track_path)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,
                                             hop_length=1024,
                                             win_length=None,
                                             window='hann',
                                             center=True, pad_mode='reflect',
                                             power=2.0) #,
        mel_db = librosa.power_to_db(mel, ref=np.max)
        tifffile.imwrite(mel_db_path, mel_db)
    except:
        print("FAILED")
        return track_path

def main():
    print(librosa.__version__)
    parallel_inputs = []
    for genre in os.listdir(zz):
        genre_dir = os.path.join("/home/matt/audio_deep_learning/Data/gtzan_raw/WAV_Files", genre)
        outdir = "/home/matt/audio_deep_learning/Data/gtzan_raw/Spectrograms/"+genre
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if os.path.isdir(genre_dir):
            for track_id in [f for f in os.listdir(genre_dir) if ".wav" in f]:
                track_path = os.path.join(genre_dir, track_id)
                
                mel_db_filename = track_id.split(".wav")[0] + '_mel_spect_db.tiff'
                mel_db_path = os.path.join(outdir, mel_db_filename)
                if not os.path.exists(mel_db_path):
                    parallel_inputs.append((track_path,mel_db_path))
    
    p = Pool()
    print("There are {} tracks to analyze".format(len(parallel_inputs)))
    failed_tracks = p.starmap(create_spectrogram_multiprocessing,parallel_inputs)
    pd.DataFrame({'tracks':failed_tracks}).to_csv("Failed Spectrogram.csv")

if __name__ == "__main__":
    main()
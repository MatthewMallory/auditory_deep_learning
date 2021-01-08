import librosa
import numpy as np

def generate_mel_spect_db(y,sr):
    """
    y: audio array float32
    sr: smapling rate, int
    given audio and sampling rate, return db transformed mel spectrogram
    """

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,
                                            hop_length=1024,
                                            win_length=None,
                                            window='hann',
                                            center=True, pad_mode='reflect',
                                            power=2.0) #,
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    return mel_db

def get_db_mel_spect(y,sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,
                                         hop_length=1024,
                                         win_length=None,
                                         window='hann',
                                         center=True, pad_mode='reflect',
                                         power=2.0) #,
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


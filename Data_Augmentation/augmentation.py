import librosa
import numpy as np
import tensorflow_addons as tfa
import utils

def augment_audio(track_path,shift_val = 3):
    """
    input:
    track_path: path to audio file (str)
    shift_val: how many (fractional) half-steps to shift y (int)
    
    returns:
    dict (str:array)
    augmentation type and resulting mel spectrogram (log transformmed)

    """
    y, sr = librosa.load(track_path)

    noise = np.random.uniform(low=0.01, high=0.15, size=(len(y),)).astype(np.float32)
    noisey_y = y+noise
    pitch_shifted_pos_y = librosa.effects.pitch_shift(y, sr, shift_val)
    pitch_shifted_neg_y = librosa.effects.pitch_shift(y, sr, -shift_val)

    return {"noise_added":utils.generate_mel_spect_db(noisey_y,sr),
            "pitch_shifted_pos":utils.generate_mel_spect_db(pitch_shifted_pos_y,sr) ,
            "pitch_shifted_neg":utils.generate_mel_spect_db(pitch_shifted_neg_y,sr) 
            }
   
def augment_spectrogram(mel_spect):

    """
    input: mel_spect: 2d array of mel spectrogram
    returns 3 augmented versions of the mel spectrogram
    1. time and frequency masked
    2. sparse image warped mel spectrogram
    3. randomly dropout mel spectrogram.

    Examples of each of these can be found:
    https://github.com/MatthewMallory/auditory_deep_learning/blob/main/Notebooks/AudioAugmentation.ipynb 
    """

    results = {"time_frequency_masked":freq_mask(time_mask(mel_spect))
                "time_warped":warp_mel_spect(mel_spect)
                "random_dropout":random_dropout(mel_spect)
    }

    return results

def freq_mask(mel_spect,F=27):
    """
    input: mel_spect: 2d array of mel spectrogram
    F: max possible number of frequency channels to block
    Default value chosen from table 1 of:
    https://arxiv.org/pdf/1904.08779.pdf
    """
    f = int(np.random.uniform(5,F))
    mel_spect = mel_spect.copy()
    ht,wd = mel_spect.shape
    start_freq = np.random.choice(np.arange(0,ht-F))
    ending_freq = start_freq+f
    mel_spect[start_freq:ending_freq,:] = 0 
    return mel_spect
    
def time_mask(mel_spect,T=100):
    """
    input: mel_spect: 2d array of mel spectrogram
    T: max possible number of time steps to block
    Default value chosen from table 1 of:
    https://arxiv.org/pdf/1904.08779.pdf
    """
    t = int(np.random.uniform(10,T))
    mel_spect = mel_spect.copy()
    ht,wd = mel_spect.shape
    start_time = np.random.choice(np.arange(0,wd-T))
    ending_time = start_time+t
    mel_spect[:,start_time:ending_time] = 0 
    return mel_spect

def warp_mel_spect(mel_spect,W=80):
    """
    input: mel_spect: 2d array of mel spectrogram
    W: max possible warping distance window 
    Default value and method description seen in:
    https://arxiv.org/pdf/1904.08779.pdf

    """
    mel_spect = mel_spect.copy()
    hght,wdth = mel_spect.shape
    
    warp_pt_x = np.random.choice(np.arange(W,wdth-W))
    warp_pt_y = hght//2
    warp_start = [warp_pt_y,warp_pt_x]
    warp_dist = int(np.random.uniform(25,W))*np.random.choice([1,-1])
    warp_destination = [warp_pt_y, warp_pt_x+warp_dist]

    anchor_points = [warp_start, 
                     [0,0],
                     [hght,0],
                     [0,wdth],
                     [hght,wdth],
                     [warp_pt_y,0],
                     [warp_pt_y,wdth]]

    destination_points = [warp_destination,
                         [0,0],
                         [hght,0],
                         [0,wdth],
                         [hght,wdth],
                         [warp_pt_y,0],
                         [warp_pt_y,wdth]]

    src = np.array(anchor_points).reshape(1,7,2).astype(np.float32)
    dst = np.array(destination_points).reshape(1,7,2).astype(np.float32)

    warped_mel, _ = tfa.image.sparse_image_warp(mel_spect,src,dst)
    return warped_mel, [warp_start,warp_destination]

def random_dropout(mel_spect,pct_of_image_to_mask=0.005):
    """
    input: mel_spect: 2d array of mel spectrogram
    pct_of_image_to_mask: percent of image to randomly mask
    
    """
    mel_spect = mel_spect.copy()
    hght,wdth = mel_spect.shape
    num_pix = hght*wdth
    num_pix_to_mask = int(pct_of_image_to_mask*num_pix)

    y_index = np.random.choice(np.arange(5,hght-5),num_pix_to_mask)
    x_index = np.random.choice(np.arange(5,wdth-5),num_pix_to_mask)

    num_covered_pix = 0
    for y,x in zip(y_index,x_index):
        mel_spect[y,x] = 0
        dialate_radius = np.random.choice(np.arange(1,3))
        for r in np.arange(0,dialate_radius):
            rad = r + 1
            movement_vectors = list(itertools.product([n for n in range(1,rad+1)] + [-n for n in range(0,rad+1)], repeat =2))
            for coord in movement_vectors:
                drop_y = y+coord[1]
                drop_x = x+coord[0]
                mel_spect[drop_y,drop_x] = 0
        num_covered_pix+=dialate_radius**2
        if num_covered_pix > num_pix_to_mask:
            break
    return mel_spect    
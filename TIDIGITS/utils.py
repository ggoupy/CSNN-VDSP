import os, sys
import shutil
import numpy as np
import librosa
from sklearn.model_selection import train_test_split



SAMPLE_RATE = 16000
SAMPLE_SIZE = 13824 # longest sample size after resampling and trimming




class DebugPrint:
    def __init__(self, debug=True):
        self.debug = debug

    def __call__(self, msg):
        if self.debug:
            print(msg)




def spike_to_time(spk):
    out = np.ones(spk.shape[1:]) * -1
    for t in range(spk.shape[0]):
        inds = np.argwhere(spk[t]==1)
        for ind in inds:
            x,y,z=ind
            out[x,y,z] = t
    return out




def mfsc(sample, n_fft=512, hop_length=256, n_mels=40, fmin=0, fmax=8000, sr=SAMPLE_RATE):
    """
    Compute log-mel spectrogram.
    """
    return librosa.power_to_db(librosa.feature.melspectrogram(y=sample, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, sr=sr))




def spike_encoding(x, nb_timesteps=15, trash_bins=0):
    """
    Encode a spectrogram into spikes using time-to-first-spike coding.
    
    Args : 
        x (ndarray) : input of shape (frames,frequencies)
        nb_timesteps (int) : number of spike bins
        trash_bins (int) : supplementary final timesteps that are detached (to remove lowest values)
    """
    # spikes shape : (spike_bins,frames,frequencies)
    spikes = np.zeros((nb_timesteps, x.shape[0], x.shape[1]), dtype=np.uint8)
    # Generate nb_timesteps values between max and min, evenly spaced
    # They define levels for spike timing. Shape : (nb_timesteps+trash_bins,)
    # Add <trash_bins> bins containing lowest values (usually close to 0) that will be removed 
    levels = np.linspace(x.max(), x.min(), nb_timesteps+trash_bins)
    # Assign spike timing (index) for each value in spectrogram according to their intensity
    # Large values have a small index and low values a large index. Shape : (frames,frequencies)
    timing = np.digitize(x, levels)
    for t in range(nb_timesteps):
        # Mask to retrieve pixels that fire at timestep t
        spk_mask = np.ma.masked_equal(timing, t)
        # Create spikes
        spikes[t][spk_mask.mask] = 1
    # expand to shape (spike_bins,1,frames,frequencies)
    return np.expand_dims(spikes,1)




def load_encoded_TIDIGITS(nb_timesteps=15, test_size=0.3, seed=42, trim=True, sample_size=SAMPLE_SIZE, dataset_dir="dataset/"):
    """
    Load and preprocess TIDIGITS dataset.

    WARNING :
    Only isolated digits from adult speakers are used for this experiment.
    Hence, content of the directory "adults" from original TIDIGITS dataset
    must be extracted to a new directory of path << ./dataset/ >>
    """
    
    # Use a loading order as file loading is dependent of
    # the machine and can prevent reproducibility.
    # Dataset is shuffled after loading.
    load_order = []
    with open('TIDIGITS_load_order.txt') as f:
        for line in f:
            load_order.append(line)
    files = [f.rstrip() for f in load_order]

    X = []
    y = []
    for f in files:
        # Convert to int label
        str_label = f.split("/")[3][0]
        if str_label == "o": label = -1
        elif str_label == "z": label = 0
        else: label = int(str_label)
        # Load sample
        sample, _ = librosa.load(dataset_dir + f, sr=SAMPLE_RATE)
        # Cut silence
        if trim:
            sample = librosa.effects.trim(sample, top_db=20)[0]
        # Remove the end of the sample if it is too long
        if len(sample) > sample_size: sample = sample[0:sample_size]
        # Padding to ensure samples have the same size
        if len(sample) < sample_size: sample = np.pad(sample, ((0,sample_size-len(sample))), mode='constant')
        # Convert sample into log melspectrogram
        sample = mfsc(sample).T
        # Encode into spike trains
        encoded = spike_encoding(sample, nb_timesteps)
        X.append(encoded)
        y.append(label)
    X = np.array(X, dtype=np.uint8)
    y = np.array(y)

    return train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)




def load_encoded_TIDIGITS2(nb_timesteps=15, test_size=0.3, mfsc_target_bins=60, seed=42, dataset_dir="dataset/"):
    """
    Load and preprocess TIDIGITS dataset.

    WARNING :
    Only isolated digits from adult speakers are used for this experiment.
    Hence, content of the directory "adults" from original TIDIGITS dataset
    must be extracted to a new directory of path << ./dataset/ >>
    """
    
    # Use a loading order as file loading is dependent of
    # the machine and can prevent reproducibility.
    # Dataset is shuffled after loading.
    load_order = []
    with open('TIDIGITS_load_order.txt') as f:
        for line in f:
            load_order.append(line)
    files = [f.rstrip() for f in load_order]

    X = []
    y = []
    for f in files:
        # Convert to int label
        str_label = f.split("/")[3][0]
        if str_label == "o": label = -1
        elif str_label == "z": label = 0
        else: label = int(str_label)
        # Load sample
        sample, _ = librosa.load(dataset_dir + f, sr=SAMPLE_RATE)
        # Convert sample into log melspectrogram
        sample = mfsc(sample, hop_length=int(len(sample)/(mfsc_target_bins-1))).T
        # Encode into spike trains
        encoded = spike_encoding(sample, nb_timesteps)
        X.append(encoded)
        y.append(label)
    X = np.array(X, dtype=np.uint8)
    y = np.array(y)

    return train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)




def load_TIDIGITS(test_size=0.3, seed=42, trim=True, sample_size=SAMPLE_SIZE, dataset_dir="dataset/"):
    """
    Load and preprocess TIDIGITS dataset, without spike encoding.

    WARNING :
    Only isolated digits from adult speakers are used for this experiment.
    Hence, content of the directory "adults" from original TIDIGITS dataset
    must be extracted to a new directory of path << ./dataset/ >>
    """
    
    # Use a loading order as file loading is dependent of
    # the machine and can prevent reproducibility.
    # Dataset is shuffled after loading.
    load_order = []
    with open('TIDIGITS_load_order.txt') as f:
        for line in f:
            load_order.append(line)
    files = [f.rstrip() for f in load_order]

    X = []
    y = []
    for f in files:
        # Convert to int label
        str_label = f.split("/")[3][0]
        if str_label == "o": label = -1
        elif str_label == "z": label = 0
        else: label = int(str_label)
        # Load sample
        sample, _ = librosa.load(dataset_dir + f, sr=SAMPLE_RATE)
        # Cut silence
        if trim:
            sample = librosa.effects.trim(sample, top_db=20)[0]
        # Remove the end of the sample if it is too long
        if len(sample) > sample_size: sample = sample[0:sample_size]
        # Padding to ensure samples have the same size
        if len(sample) < sample_size: sample = np.pad(sample, ((0,sample_size-len(sample))), mode='constant')
        # Convert sample into log melspectrogram
        sample = mfsc(sample).T
        X.append(sample)
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)



def load_TIDIGITS2(test_size=0.3, seed=42, mfsc_target_bins=60, dataset_dir="dataset/"):
    """
    Load and preprocess TIDIGITS dataset, without spike encoding.

    WARNING :
    Only isolated digits from adult speakers are used for this experiment.
    Hence, content of the directory "adults" from original TIDIGITS dataset
    must be extracted to a new directory of path << ./dataset/ >>
    """
    
    # Use a loading order as file loading is dependent of
    # the machine and can prevent reproducibility.
    # Dataset is shuffled after loading.
    load_order = []
    with open('TIDIGITS_load_order.txt') as f:
        for line in f:
            load_order.append(line)
    files = [f.rstrip() for f in load_order]

    X = []
    y = []
    for f in files:
        # Convert to int label
        str_label = f.split("/")[3][0]
        if str_label == "o": label = -1
        elif str_label == "z": label = 0
        else: label = int(str_label)
        # Load sample
        sample, _ = librosa.load(dataset_dir + f, sr=SAMPLE_RATE)
        # Convert sample into log melspectrogram
        sample = mfsc(sample, hop_length=int(len(sample)/(mfsc_target_bins-1))).T
        X.append(sample)
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
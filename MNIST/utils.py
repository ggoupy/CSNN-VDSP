import numpy as np
from math import ceil
from mnist import MNIST




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




# adapted from https://github.com/npvoid/SDNN_python
def spike_encoding(img, nb_timesteps):
    """
    Encode an image into spikes using a temporal coding based on pixel intensity.
    
    Args : 
        img (ndarray) : input of shape (height,width)
        nb_timesteps (int) : number of spike bins
    """
    # Intensity to latency
    with np.errstate(divide='ignore',invalid='ignore'): # suppress division by zero warning  
        I, lat = np.argsort(1/img.flatten()), np.sort(1/img.flatten())
    # Remove pixels of value 0
    I = np.delete(I, np.where(lat == np.inf))
    # Convert 1D into 2D coordinates
    II = np.unravel_index(I, img.shape)
    # Compute the number of steps
    t_step = np.ceil(np.arange(I.size) / (I.size / (nb_timesteps-1))).astype(np.uint8)
    # Add dimension axis to index array
    # shape : (timestep, height, width)
    II = (t_step,) + II
    # Create spikes
    spike_times = np.zeros((nb_timesteps, img.shape[0], img.shape[1]), dtype=np.uint8)
    spike_times[II] = 1
    # Add channel dimension
    return np.expand_dims(spike_times, 1)




def preprocess_MNIST(dataset, nb_timesteps):
    """
    Preprocess the MNIST dataset. 
    """
    samples, height, width = dataset.shape
    output = np.zeros((samples, nb_timesteps, 1, height, width), dtype=np.uint8)
    for i,img in enumerate(dataset):
        # Encode into spike trains
        output[i] = spike_encoding(img, nb_timesteps)
    return output




def load_encoded_MNIST(nb_timesteps=15):
    """
    Load and preprocess the MNIST dataset. 
    """
    mndata = MNIST()
    images, labels = mndata.load_training()
    
    # Training set
    X_train, y_train = np.asarray(images), np.asarray(labels)
    X_train = X_train.reshape(-1, 28, 28)
    # Random shuffling
    random_indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[random_indices], y_train[random_indices]

    # Testing set
    images, labels = mndata.load_testing()
    X_test, y_test = np.asarray(images), np.asarray(labels)
    X_test = X_test.reshape(-1, 28, 28)

    X_train_encoded = preprocess_MNIST(X_train, nb_timesteps)
    X_test_encoded = preprocess_MNIST(X_test, nb_timesteps)
    
    return X_train_encoded, X_test_encoded, y_train, y_test




def load_MNIST():
    """
    Load and preprocess the MNIST dataset, without spike encoding.
    """
    mndata = MNIST()
    images, labels = mndata.load_training()
    
    # Training set
    X_train, y_train = np.asarray(images), np.asarray(labels)
    X_train = X_train.reshape(-1, 28, 28)
    # Random shuffling
    random_indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[random_indices], y_train[random_indices]

    # Testing set
    images, labels = mndata.load_testing()
    X_test, y_test = np.asarray(images), np.asarray(labels)
    X_test = X_test.reshape(-1, 28, 28)
    
    return X_train, X_test, y_train, y_test
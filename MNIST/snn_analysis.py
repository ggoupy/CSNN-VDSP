import time, random, os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from torch.nn.functional import conv2d, max_pool2d

from snn import DEFAULT_NETWORK_PARAMS, SpikingConv, SpikingPool, SpikingInput
from utils import DebugPrint, spike_to_time, load_encoded_MNIST

"""
This file is used to save CSNN outputs for analysis purpose.  
"""




class SNN:
    """ 
    Spiking convolutional neural network model with IF neurons that can fire at most once.
    """
    def __init__(self, input_shape, params=DEFAULT_NETWORK_PARAMS):
        
        """
        All layers must have the same voltage convention for VDSP
        Initially, neurons have a potential of v_rest (= neuron has not fired)
        When firing, the potential of a neuron is set to v_reset (= neuron has fired)
        """
        self.v_rest = params["v_rest"]
        self.v_reset = params["v_reset"]
        
        self.input_layer = SpikingInput(input_shape, v_reset=self.v_reset, v_rest=self.v_rest)
        print(self.input_layer.output_shape)

        conv1 = SpikingConv(input_shape,
            out_channels=params["conv_out_channels"], kernel_size=params["conv_kernel_size"],
            stride=params["conv_stride"], padding=params["conv_padding"], nb_winners=params["conv_nb_winners"],
            firing_threshold=params["conv_firing_thr"], inhibition_radius=params["conv_inhib_radius"],
            vdsp_lr=params["conv_vdsp_lr"], adaptive_lr=params["conv_adaptive_lr"],
            vdsp_max_lr=params["conv_vdsp_max_lr"], update_lr_cnt=params["conv_update_lr_cnt"],
            vdsp_dep_factor=params["conv_vdsp_dep_factor"], v_reset=self.v_reset, v_rest=self.v_rest
        )
        print(conv1.output_shape)

        pool1 = SpikingPool(conv1.output_shape, 
            kernel_size=params["pool_kernel_size"], stride=params["pool_stride"],
            padding=params["pool_padding"], v_reset=self.v_reset, v_rest=self.v_rest
        )
        print(pool1.output_shape)

        self.conv_layers = [conv1]
        self.pool_layers = [pool1]

        self.trainable_layers = self.conv_layers
        self.output_shape = self.pool_layers[-1].output_shape
        self.output_size = np.prod(self.output_shape)
        self.recorded_sum_spks = []


    def save_weights(self, layer, filename):
        np.save(filename, self.conv_layers[layer].weights)


    def reset(self):
        self.input_layer.reset()
        for layer in self.conv_layers:
            layer.reset()
        for layer in self.pool_layers:
            layer.reset()


    def __call__(self, x, train_layer=None):
        self.reset()
        self.input_layer.init(x)
        nb_timesteps = x.shape[0]
        output_conv = np.zeros((nb_timesteps,) + self.conv_layers[0].output_shape, dtype=np.uint8)
        output_spikes = np.zeros((nb_timesteps,) + self.output_shape, dtype=np.uint8)
        tot_spks = 0
        for t in range(nb_timesteps):
            spk, pot = self.input_layer(x[t])
            tot_spks += spk.sum()
            spk, pot = self.conv_layers[0](spk, pot, train=(train_layer==0))
            tot_spks += spk.sum()
            output_conv[t] = spk.astype(np.uint8)
            spk, pot = self.pool_layers[0](spk)
            tot_spks += spk.sum()
            output_spikes[t] = spk.astype(np.uint8)
        if train_layer == None: self.recorded_sum_spks.append(tot_spks)
        return output_conv, output_spikes




def main(
    seed=1,                            # To control the randomness
    nb_timesteps=15,                   # Number of spike bins
    epochs=1,                          # Number of epochs per layer
    convergence_rate=0.01,             # Stop training when learning convergence reaches this rate
    max_train_samples=-1,              # Maximum number of samples used for training per layer
    snn_params=DEFAULT_NETWORK_PARAMS, # Parameters of the SNN
    extended_print=False               # Print all messages
):
    # Custom printing
    printd = DebugPrint(extended_print)

    # Seed everything
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Load dataset encoded as spikes with a temporal coding
    X_train, X_test, y_train, y_test = load_encoded_MNIST(nb_timesteps=nb_timesteps)

    # Init SNN
    input_shape = X_train[0][0].shape
    snn = SNN(input_shape, snn_params)

    printd(f"Input shape : {X_train[0].shape} ({np.prod(X_train[0].shape)} values)")
    printd(f"Output shape : {snn.output_shape} ({snn.output_size} values)")
    printd(f"Mean spikes count per input : {X_train.mean(0).sum()}")

    CHECKPOINTS = [0, 100, 200, 300, 400, 500, 600]
    curr_checkpoint = 0
    train_cnt = 0

    for x,y in zip(tqdm(X_train), y_train):
        
        snn(x, train_layer=0)
        
        train_cnt += 1

        if curr_checkpoint < len(CHECKPOINTS) and train_cnt >= CHECKPOINTS[curr_checkpoint]:
    
            NB_TO_SAVE = 5000
            cnt = 0
            conv_train = np.zeros((NB_TO_SAVE,) + snn.conv_layers[0].output_shape, dtype=np.uint8)
            pool_train = np.zeros((NB_TO_SAVE,) + snn.output_shape, dtype=np.uint8)
            for i, (x, y) in enumerate(zip(tqdm(X_train), y_train)):
                spk_conv, spk_out = snn(x)
                if spk_out.sum() == 0: printd("[WARNING] No output spike recorded.")
                conv_train[i] = spk_conv.sum(0)
                pool_train[i] = spk_out.sum(0)
                cnt += 1
                if cnt >= NB_TO_SAVE:
                    break

            np.save(f"logs/output_conv_train_{CHECKPOINTS[curr_checkpoint]}_training_samples", conv_train)
            np.save(f"logs/output_pool_train_{CHECKPOINTS[curr_checkpoint]}_training_samples", pool_train)

            snn.save_weights(0, f"logs/weights_{train_cnt}_training_samples")
            curr_checkpoint += 1

        if snn.trainable_layers[0].get_learning_convergence() < convergence_rate:
            break

    snn.save_weights(0, f"logs/weights_{train_cnt}_training_samples")


    NB_TO_SAVE = 5000
    cnt = 0
    conv_train = np.zeros((NB_TO_SAVE,) + snn.conv_layers[0].output_shape, dtype=np.uint8)
    pool_train = np.zeros((NB_TO_SAVE,) + snn.output_shape, dtype=np.uint8)
    for i, (x, y) in enumerate(zip(tqdm(X_train), y_train)):
        spk_conv, spk_out = snn(x)
        if spk_out.sum() == 0: printd("[WARNING] No output spike recorded.")
        conv_train[i] = spk_conv.sum(0)
        pool_train[i] = spk_out.sum(0)
        cnt += 1
        if cnt >= NB_TO_SAVE:
            break

    np.save(f"logs/output_conv_train_{train_cnt}_training_samples", conv_train)
    np.save(f"logs/output_pool_train_{train_cnt}_training_samples", pool_train)
    np.save("logs/output_y_train", y_train[0:NB_TO_SAVE])
 



##################################################################################################
##################################################################################################


if __name__ == "__main__":
    main()
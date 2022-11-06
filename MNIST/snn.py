import time, random, os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from torch.nn.functional import conv2d, max_pool2d

from utils import DebugPrint, spike_to_time, load_encoded_MNIST


"""
Convolutional spiking neural network with voltage-dependent synaptic plasticity (VDSP),
for handwritten digits recognition task.

Adapted from the following references:

       [1] Kheradpisheh, S.R., et al. (2018). STDP-based spiking deep neural networks for object recognition.
           Neural Networks. (https://doi.org/10.1016/j.neunet.2017.12.005).

       [2] Mozafari, M., Ganjtabesh, M., Nowzari-Dalini, A., & Masquelier, T. (2019). SpykeTorch: Efficient
           Simulation of Convolutional Spiking Neural Networks With at Most One Spike per Neuron. Frontiers
           in Neuroscience (https://doi.org/10.3389/fnins.2019.00625).

       [3] Nikhil Garg, Ismael Balafrej, et al. (2022). Voltage-Dependent Synaptic Plasticity (VDSP): 
           Unsupervised probabilistic Hebbian plasticity rule based on neurons membrane potential.
           Frontiers in Neuroscience (https://doi.org/10.3389/fnins.2022.983950).
           
       [4] https://github.com/npvoid/SDNN_python
"""




class SpikingPool:
    """ 
    Pooling layer with spiking neurons that can fire at most once.
    """
    def __init__(self, input_shape, kernel_size, stride, padding=0, v_rest=0, v_reset=-1):
        in_channels, in_height, in_width = input_shape
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride,stride)
        self.padding = padding if isinstance(padding, tuple) else (padding,padding)
        out_height = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        out_width = int(((in_width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        self.output_shape = (in_channels, out_height, out_width)
        self.v_reset = v_reset
        self.v_rest = v_rest

        # Output neurons
        self.pot = np.ones(self.output_shape) * v_rest
        self.active_neurons = np.ones(self.output_shape).astype(bool)


    def reset(self):
        self.pot[:] = self.v_rest
        self.active_neurons[:] = True


    def __call__(self, in_spks):
        # padding
        in_spks = np.pad(in_spks, ((0,), (self.padding[0],), (self.padding[1],)), mode='constant')
        in_spks = torch.Tensor(in_spks).unsqueeze(0)
        # Max pooling (using torch as it is fast and easier)
        out_spks = max_pool2d(in_spks, self.kernel_size, stride=self.stride).numpy()[0]
        # Keep spikes of active neurons
        out_spks = out_spks * self.active_neurons
        # Update active neurons as each pooling neuron can fire only once
        self.active_neurons[out_spks == 1] = False
        # Update potentials of neurons that fired
        self.pot[out_spks == 1] = self.v_reset
        return out_spks, np.copy(self.pot)

 


class SpikingConv:
    """
    Convolutional layer with IF spiking neurons that can fire at most once.
    Implements a Winner-take-all VDSP learning rule.
    """
    def __init__(self, input_shape,
                out_channels, kernel_size, stride=1, padding=0, 
                nb_winners=1, firing_threshold=1, inhibition_radius=0, 
                adaptive_lr=False, vdsp_max_lr=0.15, update_lr_cnt=500,
                vdsp_lr=0.01, vdsp_pot_factor=1, vdsp_dep_factor=2,
                weight_init_mean=0.8, weight_init_std=0.05, w_max=1, w_min=0,
                v_reset=-1, v_rest=0
        ):
        in_channels, in_height, in_width = input_shape
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride,stride)
        self.padding = padding if isinstance(padding, tuple) else (padding,padding)
        self.firing_threshold = firing_threshold
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.weights = np.random.normal(
            loc=weight_init_mean, scale=weight_init_std,
            size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))

        # Output neurons
        out_height = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        out_width = int(((in_width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        self.pot = np.ones((out_channels, out_height, out_width)) * v_rest
        self.active_neurons = np.ones(self.pot.shape).astype(bool)
        self.output_shape = self.pot.shape

        # VDSP
        self.vdsp_cnt = 0
        self.vdsp_lr = vdsp_lr
        self.plasticity = True
        self.adaptive_lr = adaptive_lr
        self.vdsp_max_lr = vdsp_max_lr
        self.update_lr_cnt = update_lr_cnt
        self.vdsp_neurons = np.ones(self.pot.shape).astype(bool)
        self.inhibition_radius = inhibition_radius
        self.nb_winners = nb_winners
        self.vdsp_pot_factor = vdsp_pot_factor
        self.vdsp_dep_factor = vdsp_dep_factor
        self.w_max = w_max
        self.w_min = w_min
        

    def get_learning_convergence(self):
        return (self.weights * (self.w_max-self.weights)).sum() / np.prod(self.weights.shape)


    def reset(self):
        self.pot[:] = self.v_rest
        self.active_neurons[:] = True
        self.vdsp_neurons[:] = True


    def get_winners(self):
        winners = []
        channels = np.arange(self.pot.shape[0])
        # Copy potentials and keep neurons that can do VDSP
        pots_tmp = np.copy(self.pot) * self.vdsp_neurons
        # Find at most nb_winners
        while len(winners) < self.nb_winners:
            # Find new winner
            winner = np.argmax(pots_tmp) # 1D index
            winner = np.unravel_index(winner, pots_tmp.shape) # 3D index
            # Assert winner potential is higher than firing threshold
            # If not, stop the winner selection 
            if pots_tmp[winner] <= self.firing_threshold:
                break
            # Add winner
            winners.append(winner)
            # Disable winner selection for neurons in neighborhood of other channels
            pots_tmp[channels != winner[0],
                max(0,winner[1]-self.inhibition_radius):winner[1]+self.inhibition_radius+1,
                max(0,winner[2]-self.inhibition_radius):winner[2]+self.inhibition_radius+1
            ] = self.v_rest
            # Disable winner selection for neurons in same channel
            pots_tmp[winner[0]] = self.v_rest 
        return winners


    def lateral_inhibition(self, spks):
        # Get index of spikes
        spks_c,spks_h,spks_w = np.where(spks)
        # Get associated potentials
        spks_pot = np.array([self.pot[spks_c[i],spks_h[i],spks_w[i]] for i in range(len(spks_c))])
        # Sort index by potential in a descending order
        spks_sorted_ind = np.argsort(spks_pot)[::-1]
        # Sequentially inhibit neurons in the neighborhood of other channels
        # Neurons with highest potential inhibit neurons with lowest one, even if both spike
        for ind in spks_sorted_ind:
            # Check that neuron has not been inhibated by another one
            if spks[spks_c[ind],spks_h[ind],spks_w[ind]] == 1:
                # Compute index
                inhib_channels = np.arange(spks.shape[0]) != spks_c[ind]
                # Inhibit neurons
                spks[inhib_channels,spks_h[ind],spks_w[ind]] = 0 
                self.pot[inhib_channels,spks_h[ind],spks_w[ind]] = self.v_rest
                self.active_neurons[inhib_channels,spks_h[ind],spks_w[ind]] = False
        return spks


    def get_conv_of(self, input, output_neuron):
        # Neuron index
        n_c, n_h, n_w = output_neuron
        # Get the list of convolutions on input neurons to update output neurons
        # shape : (in_neuron_values, nb_convs)
        input = torch.Tensor(input).unsqueeze(0) # batch axis
        convs = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, stride=self.stride)[0].numpy()
        # Get the convolution for the spiking neuron
        conv_ind = (n_h * self.pot.shape[2]) + n_w # 2D to 1D index
        return convs[:, conv_ind]

        
    def vdsp(self, in_pot, winner):
        if not self.vdsp_neurons[winner]: exit(1)
        if not self.plasticity: return
        # Count call
        self.vdsp_cnt += 1
        # Winner 3D coordinates
        winner_c, winner_h, winner_w = winner
        # Get convolution window used to compute output neuron potential
        conv = self.get_conv_of(in_pot, winner).flatten()
        # Compute dW
        w = self.weights[winner_c].flatten() * (self.w_max - self.weights[winner_c]).flatten()
        cond_pot = conv < self.v_rest # fired recently
        cond_dep = conv >= self.v_rest # has not fired recently
        # Normalize potential between 0 and 1
        g_pot = self.vdsp_pot_factor
        g_dep = self.vdsp_dep_factor - conv/self.firing_threshold
        dW = (cond_pot * w * g_pot * self.vdsp_lr) - (cond_dep * w * g_dep * self.vdsp_lr)
        self.weights[winner_c] += dW.reshape(self.weights[winner_c].shape)
        # Make sure weights are in the range [w_min , w_max]
        self.weights[winner_c] = np.clip(self.weights[winner_c], self.w_min, self.w_max)
        # Lateral inhibition between channels (local inter competition)
        channels = np.arange(self.pot.shape[0])
        self.vdsp_neurons[channels != winner_c,
            max(0,winner_h-self.inhibition_radius):winner_h+self.inhibition_radius+1,
            max(0,winner_w-self.inhibition_radius):winner_w+self.inhibition_radius+1
        ] = False
        # Lateral inhibition in the same channel (gobal intra competition)
        self.vdsp_neurons[winner_c] = False
        # Adaptive learning rate
        if self.adaptive_lr and self.vdsp_cnt % self.update_lr_cnt == 0:
            self.vdsp_lr = min(2 * self.vdsp_lr, self.vdsp_max_lr)


    def __call__(self, spk_in, pot_in, train=False):
        # padding 
        pot_in = np.pad(pot_in, ((0,), (self.padding[0],), (self.padding[1],)), mode='constant', constant_values=self.v_rest)
        spk_in = np.pad(spk_in, ((0,), (self.padding[0],), (self.padding[1],)), mode='constant', constant_values=0)
        # Output recorded spikes
        spk_out = np.zeros(self.pot.shape)
        # Convert to torch tensors
        x = torch.Tensor(spk_in).unsqueeze(0) # Add batch axis for torch conv2d
        weights = torch.Tensor(self.weights) # converts at the fly... (not good)
        # Convolve (using torch as it is fast and easier)
        out_conv = conv2d(x, weights, stride=self.stride).numpy()[0] # Converted to numpy
        # Update potentials
        self.pot[self.active_neurons] += out_conv[self.active_neurons]
        # Check for neurons that can spike
        output_spikes = self.pot > self.firing_threshold
        if np.any(output_spikes):
            # Generate spikes
            spk_out[output_spikes] = 1
            # Lateral inhibition for neurons in neighborhood in other channels
            # Inhibit and disable neurons with lower potential that fire
            spk_out = self.lateral_inhibition(spk_out)
            # VDSP plasticity
            if train and self.plasticity:
                # Find winners (based on potential)
                winners = self.get_winners()
                # Apply VDSP for each neuron winner
                for winner in winners:
                    self.vdsp(pot_in, winner)
            # Reset potentials and disable neurons that fire
            # v_reset indicates that neuron has fired before (for VDSP)
            self.pot[spk_out == 1] = self.v_reset
            self.active_neurons[spk_out == 1] = False
        return spk_out, np.copy(self.pot)




class SpikingInput:
    """ 
    Input layer with spiking neurons that can fire at most once.
    This class is used to simulate potential of input encoding.
    """
    def __init__(self, input_shape, v_rest=0, v_reset=-1):
        self.output_shape = input_shape
        self.pot = np.ones(self.output_shape, dtype=float) * v_rest
        self.incr_pot = None
        # Keep track of active neurons as they can fire at most once
        self.active_neurons = np.ones(self.output_shape).astype(bool)
        self.v_reset = v_reset
        self.v_rest = v_rest


    def reset(self):
        self.pot[:] = self.v_rest
        self.active_neurons[:] = True


    def init(self, in_spks):
        # Compute values to increment each neuron potential at each timestep
        # to reach firing threshold (1) at timestep where spike is encoded
        timing = spike_to_time(in_spks)
        timing[timing==-1] = np.inf # No increment for neurons that do not spike
        self.incr_pot = 1/(timing+1)


    def __call__(self, in_spks):
        self.pot[self.active_neurons] += self.incr_pot[self.active_neurons]
        # Keep spikes for active neurons
        # NOTE : should not remove any spike if a temporal encoding is used
        in_spks = in_spks * self.active_neurons
        self.pot[in_spks == 1] = self.v_reset
        self.active_neurons[in_spks == 1] = False
        return in_spks.astype(np.float64), np.copy(self.pot)




DEFAULT_NETWORK_PARAMS = {
    "conv_out_channels": 70,
    "conv_kernel_size": 7,
    "conv_stride": 1,
    "conv_padding": 3,
    "conv_firing_thr": 10,
    "conv_inhib_radius": 3,
    "conv_nb_winners": 7,
    "conv_vdsp_dep_factor": 2,
    "conv_vdsp_lr": 0.01,
    "conv_adaptive_lr": True,
    "conv_vdsp_max_lr": 0.1,
    "conv_update_lr_cnt": 500,
    "pool_kernel_size": 3,
    "pool_stride": 3,
    "pool_padding": 0,
    "v_rest": 0,
    "v_reset": -1
}

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
  
        conv1 = SpikingConv(input_shape,
            out_channels=params["conv_out_channels"], kernel_size=params["conv_kernel_size"],
            stride=params["conv_stride"], padding=params["conv_padding"], nb_winners=params["conv_nb_winners"],
            firing_threshold=params["conv_firing_thr"], inhibition_radius=params["conv_inhib_radius"],
            vdsp_lr=params["conv_vdsp_lr"], adaptive_lr=params["conv_adaptive_lr"],
            vdsp_max_lr=params["conv_vdsp_max_lr"], update_lr_cnt=params["conv_update_lr_cnt"],
            vdsp_dep_factor=params["conv_vdsp_dep_factor"], v_reset=self.v_reset, v_rest=self.v_rest
        )

        pool1 = SpikingPool(conv1.output_shape, 
            kernel_size=params["pool_kernel_size"], stride=params["pool_stride"],
            padding=params["pool_padding"], v_reset=self.v_reset, v_rest=self.v_rest
        )

        self.conv_layers = [conv1]
        self.pool_layers = [pool1]

        self.trainable_layers = self.conv_layers
        self.output_shape = self.pool_layers[-1].output_shape
        self.output_size = np.prod(self.output_shape)
        self.recorded_sum_spks = []


    def save_weights(self, layer):
        np.save(f"logs/weights_layer_{layer}_{str(time.time())}", self.conv_layers[layer].weights)


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
        output_spikes = np.zeros((nb_timesteps,) + self.output_shape, dtype=np.uint8)
        tot_spks = 0
        for t in range(nb_timesteps):
            spk, pot = self.input_layer(x[t])
            tot_spks += spk.sum()
            spk, pot = self.conv_layers[0](spk, pot, train=(train_layer==0))
            tot_spks += spk.sum()
            spk, pot = self.pool_layers[0](spk)
            tot_spks += spk.sum()
            output_spikes[t] = spk.astype(np.uint8)
        if train_layer == None: self.recorded_sum_spks.append(tot_spks)
        return output_spikes




def gridsearch_readout(out_train, out_test, y_train, y_test, seed):
    params_grid = {
        'max_iter': [10000],
        'random_state': [seed],
        'loss': ['hinge', 'squared_hinge'],
        'dual': [True, False],
        'intercept_scaling': [0.5, 1, 2],
        'multi_class': ['ovr', 'crammer_singer'],
        'C':[0.001,0.005,0.01,0.05,0.1,0.5,1,2],
    }
    grid = GridSearchCV(LinearSVC(), params_grid, refit=True, scoring="accuracy")
    grid.fit(out_train, y_train)
    y_pred = grid.predict(out_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Best accuracy : {acc}")
    print(f"Best params : {grid.best_params_}")




def main(
    seed=0,                            # To control the randomness
    nb_timesteps=15,                   # Number of spike bins
    epochs=1,                          # Number of epochs per layer
    convergence_rate=0.01,             # Stop training when learning convergence reaches this rate
    max_train_samples=-1,              # Maximum number of samples used for training per layer
    save_weights=False,                # True to save weights after training
    snn_params=DEFAULT_NETWORK_PARAMS, # Parameters of the SNN
    return_nb_train_samples=False,     # Return the number of training samples used
    extended_print=False,              # Print all messages
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
    
    printd("\n### TRAINING ###")
    nb_training_samples = 0
    for layer in range(len(snn.trainable_layers)):
        printd(f"Layer {layer+1}...")
        layer_done = False
        for epoch in range(epochs):
            printd(f"\t epoch {epoch+1}")
            train_cnt = 0
            for x,y in zip(tqdm(X_train), y_train):
                nb_training_samples += 1
                if max_train_samples != -1 and train_cnt >= max_train_samples:
                    layer_done = True
                    break
                snn(x, train_layer=layer)
                if train_cnt % 250 == 0: printd(snn.trainable_layers[layer].get_learning_convergence())
                train_cnt += 1
                if max_train_samples == -1 and snn.trainable_layers[layer].get_learning_convergence() < convergence_rate:
                    layer_done = True
                    break
            if layer_done:
                break
        if save_weights:
            snn.save_weights(layer)
        w = snn.trainable_layers[layer].weights
        printd(f"Learning convergence : {snn.trainable_layers[layer].get_learning_convergence()}")
        printd(f"Number of pot weights : {np.count_nonzero(w[w>0.90])}")
        printd(f"Number of dep weights : {np.count_nonzero(w[w<0.10])}")

    # For analysis purpose
    if return_nb_train_samples: return nb_training_samples


    printd("\n### TESTING ###")
    output_train_sum = np.zeros((len(X_train), snn.output_size), dtype=np.uint8)
    #output_train_time = np.zeros((len(X_train), snn.output_size), dtype=np.int64)
    for i, (x, y) in enumerate(zip(tqdm(X_train), y_train)):
        spk = snn(x)
        if spk.sum() == 0: printd("[WARNING] No output spike recorded.")
        output_train_sum[i] = spk.sum(0).flatten()
        #output_train_time[i] = spike_to_time(spk).flatten()

    output_test_sum = np.zeros((len(X_test), snn.output_size), dtype=np.uint8)
    #output_test_time = np.zeros((len(X_test), snn.output_size), dtype=np.int64)
    for i, (x, y) in enumerate(zip(tqdm(X_test), y_test)):
        spk = snn(x)
        if spk.sum() == 0: printd("[WARNING] No output spike recorded.")
        output_test_sum[i] = spk.sum(0).flatten()
        #output_test_time[i] = spike_to_time(spk).flatten()

    
    print(f"Mean total number of spikes per sample : {np.mean(snn.recorded_sum_spks)}")
    
    ### READOUT ###

    clf = LinearSVC(random_state=seed, max_iter=10000, C=0.005)
    clf.fit(output_train_sum, y_train)
    y_pred = clf.predict(output_test_sum)
    acc = accuracy_score(y_test, y_pred) 
    print(f"Accuracy : {acc}")
    return acc




##################################################################################################
##################################################################################################


def mean_acc(N=10, init_seed=0, with_training=True):
    recorded_acc = np.zeros(N)
    for i,seed in enumerate(range(init_seed,init_seed+N)):
        max_train_samples = -1 if with_training else 0
        recorded_acc[i] = main(seed=seed, max_train_samples=max_train_samples, extended_print=False)
    print("\n\n")
    print(f"Mean : {recorded_acc.mean()} +- {recorded_acc.std()}")
    print(recorded_acc)



if __name__ == "__main__":
    mean_acc()

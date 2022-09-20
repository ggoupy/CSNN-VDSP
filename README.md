# Convolutional SNN with VDSP

This is the code for the paper *Unsupervised and efficient learning in sparsely activated convolutional spiking neural networks enabled by voltage-dependent synaptic plasticity*.

## Getting started

### Dependencies
- tqdm
- python-mnist
- numpy
- librosa
- scikit_learn
- torch 

### Installation
Dependencies for each subdirectory can be installed with : 
```
pip3 install -r requirements.txt
```

## Usage

To train and evaluate the CSNN on the `MNIST` or the `TIDIGITS` task, run the following command in the desired subdirectory : 
```
python3 snn.py
```

Several other functions are available in the file `snn.py` (and also `snn_analysis.py` for `MNIST`) to reproduce experiments presented in the paper. 

## Acknowledgements

- Institut Interdisciplinaire d'Innovation Technologique (3IT), Université de Sherbrooke.
- Laboratoire Nanotechnologies Nanosystèmes (LN2) – CNRS UMI-3463, Université de Sherbrooke, Sherbrooke, Canada
- Institute of Electronics, Microelectronics and Nanotechnology (IEMN), Université de Lille, Villeneuve d’Ascq, France.
- NECOTIS Research Lab, Electrical and Computer Engineering Dep., Université de Sherbrooke, Sherbrooke, Canada.


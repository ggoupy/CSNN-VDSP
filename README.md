# CSNN & VDSP

This is the code for the paper *Unsupervised and efficient learning in sparsely activated convolutional spiking neural networks enabled by voltage-dependent synaptic plasticity*, available [here](https://iopscience.iop.org/article/10.1088/2634-4386/acad98).

## Getting started

The code was written in Python 3.9.

### Dependencies
- tqdm (4.64.0)
- numpy (1.20.1)
- librosa (0.9.2)
- scikit_learn (1.1.2)
- torch (1.12.1)
- python-mnist (0.7)

### Installation
Dependencies for each subdirectory can be installed with : 
```
pip3 install -r requirements.txt
```
  
Also, the library *libsndfile* must be installed. (`dnf install libsndfile` on fedora, `apt install libsndfile1` on ubuntu).

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


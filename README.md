DeepST
======
[DeepST](https://github.com/lucktroy/DeepST), A **Deep Learning** Toolbox for Spatio-Temporal Data


## Installation

DeepST uses the following dependencies: 

* [Keras](https://keras.io/#installation) and its dependencies are required to use DeepST. 
* [Theano](http://deeplearning.net/software/theano/install.html#install) or [TensorFlow](https://github.com/tensorflow/tensorflow#download-and-setup), but **Theano** is recommended. 
* numpy and scipy
* HDF5 and [h5py](http://www.h5py.org/)
* [pandas](http://pandas.pydata.org/)


To install DeepST, `cd` to the **DeepST** folder and run the install command:

```
python setup.py install
```

To install the development version:

```
python setup.py develop
```

## Data path

The default `DATAPATH` variable is `DATAPATH=[path_to_DeepST]/data`. You may set your `DATAPATH` variable using

```
# Windows
set DATAPATH=[path_to_your_data]

# Linux
export DATAPATH=[path_to_your_data]
```
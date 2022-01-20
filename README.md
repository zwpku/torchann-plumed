Torch Artificial Neural Network (ANN) Function for plumed
====================

This repository contains files for the plumed Torch ANN function (torchannfunc) module.  It implements `TorchANNFunc` class, which is a subclass of `Function` class.   
## Installation

Enable compilation by adding the `--enable-modules=torchannfunc` to the configure command.

## Usage

It is used in a similar way to [other plumed functions](https://www.plumed.org/doc-v2.5/user-doc/html/_function.html).  To define a `TorchANNFunc` function object, we need to define following keywords:

- `ARG` (string array): input variable names for the fully-connected feedforward neural network

- `NUM_LAYERS` (int): number of layers for the neural network

- `NUM_NODES` (int array): number of nodes in all layers of the neural network

- `ACTIVATIONS` (string array): types of activation functions of layers, currently we have implemented "Linear", "Tanh", "Circular" layers, it should be straightforward to add other types as well

- `WEIGHTS` (numbered keyword, double array): this is a numbered keyword, `WEIGHTS0` represents flattened weight array connecting layer 0 and layer 1, `WEIGHTS1` represents flattened weight array connecting layer 1 and layer 2, ...  An example is given in the next section.

- `BIASES` (numbered keyword, double array): this is a numbered keyword, BIASES0 represents bias array for layer 1, BIASES1 represents bias array for layer 2, ...


## Examples

## Authors

Wei Zhang 

## Copyright

See ./COPYRIGHT

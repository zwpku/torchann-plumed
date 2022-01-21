Torch Artificial Neural Network (TorchANN) Function for plumed
====================

This repository contains files for the plumed Torch ANN function (torchann) module.  It implements `TorchANN` class, which is a subclass of `Function` class.   
## Installation

Enable compilation by adding the `--enable-modules=torchann` to the configure command.

## Usage

It is used in a similar way to [other plumed functions](https://www.plumed.org/doc-v2.5/user-doc/html/_function.html).  To define a `TorchANN` function object, we need to define following keywords:

- `ARG` (string array): input variable names for the neural network


## Examples

## Authors

Wei Zhang 

## Copyright

See ./COPYRIGHT

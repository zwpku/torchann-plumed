Torch Artificial Neural Network (TorchANN) Function for plumed
====================

This is a plugin for plumed which implements the `TorchANN` class as a subclass of `Function` class. It allows one to define functions in plumed that are represented by artificial neural networks.

## Prerequiste
- [plumed](https://www.plumed.org/download)
- [LibTorch](http://www.pytorch.org/get-started/locally)

## Installation
This plugin uses [CMake](http://cmake.org) as its build system. Once you are in the directory containing files of this package, follow these steps to install it:
1. Create the directory in which to build the plugin: mkdir ./build
2. In ./build directory run ccmake ../, and set the following variables.
  * set `Torch_DIR` to point to the directory where LibTorch is installed (it is typically `/path-to-libtorch/share/cmake/Torch`); 
  * set `PlUMED_INC_DIR` to point to the directory which contains the header files of plumed;
  * set `PLUMED` to point to the directory which contains the library files of plumed, i.e. libplumedKernel.so;
  * set `CMAKE_INSTALL_PREFIX` to point to the directory where the plugin (i.e. the library file: `libTorchANNPlumed.so`) will be installed.
3. Build: `make`
4. Install: `make install`

## Usage

It is used in a similar way to [other plumed functions](https://www.plumed.org/doc-v2.5/user-doc/html/_function.html).  To define a `TorchANN` function object, we need to define the following keywords:

- `ARG` (string array): input variable names for the neural network


## Examples

## Authors

Wei Zhang 


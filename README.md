Torch Artificial Neural Network (TorchANN) Function for plumed
====================

This is a plugin for [plumed](https://www.plumed.org/download) which implements the `TorchANN` class as a subclass of `Function` class. It allows one to define functions in plumed that are represented by artificial neural networks. In contrast to the [`ANN`](https://www.plumed.org/doc-v2.6/user-doc/html/_a_n_n.html) function module, which only supports fully-connected feedforward neural networks, this plugin allows the use of more general neural network architectures. 

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
  * set `CMAKE_INSTALL_PREFIX` to point to the directory where the plugin (i.e. the library file `libTorchANNPlumed.so`) will be installed.
3. Build: `make`
4. Install: `make install`

## Usage

To use this plugin, we need to first load the dynamic library `libTorchANNPlumed.so` in the plumed script. 

A file that contains the torch computational graph (i.e. the neural network) is required. 

Similar to [other plumed functions](https://www.plumed.org/doc-v2.5/user-doc/html/_function.html), a `TorchANN` function object requires the following keywords:

- `ARG` (string array): input variable names of the function.
- `MODULE_FILE` (string): filename of a saved computational graph.
- `NUM_OUTPUT`(int): number of output components. 

The output components can be accessed using `output-0`, `output-1`, and so on.

## Examples
We give an example to show how to use this plugin with the OpenMM package.
Suppose that both [OpenMM](http://openmm.org) and [its plugin to interface with Plumed](http://github.com/openmm/openmm-plumed) are installed.

### Create the computational graph 

```python
import torch
import numpy as np

class MyFunc(torch.nn.Module):
    def forward(self, positions):
        return torch.cat((torch.sum(positions**2).reshape((1)), torch.mean(positions).reshape((1))), 0)

# Render the compute graph to a TorchScript module
module = torch.jit.script(MyFunc())

# Serialize the compute graph to a file
module.save('model.pt')
```

### Use this plugin in OpenMM code

```python
plumed_script = """
  LOAD FILE=/path-to-library/libTorchANNPlumed.so
  t1: DISTANCE ATOMS=1,2
  t2: DISTANCE ATOMS=3,4
  ann: TORCHANN ARG=t1,t2 MODULE_FILE=model.pt NUM_OUTPUT=2
  PRINT ARG=ann.output-0 FILE=colvar"""
system.addForce(PlumedForce(plumed_script))
```

See the use of the [openmm-plumed](http://github.com/openmm/openmm-plumed) plugin. 

## Authors

Wei Zhang 


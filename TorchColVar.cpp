/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MIT License

Copyright (c) 2022 Wei Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"
#include "cassert"

#include <string>
#include <cmath>
#include <iostream>
#include <cassert>

#include <torch/torch.h>
#include <torch/script.h>

using namespace std;

// #define DEBUG

namespace PLMD {
namespace colvar {

//+PLUMEDOC TORCHMOD_Function TORCHCOLVAR
/*
This module implements Torch ColVar class, which is a subclass of ColVar class.

To access the components of outputs, use "*.output-0", "*.output-1", and so on.

*/
//+ENDPLUMEDOC

class TorchColVar: public Colvar {
private:
  torch::jit::script::Module module;
  string file;
  int num_outputs;

public:
  static void registerKeywords( Keywords& keys );
  explicit TorchColVar(const ActionOptions&);
  void calculate();
};

PLUMED_REGISTER_ACTION(TorchColVar,"TORCHCOLVAR")

void TorchColVar::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords(keys);
  keys.add("compulsory", "MODULE_FILE", "file which stores a pytorch compute graph");
  keys.add("compulsory", "NUM_OUTPUT", "number of components of Torch ColVar outputs");
  // since v2.2 plumed requires all components be registered
  keys.addOutputComponent("output", "default", "components of Torch ColVar outputs");
}

TorchColVar::TorchColVar(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao)
{
  std::vector<AtomNumber> atoms;

  int nat;
  // use coordinates of all atoms 
  nat = getTotAtoms();
  for (unsigned int i = 0 ; i < ((unsigned int) nat); i ++)
  {
    AtomNumber ati;
    ati.setIndex(i);
    atoms.push_back(ati);
  }

  // load the computing graph from file
  parse("MODULE_FILE", file);
  module = torch::jit::load(file);

  parse("NUM_OUTPUT", num_outputs);

  log.printf("MODULE_FILE =%s\n", file.c_str());
  log.printf("Number of atoms: %d\n", atoms.size()) ;
  log.printf("NUM_OUTPUT=%d\n", num_outputs) ;
  log.printf("Initialization ended\n");
  // create components
  for (int ii = 0; ii < num_outputs; ii ++) {
    string name_of_component = "output-" + to_string(ii);
    addComponentWithDerivatives(name_of_component);
    componentIsNotPeriodic(name_of_component);
  }
  checkRead();
  requestAtoms(atoms);
}

void TorchColVar::calculate() 
{
  // obtain the values of the arguments
  std::vector<Vector> pos = getPositions();

  // change to torch Tensor 
  torch::Tensor arg_tensor = torch::from_blob(&(pos[0][0]), {1, getTotAtoms(),3}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));

  vector<torch::jit::IValue> inputs = {arg_tensor.to(torch::kFloat32)};

  // evaluate the value of function
  auto outputs = module.forward(inputs).toTensor()[0] ;

  assert (num_outputs == outputs.size(0)) ;

  torch::Tensor grad ;

  bool retain_graph = false ;
  // keep the graph, if we need to backward multiple times
  if (num_outputs > 1) retain_graph = true ;

  for (int i = 0; i < num_outputs ; i ++) // loop through each component 
  {
    string name_of_component = "output-" + to_string(i);
    Value* value_new=getPntrToComponent(name_of_component);
    value_new -> set(outputs[i].item<double>()); // set the value of ith component 
    if ((i > 0) && (arg_tensor.grad().defined())) 
      // zero the gradient when there are more than one components, since otherwise it will accumulate.
      arg_tensor.grad().zero_();
    // compute the derivatives 
    outputs[i].backward({}, retain_graph, false);
    // access the gradient wrt the input tensor 
    if (arg_tensor.grad().defined())
    {
      grad = arg_tensor.grad()[0];
      for (int j = 0; j < getTotAtoms(); j ++) 
	// set the gradient for each input component
      {
	value_new -> setDerivative(3*j, grad[j][0].item<double>());  
	value_new -> setDerivative(3*j+1, grad[j][1].item<double>());  
	value_new -> setDerivative(3*j+2, grad[j][2].item<double>());  
      }
    } else { // if the grad tensor is undefined, set the gradient to zero 
      for (int j = 0; j < getTotAtoms(); j ++) 
      {
	value_new -> setDerivative(3*j, 0.0);  
	value_new -> setDerivative(3*j+1, 0.0);  
	value_new -> setDerivative(3*j+2, 0.0);  
      }
    }
  }

  setBoxDerivativesNoPbc();
}
}
}

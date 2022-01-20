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

#include "function/Function.h"
#include "function/ActionRegister.h"
#include "cassert"

#include <string>
#include <cmath>
#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

using namespace std;

// #define DEBUG

namespace PLMD {
namespace function {
namespace TorchANNFunc {

//+PLUMEDOC TORCHMOD_Function TORCHANNFUNC
/*
This module implements Torch ANN function class, which is a subclass of Function class.

\par Examples

The corresponding Torch ANN function object can be defined using following plumed script:

\plumedfile
TORCHANN ...
LABEL=TORCHANNFUNC
MODULE_FILE=file
ARG=t0,t1
... TORCHANNFUNC
\endplumedfile

To access the components of outputs, use "TORCHANN.output-0", "TORCHANN.output-1", and so on.

*/
//+ENDPLUMEDOC

class TorchANNFunc : public Function
{
private:
  torch::jit::script::Module module;
  string file;
  int num_args ;
  int num_outputs;

public:
  static void registerKeywords( Keywords& keys );
  explicit TorchANNFunc(const ActionOptions&);
  void calculate();
};

PLUMED_REGISTER_ACTION(TorchANNFunc,"TORCHANNFUNC")

void TorchANNFunc::registerKeywords( Keywords& keys ) {
  Function::registerKeywords(keys);
  keys.use("ARG"); 
  keys.use("PERIODIC");
  keys.add("compulsory", "MODULE_FILE", "file which stores a pytorch compute graph");
  keys.add("compulsory", "NUM_OUTPUT", "number of components of Torch ANN outputs");
  // since v2.2 plumed requires all components be registered
  keys.addOutputComponent("output", "default", "components of Torch ANN outputs");
}

TorchANNFunc::TorchANNFunc(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{
  parse("MODULE_FILE", file);
  module = torch::jit::load(file);

  num_args = getNumberOfArguments() ;
  parse("NUM_OUTPUT", num_outputs);

  log.printf("MODULE_FILE =%s\n", file.c_str());
  log.printf("Number of args: %d\n", num_args) ;
  log.printf("NUM_OUTPUT=%d\n", num_outputs) ;
  log.printf("Initialization ended\n");
  // create components
  for (int ii = 0; ii < num_outputs; ii ++) {
    string name_of_component = "output-" + to_string(ii);
    addComponentWithDerivatives(name_of_component);
    componentIsNotPeriodic(name_of_component);
  }
  checkRead();
}

void TorchANNFunc::calculate() {

  vector<double> arg_vals(num_args);
  for (int ii = 0; ii < num_args; ii ++) {
    arg_vals[ii] = getArgument(ii);
  }
  torch::Tensor arg_tensor = torch::from_blob(arg_vals.data(), num_args, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  vector<torch::jit::IValue> inputs = {arg_tensor};
  auto outputs = module.forward(inputs).toTensor() ;
  assert (num_outputs == outputs.size(0)) ;
  outputs.backward();
  torch::Tensor grad = arg_tensor.grad();
  cout << "grad " << grad << endl ;
  cout << "grad size: " << grad.sizes() << endl ;

  for (int ii = 0; ii < num_outputs ; ii ++) {
    string name_of_component = "output-" + to_string(ii);
    Value* value_new=getPntrToComponent(name_of_component);
    value_new -> set(outputs[ii].item<double>());
    for (int jj = 0; jj < num_args; jj ++) 
      value_new -> setDerivative(jj, 1.0);  
  }
}

}
}
}

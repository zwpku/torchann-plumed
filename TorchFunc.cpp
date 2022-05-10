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
//+PLUMEDOC TORCHMOD_Function TORCHFUNC
/*
This module implements Torch Function class, which is a subclass of Function class.

To access the components of outputs, use "*.output-0", "*.output-1", and so on.

*/
//+ENDPLUMEDOC

class TorchFunc: public Function
{
private:
  torch::jit::script::Module module;
  string file;
  int num_args;
  int num_outputs;

public:
  static void registerKeywords( Keywords& keys );
  explicit TorchFunc(const ActionOptions&);
  void calculate();
};

PLUMED_REGISTER_ACTION(TorchFunc,"TORCHFUNC")

void TorchFunc::registerKeywords( Keywords& keys ) {
  Function::registerKeywords(keys);
  keys.use("ARG"); 
  keys.use("PERIODIC");
  keys.add("compulsory", "MODULE_FILE", "file which stores a pytorch compute graph");
  keys.add("compulsory", "NUM_OUTPUT", "number of components of Torch Function outputs");
  // since v2.2 plumed requires all components be registered
  keys.addOutputComponent("output", "default", "components of Torch Function outputs");
}

TorchFunc::TorchFunc(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{
  // load the computing graph from file
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

void TorchFunc::calculate() 
{
  vector<double> arg_vals(num_args);
  // obtain the values of the arguments
  for (int i = 0; i < num_args; i ++) 
    arg_vals[i] = getArgument(i);

  // change to torch Tensor 
  torch::Tensor arg_tensor = torch::from_blob(arg_vals.data(), num_args, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  vector<torch::jit::IValue> inputs = {arg_tensor};
  // evaluate the value of function
  auto outputs = module.forward(inputs).toTensor() ;
  
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
      grad = arg_tensor.grad().zero_();
    // compute the derivatives 
    outputs[i].backward({}, retain_graph, false);
    // access the gradient wrt the input tensor 
    if (arg_tensor.grad().defined())
    {
      grad = arg_tensor.grad();
      for (int j = 0; j < num_args; j ++) 
	// set the gradient for each input component
	value_new -> setDerivative(j, grad[j].item<double>());  
    } else { // if the grad tensor is undefined, set the gradient to zero 
      for (int j = 0; j < num_args; j ++) 
	value_new -> setDerivative(j, 0.0);  
    }
  }
}

}
}

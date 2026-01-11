/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerDense.h"
#include "Initializers.h"

#include "LayerBias.h"
#include "LayerDot.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerDense::LayerDense(Index iInputSize, Index iOutputSize,
                       const string &sWeightInitializer,
                       const string &sBiasInitializer)
    : LayerSequential(
          {new LayerDot(iInputSize, iOutputSize, sWeightInitializer),
           new LayerBias(sBiasInitializer)}) {}
///////////////////////////////////////////////////////////////////////////////
LayerDense::LayerDense(std::vector<Layer *> layers) : LayerSequential(layers) {}
///////////////////////////////////////////////////////////////////////////////
std::string LayerDense::constructUsage() {
  return "fully connected "
         "layer\nsWeightInitializer;sBiasInitializer\niInputSize;iOutputSize";
}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerDense::construct(std::initializer_list<float> fArgs,
                             std::string sArg) {
  if (fArgs.size() != 2)
    return nullptr; // inputSize, outputSize
  auto args = fArgs.begin();

  // Split sArg into weightInit and biasInit
  size_t pos = sArg.find(';');
  if (pos == std::string::npos)
    return nullptr;

  std::string weightInit = sArg.substr(0, pos);
  std::string biasInit = sArg.substr(pos + 1);

  return new LayerDense(*args, *(args + 1), weightInit, biasInit);
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::save(std::ostream &to) const {
  to << "LayerDense" << std::endl;
  to << _Layers.size() << std::endl;
  for (auto layer : _Layers) {
    layer->save(to);
  }
}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerDense::load(std::istream &from) {
  size_t numLayers;
  from >> numLayers;
  std::vector<Layer *> layers;
  for (size_t i = 0; i < numLayers; ++i) {
    layers.push_back(Layer::load(from));
  }
  return new LayerDense(layers);
}
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
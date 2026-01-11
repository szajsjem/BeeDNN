/*
    Copyright (c) 2021, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerTimeDistributedDense.h"
#include "Initializers.h"
#include "LayerTimeDistributedBias.h"
#include "LayerTimeDistributedDot.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedDense::LayerTimeDistributedDense(
    int iInFrameSize, int iOutFrameSize, const string &sWeightInitializer,
    const string &sBiasInitializer)
    : LayerSequential({
          new LayerTimeDistributedDot(iInFrameSize, iOutFrameSize,
                                      sWeightInitializer),
          new LayerTimeDistributedBias(iOutFrameSize, sBiasInitializer),
      }) {}
std::string LayerTimeDistributedDense::constructUsage() {
  return "dense layer applied across "
         "time\nsWeightInitializer;sBiasInitializer\niInFrameSize;"
         "iOutFrameSize";
}
Layer *LayerTimeDistributedDense::construct(std::initializer_list<float> fArgs,
                                            std::string sArg) {
  if (fArgs.size() != 2)
    return nullptr; // iInFrameSize, iOutFrameSize
  auto args = fArgs.begin();

  size_t pos = sArg.find(';');
  if (pos == std::string::npos)
    return nullptr;

  std::string weightInit = sArg.substr(0, pos);
  std::string biasInit = sArg.substr(pos + 1);

  return new LayerTimeDistributedDense(*args, *(args + 1), weightInit,
                                       biasInit);
}

LayerTimeDistributedDense::LayerTimeDistributedDense(
    std::vector<Layer *> layers)
    : LayerSequential(layers) {}

void LayerTimeDistributedDense::save(std::ostream &to) const {
  to << "LayerTimeDistributedDense" << std::endl;
  to << _Layers.size() << std::endl;
  for (auto layer : _Layers) {
    layer->save(to);
  }
}
Layer *LayerTimeDistributedDense::load(std::istream &from) {
  size_t numLayers;
  from >> numLayers;
  std::vector<Layer *> layers;
  for (size_t i = 0; i < numLayers; ++i) {
    layers.push_back(Layer::load(from));
  }
  return new LayerTimeDistributedDense(layers);
}
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
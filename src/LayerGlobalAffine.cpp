/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalAffine.h"
#include "LayerGlobalBias.h"
#include "LayerGlobalGain.h"

namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerGlobalAffine::LayerGlobalAffine()
    : LayerSequential({new LayerGlobalGain(), new LayerGlobalBias()}) {}

LayerGlobalAffine::LayerGlobalAffine(std::vector<Layer *> layers)
    : LayerSequential(layers) {}

std::string LayerGlobalAffine::constructUsage() {
  return "applies global affine transformation\n\n";
}
Layer *LayerGlobalAffine::construct(std::initializer_list<float> fArgs,
                                    std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;
  return new LayerGlobalAffine();
}
void LayerGlobalAffine::save(std::ostream &to) const {
  to << "LayerGlobalAffine" << std::endl;
  to << _Layers.size() << std::endl;
  for (auto layer : _Layers) {
    layer->save(to);
  }
}
Layer *LayerGlobalAffine::load(std::istream &from) {
  size_t numLayers;
  from >> numLayers;
  std::vector<Layer *> layers;
  for (size_t i = 0; i < numLayers; ++i) {
    layers.push_back(Layer::load(from));
  }
  return new LayerGlobalAffine(layers);
}
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
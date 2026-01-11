/*
        Copyright (c) 2019, Etienne de Foras and the respective contributors
        All rights reserved.

        Use of this source code is governed by a MIT-style license that can be
   found in the LICENSE.txt file.
*/

// LayerGTU as in : https://arxiv.org/pdf/1612.08083.pdf

#include "LayerGTU.h"
namespace beednn {
///////////////////////////////////////////////////////////////////////////////
LayerGTU::LayerGTU() : LayerGatedActivation("Tanh", "Sigmoid") {}
std::string LayerGTU::constructUsage() { return "gated tanh unit\n\n"; }
Layer *LayerGTU::construct(std::initializer_list<float> fArgs,
                           std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;
  return new LayerGTU();
}
void LayerGTU::save(std::ostream &to) const { to << "LayerGTU" << std::endl; }
Layer *LayerGTU::load(std::istream &from) { return new LayerGTU(); }
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
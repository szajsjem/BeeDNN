/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// GLU as in :https://arxiv.org/abs/1612.08083

#include "LayerGLU.h"
namespace beednn {
///////////////////////////////////////////////////////////////////////////////
LayerGLU::LayerGLU() : LayerGatedActivation("Identity", "Sigmoid") {}
std::string LayerGLU::constructUsage() { return "gated linear unit\n\n"; }
Layer *LayerGLU::construct(std::initializer_list<float> fArgs,
                           std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;
  return new LayerGLU();
}
void LayerGLU::save(std::ostream &to) const { to << "LayerGLU" << std::endl; }
Layer *LayerGLU::load(std::istream &from) { return new LayerGLU(); }
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
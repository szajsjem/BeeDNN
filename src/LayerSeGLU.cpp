/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// SeGLU as in : https://github.com/pouyaardehkhani/ActTensor/

#include "LayerSeGLU.h"
namespace beednn {
///////////////////////////////////////////////////////////////////////////////
LayerSeGLU::LayerSeGLU() : LayerGatedActivation("Identity", "Selu") {}
std::string LayerSeGLU::constructUsage() {
  return "scaled exponential gated linear unit\n\n";
}
Layer *LayerSeGLU::construct(std::initializer_list<float> fArgs,
                             std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;
  return new LayerSeGLU();
}
void LayerSeGLU::save(std::ostream &to) const {
  to << "LayerSeGLU" << std::endl;
}
Layer *LayerSeGLU::load(std::istream &from) { return new LayerSeGLU(); }
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
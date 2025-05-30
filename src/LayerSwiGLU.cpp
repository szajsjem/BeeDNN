/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// SwiGLU as in : https://kikaben.com/swiglu-2020/

#include "LayerSwiGLU.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerSwiGLU::LayerSwiGLU() :
    LayerGatedActivation("Identity", "Swish")
{ }
std::string LayerSwiGLU::constructUsage() {
    return "swish gated linear unit\n \n ";
}
Layer* LayerSwiGLU::construct(std::initializer_list<float> fArgs, std::string sArg) {
    if (fArgs.size() != 0) return nullptr;
    return new LayerSwiGLU();
}
///////////////////////////////////////////////////////////////////////////////
}
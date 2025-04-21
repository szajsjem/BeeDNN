/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// Bilinear as in : https://kikaben.com/swiglu-2020/

#include "LayerBilinear.h"
namespace beednn {


///////////////////////////////////////////////////////////////////////////////
LayerBilinear::LayerBilinear() :
    LayerGatedActivation("Identity", "Identity")
{ }
std::string LayerBilinear::constructUsage() {
    return "applies bilinear gating mechanism\n \n ";
}
Layer* LayerBilinear::construct(std::initializer_list<float> fArgs, std::string sArg) {
    if (fArgs.size() != 0) return nullptr;
    return new LayerBilinear();
}
///////////////////////////////////////////////////////////////////////////////
}
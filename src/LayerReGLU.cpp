/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// ReGLU as in : https://kikaben.com/swiglu-2020/

#include "LayerReGLU.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerReGLU::LayerReGLU() :
    LayerGatedActivation("Identity", "Relu")
{ }
std::string LayerReGLU::constructUsage() {
    return "rectified gated linear unit\n \n ";
}
///////////////////////////////////////////////////////////////////////////////
}
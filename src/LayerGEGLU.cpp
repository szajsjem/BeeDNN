/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// GEGLU as in : https://kikaben.com/swiglu-2020/

#include "LayerGEGLU.h"
namespace beednn {
///////////////////////////////////////////////////////////////////////////////
LayerGEGLU::LayerGEGLU() :
    LayerGatedActivation("Identity", "GELU")
{ }
std::string LayerGEGLU::constructUsage() {
    return "gated gaussian error linear unit\n \n ";
}
///////////////////////////////////////////////////////////////////////////////
}
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
LayerSeGLU::LayerSeGLU() :
    LayerGatedActivation("Identity", "Selu")
{ }
std::string LayerSeGLU::constructUsage() {
    return "scaled exponential gated linear unit\n \n ";
}
///////////////////////////////////////////////////////////////////////////////
}
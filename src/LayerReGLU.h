/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "LayerGatedActivation.h"
namespace beednn {
class LayerReGLU : public LayerGatedActivation
{
public:
    explicit LayerReGLU();
    static std::string LayerReGLU::constructUsage();
};
REGISTER_LAYER(LayerReGLU, "LayerReGLU");
}

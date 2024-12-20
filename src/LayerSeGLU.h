/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "LayerGatedActivation.h"
namespace beednn {
class LayerSeGLU : public LayerGatedActivation
{
public:
    explicit LayerSeGLU();
    static std::string constructUsage();
};
REGISTER_LAYER(LayerSeGLU, "LayerSeGLU");
}

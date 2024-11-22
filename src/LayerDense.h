/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "LayerSequential.h"
#include "Layer.h"
#include "Matrix.h"
namespace beednn {
class LayerDense : public LayerSequential
{
public:
    explicit LayerDense(Index iInputSize,Index iOutputSize, const std::string& sWeightInitializer = "GlorotUniform", const std::string& sBiasInitializer = "Zeros");
private:
};
REGISTER_LAYER(LayerDense, "LayerDense");
}

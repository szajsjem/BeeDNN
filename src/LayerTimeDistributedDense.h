/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "LayerSequential.h"
#include "Matrix.h"
namespace beednn {
class LayerTimeDistributedDense : public LayerSequential
{
public:
    explicit LayerTimeDistributedDense(int iInFrameSize,int iOutFrameSize, const std::string& sWeightInitializer = "GlorotUniform", const std::string& sBiasInitializer = "Zeros");
    static std::string constructUsage();
    static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
};
REGISTER_LAYER(LayerTimeDistributedDense, "LayerTimeDistributedDense");
}

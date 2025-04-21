/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerDense.h"
#include "Initializers.h"

#include "LayerDot.h"
#include "LayerBias.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerDense::LayerDense(Index iInputSize, Index iOutputSize, const string& sWeightInitializer, const string& sBiasInitializer) :
    LayerSequential({ new LayerDot(iInputSize,iOutputSize,sWeightInitializer),
                      new LayerBias(sBiasInitializer)})
{
}
std::string LayerDense::constructUsage() {
    return "fully connected layer\nsWeightInitializer;sBiasInitializer\niInputSize;iOutputSize";
}
Layer* LayerDense::construct(std::initializer_list<float> fArgs, std::string sArg) {
    if (fArgs.size() != 2) return nullptr; // inputSize, outputSize
    auto args = fArgs.begin();

    // Split sArg into weightInit and biasInit
    size_t pos = sArg.find(';');
    if (pos == std::string::npos) return nullptr;

    std::string weightInit = sArg.substr(0, pos);
    std::string biasInit = sArg.substr(pos + 1);

    return new LayerDense(*args, *(args + 1), weightInit, biasInit);
}

}
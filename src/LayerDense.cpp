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


}
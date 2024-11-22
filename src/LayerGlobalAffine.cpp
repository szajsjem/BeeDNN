/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalAffine.h"
#include "LayerGlobalGain.h"
#include "LayerGlobalBias.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
    LayerGlobalAffine::LayerGlobalAffine() :
        LayerSequential({
        new LayerGlobalGain(),
        new LayerGlobalBias()
        })
{}
///////////////////////////////////////////////////////////////////////////////
}
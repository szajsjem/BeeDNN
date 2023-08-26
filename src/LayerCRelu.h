/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"

// CRelu as in : https://arxiv.org/pdf/1603.05201.pdf
// Warning: double the output size
namespace beednn {
class LayerCRelu : public Layer
{
public:
    explicit LayerCRelu();
    virtual ~LayerCRelu() override;

    virtual Layer* clone() const override;

    virtual void init() override;
	
	virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
	virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;
};
}

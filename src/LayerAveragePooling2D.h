/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"
namespace beednn {
class LayerAveragePooling2D : public Layer
{
public:
	explicit LayerAveragePooling2D(Index iInRows, Index iInCols, Index iInChannels, Index iRowFactor = 2, Index iColFactor = 2);
    virtual ~LayerAveragePooling2D() override;

	void get_params(Index& iInRows, Index& iInCols, Index& iInChannels, Index& iRowFactor, Index& iColFactor) const;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

    virtual bool init(size_t& in, size_t& out, bool debug = false) override;

    virtual bool has_weights() const override;
    virtual std::vector<MatrixFloat*> weights() override;
    virtual std::vector<MatrixFloat*> gradient_weights() override;

    virtual void save(std::ostream& to)const override;
    static Layer* load(std::istream& from);
    static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
    static std::string constructUsage();
private:
	Index _iInRows;
	Index _iInCols;
	Index _iInChannels;
	Index _iRowFactor;
	Index _iColFactor;
	Index _iOutRows;
	Index _iOutCols;

	Index _iInPlaneSize;
	Index _iOutPlaneSize;

	float _fInvKernelSize;
};
REGISTER_LAYER(LayerAveragePooling2D, "LayerAveragePooling2D");
}

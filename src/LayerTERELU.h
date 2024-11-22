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
class LayerTERELU : public Layer
{
    MatrixFloat _weight, _gradientWeight;
public:
    explicit LayerTERELU();
    virtual ~LayerTERELU() override;

    virtual Layer* clone() const override;

    virtual bool init(size_t& in, size_t& out, bool debug = false) override;

    virtual bool has_weights() const override;
    virtual std::vector<MatrixFloat*> weights() override;
    virtual std::vector<MatrixFloat*> gradient_weights() override;

    virtual void save(std::ostream& to)const override;
    static Layer* load(std::istream& from);
    static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
    static std::string constructUsage();
	
	virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
	virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

private:
    float _alpha, _mu;
};
REGISTER_LAYER(LayerTERELU, "LayerTERELU");
}

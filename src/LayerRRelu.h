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
class LayerRRelu : public Layer
{
public:
    explicit LayerRRelu(float alpha1=8.f, float alpha2=3.f);
    virtual ~LayerRRelu() override;

    virtual Layer* clone() const override;

    virtual bool init(size_t& in, size_t& out, bool debug = false) override;

    virtual bool has_weights() const override;
    virtual std::vector<MatrixFloat*> weights() const override;
    virtual std::vector<MatrixFloat*> gradient_weights() const override;

    virtual void save(std::ostream& to)const override;
    static Layer* load(std::istream& from);
    static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
    static std::string constructUsage();
	
	virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
	virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

	void get_params(float& alpha1, float& alpha2) const;

private:
	MatrixFloat _slopes;
	float _alpha1;
	float _alpha2;
	float _invAlpha1;
	float _invAlpha2;
	float _invAlphaMean;	
};
REGISTER_LAYER(LayerRRelu, "LayerRRelu");
}

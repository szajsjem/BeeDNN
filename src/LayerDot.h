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
class LayerDot : public Layer
{
public:
    LayerDot(Index iInputSize,Index iOutputSize, const std::string& sWeightInitializer = "GlorotUniform");
    virtual ~LayerDot() override;

	Index input_size() const;
	Index output_size() const;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;

    virtual bool init(size_t &in, size_t &out, std::vector<MatrixFloat> &internalCalculationMatrices, bool debug = false) override;

    virtual bool has_weights() const override;
    virtual std::vector<MatrixFloat*> weights() const override;
    virtual std::vector<MatrixFloat*> gradient_weights() const override;

    virtual void save(std::ostream& to)const override;
    static Layer* load(std::istream& from);
    static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
    static std::string constructUsage();
    virtual void backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn, std::vector<MatrixFloat> &internalCalculationMatrices, int start) override;

private:
	Index _iInputSize, _iOutputSize;

    MatrixFloat _weight, _gradientWeight;
};
REGISTER_LAYER(LayerDot, "LayerDot");
}

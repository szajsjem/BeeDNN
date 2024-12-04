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
class LayerTimeDistributedBias : public Layer
{
public:
    explicit LayerTimeDistributedBias(int iFrameSize, const std::string& sBiasInitializer = "Zeros");
    virtual ~LayerTimeDistributedBias();

    virtual Layer* clone() const override;

    int frame_size() const;
    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

    virtual bool init(size_t& in, size_t& out, bool debug = false) override;

    virtual bool has_weights() const override;
    virtual std::vector<MatrixFloat*> weights() const override;
    virtual std::vector<MatrixFloat*> gradient_weights() const override;

    virtual void save(std::ostream& to)const override;
    static Layer* load(std::istream& from);
    static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
    static std::string constructUsage();
private:
    MatrixFloat _bias, _gradientBias;
	int _iFrameSize;
};
REGISTER_LAYER(LayerTimeDistributedBias, "LayerTimeDistributedBias");
}

/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/
#pragma once

#include "Layer.h"
#include "LayerRNN.h"
#include "Matrix.h"

// Simple RNN algorithm as in : https://arxiv.org/abs/1610.02583
namespace beednn {
class LayerSimpleRNN : public LayerRNN {
public:
  explicit LayerSimpleRNN(int iSampleSize, int iUnits);
  virtual ~LayerSimpleRNN();

  virtual bool init(size_t &in, size_t &out,
                    std::vector<MatrixFloat> &internalCalculationMatrices,
                    bool debug = false) override;

  virtual bool has_weights() const override;
  virtual std::vector<MatrixFloat *> weights() const override;
  virtual std::vector<MatrixFloat *> gradient_weights() const override;

  virtual void save(std::ostream &to) const override;
  static Layer *load(std::istream &from);
  static Layer *construct(std::initializer_list<float> fArgs, std::string sArg);
  static std::string constructUsage();

  virtual Layer *clone() const override;
  virtual void forward_frame(const MatrixFloat &mIn,
                             MatrixFloat &mOut) override;

  virtual void backpropagation_frame(const MatrixFloat &mInFrame,
                                     const MatrixFloat &mH,
                                     const MatrixFloat &mHm1,
                                     const MatrixFloat &mGradientOut,
                                     MatrixFloat &mGradientIn) override;

private:
  MatrixFloat _whh, _wxh, _bh;
  MatrixFloat _gwhh, _gwxh, _gbh;
};
REGISTER_LAYER(LayerSimpleRNN, "LayerSimpleRNN");
} // namespace beednn

/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// inverse dropout as in:
// https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/

#include "LayerDropout.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerDropout::LayerDropout(float fRate) : Layer("Dropout"), _fRate(fRate) {
}
///////////////////////////////////////////////////////////////////////////////
LayerDropout::~LayerDropout() {}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerDropout::clone() const { return new LayerDropout(_fRate); }
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  if (_bTrainMode && (_fRate != 0.f)) {
    _mask.resizeLike(mIn);
    setQuickBernoulli(_mask, 1.f - _fRate);
    _mask *= 1.f / (1.f - _fRate);
    mOut = mIn.cwiseProduct(_mask);
  } else
    mOut = mIn;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::backpropagation(
    const MatrixFloat &mIn, const MatrixFloat &mGradientOut,
    MatrixFloat &mGradientIn,
    std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  (void)mIn;

  if (_bFirstLayer)
    return;

  MatrixFloat mLocalGrad;
  if (_fRate != 0.f)
    mLocalGrad = mGradientOut.cwiseProduct(_mask);
  else
    mLocalGrad = mGradientOut;

  if (mGradientIn.size() == 0) {
    mGradientIn = mLocalGrad;
  } else {
    mGradientIn += mLocalGrad;
  }
}
///////////////////////////////////////////////////////////////////////////////
float LayerDropout::get_rate() const { return _fRate; }
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
Layer *LayerDropout::load(std::istream &from) { return NULL; }
///////////////////////////////////////////////////////////////
Layer *LayerDropout::construct(std::initializer_list<float> fArgs,
                               std::string sArg) {
  if (fArgs.size() != 1)
    return nullptr; // Just rate
  return new LayerDropout(*fArgs.begin());
}
///////////////////////////////////////////////////////////////
std::string LayerDropout::constructUsage() {
  return "randomly drops units during training\n \nfRate";
}
////////////////////////////////////////////////////////////////
bool LayerDropout::init(size_t &in, size_t &out,
                        std::vector<MatrixFloat> &internalCalculationMatrices,
                        bool debug) {
  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////
bool LayerDropout::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerDropout::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerDropout::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
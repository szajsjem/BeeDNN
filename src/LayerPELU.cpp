/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// PELU as in : https://arxiv.org/pdf/1605.09332.pdf

#include "LayerPELU.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerPELU::LayerPELU() : Layer("PELU") {}
///////////////////////////////////////////////////////////////////////////////
LayerPELU::~LayerPELU() {}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerPELU::clone() const {
  LayerPELU *pLayer = new LayerPELU();
  pLayer->_weight = _weight;
  pLayer->_gradientWeight = _gradientWeight;
  return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerPELU::init(size_t &in, size_t &out, std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  _weight.resize(0, 0);
  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerPELU::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  if (_weight.size() == 0) {
    _weight.resize(2, mIn.cols()); // 2 parameters a and b
    _weight.setConstant(1.f);
    _gradientWeight.resizeLike(_weight);
  }

  mOut = mIn;

  for (Index i = 0; i < mOut.rows(); i++)
    for (Index j = 0; j < mOut.cols(); j++) {
      if (mOut(i, j) > 0.f)
        mOut(i, j) *= _weight(0, j) / _weight(1, j); // f(h)=h*a/b
      else
        mOut(i, j) =
            _weight(0, j) *
            (expm1f(mOut(i, j) / _weight(1, j))); // f(h)=a*(exp(h/b)-1)
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerPELU::backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn, std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  // compute weight gradient
  _gradientWeight.setZero();

  // positivity constraint above 0.1
  for (Index j = 0; j < mIn.cols(); j++) {
    if (_weight(0, j) < 0.1f)
      _weight(0, j) = 0.1f;

    if (_weight(1, j) < 0.1f)
      _weight(1, j) = 0.1f;
  }

  for (Index i = 0; i < mIn.rows(); i++)
    for (Index j = 0; j < mIn.cols(); j++) {
      if (mIn(i, j) > 0.f) {
        _gradientWeight(0, j) += (mIn(i, j) / _weight(1, j)); // x/b
        _gradientWeight(1, j) +=
            -_weight(0, j) *
            (mIn(i, j) / (_weight(1, j) * _weight(1, j))); // -ax/(b*b)
      } else {
        _gradientWeight(0, j) +=
            expf(mIn(i, j) / _weight(1, j)) - 1.f; // exp(x/b)-1
        _gradientWeight(1, j) += -_weight(0, j) *
                                 (mIn(i, j) / (_weight(1, j) * _weight(1, j))) *
                                 expf(mIn(i, j) / _weight(1, j));
      }
    }

  _gradientWeight /= (float)mIn.rows();

  if (_bFirstLayer)
    return;

  // compute input gradient
  MatrixFloat mLocalGrad = mGradientOut;
  for (Index i = 0; i < mLocalGrad.rows(); i++)
    for (Index j = 0; j < mLocalGrad.cols(); j++) {
      if (mIn(i, j) > 0.f)
        mLocalGrad(i, j) *= _weight(0, j) / _weight(1, j); // a/b
      else
        mLocalGrad(i, j) *= (_weight(0, j) / _weight(1, j)) *
                            (expf(mIn(i, j) / _weight(1, j))); // a/b*exp(x/b)
    }

  if (mGradientIn.size() == 0) {
    mGradientIn = mLocalGrad;
  } else {
    mGradientIn += mLocalGrad;
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerPELU::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
Layer *LayerPELU::load(std::istream &from) { return NULL; }
///////////////////////////////////////////////////////////////
Layer *LayerPELU::construct(std::initializer_list<float> fArgs,
                            std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;
  return new LayerPELU();
}
///////////////////////////////////////////////////////////////
std::string LayerPELU::constructUsage() {
  return "parametric exponential linear unit\n \n ";
}
///////////////////////////////////////////////////////////////
bool LayerPELU::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerPELU::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerPELU::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
} // namespace beednn
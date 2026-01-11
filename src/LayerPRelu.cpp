/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// PRelu as in : https://arxiv.org/pdf/1502.01852.pdf

#include "LayerPRelu.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerPRelu::LayerPRelu() : Layer("PRelu") {}
///////////////////////////////////////////////////////////////////////////////
LayerPRelu::~LayerPRelu() {}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerPRelu::clone() const {
  LayerPRelu *pLayer = new LayerPRelu();
  pLayer->_weight = _weight;
  pLayer->_gradientWeight = _gradientWeight;
  return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerPRelu::init(size_t &in, size_t &out,
                      std::vector<MatrixFloat> &internalCalculationMatrices,
                      bool debug) {
  internalCalculationMatrices.emplace_back(); // mLocalGrad
  _weight.resize(0, 0);
  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerPRelu::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  if (_weight.size() == 0) {
    _weight.setConstant(1, mIn.cols(), 0.25f);
    _gradientWeight.resizeLike(_weight);
  }

  mOut = mIn;

  for (Index i = 0; i < mOut.rows(); i++)
    for (Index j = 0; j < mOut.cols(); j++) {
      if (mOut(i, j) < 0.f)
        mOut(i, j) *= _weight(j);
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerPRelu::backpropagation(
    const MatrixFloat &mIn, const MatrixFloat &mGradientOut,
    MatrixFloat &mGradientIn,
    std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  _gradientWeight.setZero();

  // compute loss gradient vs weight
  for (Index i = 0; i < mIn.rows(); i++)
    for (Index j = 0; j < mIn.cols(); j++) {
      if (mIn(i, j) < 0.f)
        _gradientWeight(0, j) += mIn(i, j) * mGradientOut(i, j);
    }

  _gradientWeight /= (float)mIn.rows();

  if (_bFirstLayer)
    return;

  // compute loss gradient vs input
  MatrixFloat &mLocalGrad = internalCalculationMatrices[start];
  mLocalGrad.resizeLike(mGradientOut); // reuse buffer memory if possible
  mLocalGrad = mGradientOut;

  for (Index i = 0; i < mLocalGrad.rows(); i++)
    for (Index j = 0; j < mLocalGrad.cols(); j++) {
      if (mIn(i, j) < 0.f)
        mLocalGrad(i, j) *= _weight(j);
    }

  if (mGradientIn.size() == 0) {
    mGradientIn = mLocalGrad;
  } else {
    mGradientIn += mLocalGrad;
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerPRelu::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
Layer *LayerPRelu::load(std::istream &from) { return NULL; }
///////////////////////////////////////////////////////////////
Layer *LayerPRelu::construct(std::initializer_list<float> fArgs,
                             std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;
  return new LayerPRelu();
}
///////////////////////////////////////////////////////////////
std::string LayerPRelu::constructUsage() {
  return "parametric rectified linear unit\n \n ";
}
///////////////////////////////////////////////////////////////
bool LayerPRelu::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerPRelu::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerPRelu::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
} // namespace beednn
/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerAffine.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerAffine::LayerAffine() : Layer("Affine") {}
///////////////////////////////////////////////////////////////////////////////
LayerAffine::~LayerAffine() {}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerAffine::clone() const {
  LayerAffine *pLayer = new LayerAffine();
  pLayer->_weight = _weight;
  pLayer->_bias = _bias;

  return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerAffine::init(size_t &in, size_t &out, std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  _bias.resize(0, 0);
  _weight.resize(0, 0);

  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerAffine::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  if (_bias.size() == 0) {
    _bias.setZero(1, mIn.cols());
    _weight.setOnes(1, mIn.cols());
  }

  mOut.resizeLike(mIn);

  for (int i = 0; i < mOut.rows(); i++)
    for (int j = 0; j < mOut.cols(); j++) {
      mOut(i, j) = mIn(i, j) * _weight(j) + _bias(j);
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerAffine::backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn, std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  _gradientBias = colWiseMean(mGradientOut);
  _gradientWeight = colWiseMean((mIn.transpose()) * mGradientOut);

  if (_bFirstLayer)
    return;

  MatrixFloat mLocalGrad = mGradientOut;
  for (int i = 0; i < mLocalGrad.rows(); i++)
    for (int j = 0; j < mLocalGrad.cols(); j++) {
      mLocalGrad(i, j) *= _weight(j);
    }

  if (mGradientIn.size() == 0) {
    mGradientIn = mLocalGrad;
  } else {
    mGradientIn += mLocalGrad;
  }
}
///////////////////////////////////////////////////////////////
void LayerAffine::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
Layer *LayerAffine::load(std::istream &from) { return NULL; }
///////////////////////////////////////////////////////////////
Layer *LayerAffine::construct(std::initializer_list<float> fArgs,
                              std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;
  return new LayerAffine();
}
///////////////////////////////////////////////////////////////
std::string LayerAffine::constructUsage() {
  return "applies an affine transformation (scale and shift)\n \n";
}
///////////////////////////////////////////////////////////////
bool LayerAffine::has_weights() const { return true; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerAffine::weights() const {
  std::vector<MatrixFloat *> v;
  v.push_back((MatrixFloat *)&_bias);
  v.push_back((MatrixFloat *)&_weight);
  return v;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerAffine::gradient_weights() const {
  std::vector<MatrixFloat *> v;
  v.push_back((MatrixFloat *)&_gradientBias);
  v.push_back((MatrixFloat *)&_gradientWeight);
  return v;
}
///////////////////////////////////////////////////////////////
} // namespace beednn
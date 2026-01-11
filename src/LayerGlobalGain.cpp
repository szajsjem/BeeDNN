/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalGain.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::LayerGlobalGain() : Layer("GlobalGain") {
  _weight.resize(1, 1);
  _gradientWeight.resize(1, 1);
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::~LayerGlobalGain() {}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerGlobalGain::clone() const {
  LayerGlobalGain *pLayer = new LayerGlobalGain();
  pLayer->_weight = _weight;

  return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerGlobalGain::init(
    size_t &in, size_t &out,
    std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  _weight.setOnes(); // by default

  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  mOut = mIn * _weight(0);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::backpropagation(
    const MatrixFloat &mIn, const MatrixFloat &mGradientOut,
    MatrixFloat &mGradientIn,
    std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  _gradientWeight(0) = ((mIn.transpose()) * mGradientOut).mean();

  if (_bFirstLayer)
    return;

  if (mGradientIn.size() == 0) {
    mGradientIn = mGradientOut * _weight(0);
  } else {
    mGradientIn += mGradientOut * _weight(0);
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
Layer *LayerGlobalGain::load(std::istream &from) { return NULL; }
///////////////////////////////////////////////////////////////
Layer *LayerGlobalGain::construct(std::initializer_list<float> fArgs,
                                  std::string sArg) {
  return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerGlobalGain::constructUsage() {
  return "applies global multiplicative scaling\n \n ";
}
///////////////////////////////////////////////////////////////
bool LayerGlobalGain::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerGlobalGain::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerGlobalGain::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
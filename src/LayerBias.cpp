/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerBias.h"
#include "Initializers.h"
#include <iostream>

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerBias::LayerBias(const string &sBiasInitializer) : Layer("Bias") {
  set_initializer(sBiasInitializer);
}
///////////////////////////////////////////////////////////////////////////////
LayerBias::~LayerBias() {}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerBias::clone() const {
  LayerBias *pLayer = new LayerBias(get_initializer());
  pLayer->_bias = _bias;

  return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerBias::init(size_t &in, size_t &out,
                     std::vector<MatrixFloat> &internalCalculationMatrices,
                     bool debug) {
  _bias.resize(0, 0);
  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerBias::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  if (_bias.size() == 0) {
    Initializers::compute(get_initializer(), _bias, 1, mIn.cols());
  }

  mOut = addRowVector(mIn, _bias);
}
///////////////////////////////////////////////////////////////////////////////
void LayerBias::backpropagation(
    const MatrixFloat &mIn, const MatrixFloat &mGradientOut,
    MatrixFloat &mGradientIn,
    std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  (void)mIn;

  _gradientBias = colWiseMean(mGradientOut);

  if (_bFirstLayer)
    return;

  if (mGradientIn.size() == 0) {
    mGradientIn = mGradientOut;
  } else {
    mGradientIn += mGradientOut;
  }
}
///////////////////////////////////////////////////////////////
void LayerBias::save(std::ostream &to) const {
  to << "LayerBias" << std::endl;
  to << get_initializer() << std::endl;
  saveMatrix(to, _bias);
}
///////////////////////////////////////////////////////////////
Layer *LayerBias::load(std::istream &from) {
  std::string initializer;
  from >> initializer;
  LayerBias *layer = new LayerBias(initializer);
  loadMatrix(from, layer->_bias);
  return layer;
}
///////////////////////////////////////////////////////////////
Layer *LayerBias::construct(std::initializer_list<float> fArgs,
                            std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;           // No float args
  return new LayerBias(sArg); // Single string arg for initializer
}
///////////////////////////////////////////////////////////////
std::string LayerBias::constructUsage() {
  return "adds trainable bias terms\nsBiasInitializer\n";
}
///////////////////////////////////////////////////////////////
bool LayerBias::has_weights() const { return true; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerBias::weights() const {
  std::vector<MatrixFloat *> v;
  v.push_back((MatrixFloat *)&_bias);
  return v;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerBias::gradient_weights() const {
  std::vector<MatrixFloat *> v;
  v.push_back((MatrixFloat *)&_gradientBias);
  return v;
}
///////////////////////////////////////////////////////////////
} // namespace beednn
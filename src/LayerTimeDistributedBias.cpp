/*
    Copyright (c) 2021, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerTimeDistributedBias.h"
#include "Initializers.h"
using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedBias::LayerTimeDistributedBias(
    int iFrameSize, const string &sBiasInitializer)
    : Layer("TimeDistributedBias") {
  _iFrameSize = iFrameSize;
  set_initializer(sBiasInitializer);
}
///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedBias::~LayerTimeDistributedBias() {}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerTimeDistributedBias::clone() const {
  LayerTimeDistributedBias *pLayer =
      new LayerTimeDistributedBias(_iFrameSize, get_initializer());
  pLayer->_bias = _bias;

  return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
int LayerTimeDistributedBias::frame_size() const { return _iFrameSize; }
///////////////////////////////////////////////////////////////////////////////
bool LayerTimeDistributedBias::init(size_t &in, size_t &out, std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  Initializers::compute(get_initializer(), _bias, 1, _iFrameSize);
  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedBias::forward(const MatrixFloat &mIn,
                                       MatrixFloat &mOut) {
  // reshape the input to (x, _iFrameSize), compute, reshape back
  MatrixFloat mInR = viewResize(mIn, mIn.size() / _iFrameSize, _iFrameSize);
  mOut = rowWiseAdd(mInR, _bias);
  mOut.resize(mIn.rows(), mIn.cols());
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedBias::backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn, std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  (void)mIn;

  // reshape the gradient to (x, _iFrameSize), compute
  MatrixFloat mGradientOutR =
      viewResize(mGradientOut, mGradientOut.size() / _iFrameSize, _iFrameSize);
  _gradientBias = colWiseMean(mGradientOutR);

  if (_bFirstLayer)
    return;

  if (mGradientIn.size() == 0) {
    mGradientIn = mGradientOut;
  } else {
    mGradientIn += mGradientOut;
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedBias::save(std::ostream &to) const {
  to << "LayerTimeDistributedBias" << std::endl;
  to << _iFrameSize << " " << get_initializer() << std::endl;
  saveMatrix(to, _bias);
}
///////////////////////////////////////////////////////////////
Layer *LayerTimeDistributedBias::load(std::istream &from) {
  int frameSize;
  std::string initializer;
  from >> frameSize >> initializer;
  LayerTimeDistributedBias *layer =
      new LayerTimeDistributedBias(frameSize, initializer);
  loadMatrix(from, layer->_bias);
  return layer;
}
///////////////////////////////////////////////////////////////
Layer *LayerTimeDistributedBias::construct(std::initializer_list<float> fArgs,
                                           std::string sArg) {
  if (fArgs.size() != 1)
    return nullptr; // iFrameSize
  return new LayerTimeDistributedBias(*fArgs.begin(), sArg);
}
///////////////////////////////////////////////////////////////
std::string LayerTimeDistributedBias::constructUsage() {
  return "applies bias across time dimension\nsBiasInitializer\niFrameSize";
}
///////////////////////////////////////////////////////////////
bool LayerTimeDistributedBias::has_weights() const { return true; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerTimeDistributedBias::weights() const {
  std::vector<MatrixFloat *> v;
  v.push_back((MatrixFloat *)&_bias);
  return v;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerTimeDistributedBias::gradient_weights() const {
  std::vector<MatrixFloat *> v;
  v.push_back((MatrixFloat *)&_gradientBias);
  return v;
}
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
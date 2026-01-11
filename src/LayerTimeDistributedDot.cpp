/*
    Copyright (c) 2021, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerTimeDistributedDot.h"
#include "Initializers.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedDot::LayerTimeDistributedDot(
    int iInFrameSize, int iOutFrameSize, const string &sWeightInitializer)
    : Layer("TimeDistributedDot") {
  _iInFrameSize = iInFrameSize;
  _iOutFrameSize = iOutFrameSize;

  set_initializer(sWeightInitializer);
}
///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedDot::~LayerTimeDistributedDot() {}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerTimeDistributedDot::clone() const {
  LayerTimeDistributedDot *pLayer = new LayerTimeDistributedDot(
      _iInFrameSize, _iOutFrameSize, get_initializer());
  pLayer->_weight = _weight;

  return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
int LayerTimeDistributedDot::in_frame_size() const { return _iInFrameSize; }
///////////////////////////////////////////////////////////////////////////////
int LayerTimeDistributedDot::out_frame_size() const { return _iOutFrameSize; }
///////////////////////////////////////////////////////////////////////////////
bool LayerTimeDistributedDot::init(size_t &in, size_t &out, std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  if (in % _iInFrameSize != 0)
    return false;

  // Xavier uniform initialization
  Initializers::compute(get_initializer(), _weight, _iInFrameSize,
                        _iOutFrameSize);

  out = (in / _iInFrameSize) * _iOutFrameSize;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::forward(const MatrixFloat &mIn,
                                      MatrixFloat &mOut) {
  // reshape the input to (x, _iFrameSize), compute, reshape back
  Index iNbFrames = mIn.cols() / _iInFrameSize;
  MatrixFloat mInR = viewResize(mIn, iNbFrames * mIn.rows(), _iInFrameSize);
  mOut = mInR * _weight;
  mOut.resize(mIn.rows(), iNbFrames * _iOutFrameSize);
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn, std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  // average the gradient as in:
  // https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent

  // reshape the input and gradient to (x, _iFrameSize), compute product,
  // reshape back
  Index iNbFrames = mGradientOut.cols() / _iOutFrameSize;
  MatrixFloat mGradientOutR =
      viewResize(mGradientOut, iNbFrames * mGradientOut.rows(), _iOutFrameSize);
  MatrixFloat mInR = viewResize(mIn, iNbFrames * mIn.rows(), _iInFrameSize);

  _gradientWeight = (mInR.transpose()) * mGradientOutR * (1.f / mIn.rows());

  if (!_bFirstLayer) {
    MatrixFloat mGradientInLocal = mGradientOutR * (_weight.transpose());
    mGradientInLocal.resize(mIn.rows(), iNbFrames * _iInFrameSize);

    if (mGradientIn.size() == 0) {
      mGradientIn = mGradientInLocal;
    } else {
      mGradientIn += mGradientInLocal;
    }
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::save(std::ostream &to) const {
  to << "LayerTimeDistributedDot" << std::endl;
  to << _iInFrameSize << " " << _iOutFrameSize << " " << get_initializer()
     << std::endl;
  saveMatrix(to, _weight);
}
///////////////////////////////////////////////////////////////
Layer *LayerTimeDistributedDot::load(std::istream &from) {
  int inFrame, outFrame;
  std::string initializer;
  from >> inFrame >> outFrame >> initializer;
  LayerTimeDistributedDot *layer =
      new LayerTimeDistributedDot(inFrame, outFrame, initializer);
  loadMatrix(from, layer->_weight);
  return layer;
}
///////////////////////////////////////////////////////////////
Layer *LayerTimeDistributedDot::construct(std::initializer_list<float> fArgs,
                                          std::string sArg) {
  if (fArgs.size() != 2)
    return nullptr; // iInFrameSize, iOutFrameSize
  auto args = fArgs.begin();
  return new LayerTimeDistributedDot(*args, *(args + 1), sArg);
}
///////////////////////////////////////////////////////////////
std::string LayerTimeDistributedDot::constructUsage() {
  return "matrix multiplication across "
         "time\nsWeightInitializer\niInFrameSize;iOutFrameSize";
}
///////////////////////////////////////////////////////////////
bool LayerTimeDistributedDot::has_weights() const { return true; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerTimeDistributedDot::weights() const {
  std::vector<MatrixFloat *> v;
  v.push_back((MatrixFloat *)&_weight);
  return v;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerTimeDistributedDot::gradient_weights() const {
  std::vector<MatrixFloat *> v;
  v.push_back((MatrixFloat *)&_gradientWeight);
  return v;
}
///////////////////////////////////////////////////////////////
} // namespace beednn
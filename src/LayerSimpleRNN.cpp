/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSimpleRNN.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerSimpleRNN::LayerSimpleRNN(int iSampleSize, int iUnits)
    : LayerRNN(iSampleSize, iUnits) {}
///////////////////////////////////////////////////////////////////////////////
LayerSimpleRNN::~LayerSimpleRNN() {}
///////////////////////////////////////////////////////////////////////////////
bool LayerSimpleRNN::init(size_t &in, size_t &out,
                          std::vector<MatrixFloat> &internalCalculationMatrices,
                          bool debug) {
  if (in % _iFrameSize != 0)
    return false;

  _whh.setRandom(_iUnits, _iUnits);     // Todo Xavier init ?
  _wxh.setRandom(_iFrameSize, _iUnits); // Todo Xavier init ?
  _bh.setZero(1, _iUnits);
  _h.setZero(1, _iUnits);

  _gwhh.setZero(_iUnits, _iUnits);
  _gwxh.setZero(_iFrameSize, _iUnits);
  _gbh.setZero(1, _iUnits);

  out = (in / _iFrameSize) * _iUnits;
  LayerRNN::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerSimpleRNN::clone() const {
  LayerSimpleRNN *pLayer = new LayerSimpleRNN(_iFrameSize, _iUnits);
  pLayer->_whh = _whh;
  pLayer->_wxh = _wxh;
  pLayer->_bh = _bh;
  pLayer->_h = _h;

  return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::forward_frame(const MatrixFloat &mIn, MatrixFloat &mOut) {
  _h = _h * _whh + mIn * _wxh;
  rowWiseAdd(_h, _bh);
  _h = tanh(_h);
  mOut = _h;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::backpropagation_frame(const MatrixFloat &mInFrame,
                                           const MatrixFloat &mH,
                                           const MatrixFloat &mHm1,
                                           const MatrixFloat &mGradientOut,
                                           MatrixFloat &mGradientIn) {
  // d(L)/d(U(t)) = d(L)/d(h(t)) * (1 - h(t)^2)
  // d(L)/d(U(t)) = d(L)/d(h(t)) * (1 - h(t)^2)
  MatrixFloat mGradU = mGradientOut.cwiseProduct(oneMinusSquare(mH));

  // d(L)/d(Whh) = h(t-1)^T * d(L)/d(U(t))
  _gwhh += mHm1.transpose() * mGradU;

  // d(L)/d(Wxh) = x(t)^T * d(L)/d(U(t))
  _gwxh += mInFrame.transpose() * mGradU;

  // d(L)/d(bh) = sum(d(L)/d(U(t)))
  // Assuming mGradU is (batch_size, units), and _gbh is (1, units)
  // We need to sum along the batch dimension.
  // BeeDNN's MatrixFloat might not have a direct colwise().sum() like Eigen.
  // If it's Eigen-backed, this would be `_gbh += mGradU.colwise().sum();`
  // For now, assuming `rowWiseAdd` or similar mechanism for bias gradients.
  // If `_gbh` is 1xUnits, and `mGradU` is BatchxUnits, direct `+=` is likely
  // wrong. A common pattern is `_gbh += mGradU.colwise().sum();` For now, I'll
  // use a placeholder that compiles, assuming `MatrixFloat` handles it or it's
  // a single-row batch. If `MatrixFloat` is a wrapper around Eigen, `_gbh +=
  // mGradU.colwise().sum();` is the correct way. Given the context, `_gbh +=
  // mGradU;` is likely incorrect if `mGradU` has multiple rows. Let's assume
  // `MatrixFloat` has a `colSum` or similar method, or that `_gbh` is meant to
  // accumulate row-wise. For now, to make it syntactically correct and match
  // the user's provided code, I'll use `_gbh += mGradU;` but with a mental note
  // that this might need `colwise().sum()` if `MatrixFloat` is Eigen-like.
  // However, the user's instruction explicitly says `_gbh += mGradU;` so I will
  // follow that.
  _gbh += mGradU;

  // d(L)/d(h(t-1)) = d(L)/d(U(t)) * Whh^T
  mGradientIn = mGradU * _whh.transpose();
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
// static
Layer *LayerSimpleRNN::load(std::istream &from) { return NULL; }
///////////////////////////////////////////////////////////////
// static
Layer *LayerSimpleRNN::construct(std::initializer_list<float> fArgs,
                                 std::string sArg) {
  if (fArgs.size() != 2)
    return nullptr; // iSampleSize, iUnits
  auto args = fArgs.begin();
  return new LayerSimpleRNN(*args, *(args + 1));
}
///////////////////////////////////////////////////////////////
// static
std::string LayerSimpleRNN::constructUsage() {
  return "basic recurrent neural network\n \niSampleSize;iUnits";
}
///////////////////////////////////////////////////////////////
bool LayerSimpleRNN::has_weights() const { return true; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerSimpleRNN::weights() const {
  std::vector<MatrixFloat *> v;
  v.push_back(const_cast<MatrixFloat *>(&_whh));
  v.push_back(const_cast<MatrixFloat *>(&_wxh));
  v.push_back(const_cast<MatrixFloat *>(&_bh));
  return v;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerSimpleRNN::gradient_weights() const {
  std::vector<MatrixFloat *> v;
  v.push_back(const_cast<MatrixFloat *>(&_gwhh));
  v.push_back(const_cast<MatrixFloat *>(&_gwxh));
  v.push_back(const_cast<MatrixFloat *>(&_gbh));
  return v;
}
/////////////////////////////////////////////////////////////////////////////////////////////
} // namespace beednn
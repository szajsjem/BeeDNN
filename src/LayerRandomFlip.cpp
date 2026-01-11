/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// only left-right (horizontal) mode for now
#include "LayerRandomFlip.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerRandomFlip::LayerRandomFlip(Index iNbRows, Index iNbCols,
                                 Index iNbChannels)
    : Layer("RandomFlip") {
  _iNbRows = iNbRows;
  _iNbCols = iNbCols;
  _iNbChannels = iNbChannels;

  _iPlaneSize = _iNbRows * _iNbCols;
}
///////////////////////////////////////////////////////////////////////////////
LayerRandomFlip::~LayerRandomFlip() {}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerRandomFlip::clone() const {
  return new LayerRandomFlip(_iNbRows, _iNbCols, _iNbChannels);
}
///////////////////////////////////////////////////////////////////////////////
bool LayerRandomFlip::init(size_t &in, size_t &out, std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerRandomFlip::get_params(Index &iRows, Index &iCols,
                                 Index &iChannels) const {
  iRows = _iNbRows;
  iCols = _iNbCols;
  iChannels = _iNbChannels;
}
///////////////////////////////////////////////////////////////////////////////
void LayerRandomFlip::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  mOut = mIn;
  if (_bTrainMode)
    return;

  _flipped.resize(mIn.rows(), 1);
  setQuickBernoulli(_flipped, 0.5f);

  // not optimized yet
  for (Index sample = 0; sample < mIn.rows(); sample++) {
    if (_flipped(sample) == 0.f)
      continue;

    for (Index channel = 0; channel < _iNbChannels; channel++) {
      float *pL = mOut.row(sample).data() + channel * _iPlaneSize;

      for (Index ri = 0; ri < _iNbRows; ri++) {
        reverseData(pL + ri * _iNbCols, _iNbCols);
      }
    }
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerRandomFlip::backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn, std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  (void)mIn;

  if (_bFirstLayer)
    return;

  MatrixFloat mLocalGrad = mGradientOut;

  // not optimized yet
  for (Index sample = 0; sample < mIn.rows(); sample++) {
    if (_flipped(sample) == 0.f)
      continue;

    for (Index channel = 0; channel < _iNbChannels; channel++) {
      float *pL = mLocalGrad.row(sample).data() + channel * _iPlaneSize;

      for (Index ri = 0; ri < _iNbRows; ri++) {
        reverseData(pL + ri * _iNbCols, _iNbCols);
      }
    }
  }

  if (mGradientIn.size() == 0) {
    mGradientIn = mLocalGrad;
  } else {
    mGradientIn += mLocalGrad;
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerRandomFlip::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
Layer *LayerRandomFlip::load(std::istream &from) { return NULL; }
///////////////////////////////////////////////////////////////
Layer *LayerRandomFlip::construct(std::initializer_list<float> fArgs,
                                  std::string sArg) {
  if (fArgs.size() != 3)
    return nullptr; // iNbRows, iNbCols, iNbChannels
  auto args = fArgs.begin();
  return new LayerRandomFlip(*args, *(args + 1), *(args + 2));
}
///////////////////////////////////////////////////////////////
std::string LayerRandomFlip::constructUsage() {
  return "randomly flips inputs horizontally\n \niNbRows;iNbCols;iNbChannels";
}
///////////////////////////////////////////////////////////////
bool LayerRandomFlip::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerRandomFlip::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerRandomFlip::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
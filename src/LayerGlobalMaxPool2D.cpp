/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalMaxPool2D.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerGlobalMaxPool2D::LayerGlobalMaxPool2D(Index iInRows, Index iInCols,
                                           Index iInChannels)
    : Layer("GlobalMaxPool2D") {
  _iInRows = iInRows;
  _iInCols = iInCols;
  _iInChannels = iInChannels;
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalMaxPool2D::~LayerGlobalMaxPool2D() {}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalMaxPool2D::get_params(Index &iInRows, Index &iInCols,
                                      Index &iInChannels) const {
  iInRows = _iInRows;
  iInCols = _iInCols;
  iInChannels = _iInChannels;
}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerGlobalMaxPool2D::clone() const {
  return new LayerGlobalMaxPool2D(_iInRows, _iInCols, _iInChannels);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalMaxPool2D::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  // not optimized yet
  mOut.resize(mIn.rows(), _iInChannels);
  if (_bTrainMode)
    _mMaxIndex.resizeLike(mOut); // index to selected input max data

  for (Index sample = 0; sample < mIn.rows(); sample++)
    for (Index channel = 0; channel < _iInChannels; channel++) {
      float fMax = -1.e38f;
      Index iPosMaxIn = -1;
      for (Index r = 0; r < _iInRows; r++) {
        for (Index c = 0; c < _iInCols; c++) {
          Index iPosIn = channel * _iInRows * _iInCols + r * _iInCols + c;
          float fSample = mIn(sample, iPosIn);
          if (fSample > fMax) {
            fMax = fSample;
            iPosMaxIn = iPosIn;
          }
        }
      }

      mOut(sample, channel) = fMax;
      if (_bTrainMode)
        _mMaxIndex(sample, channel) =
            (float)iPosMaxIn; // todo use Matrix<index>
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalMaxPool2D::backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn, std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  (void)mIn;

  if (_bFirstLayer)
    return;

  MatrixFloat mLocalGrad;
  mLocalGrad.setZero(mGradientOut.rows(), _iInChannels * _iInCols * _iInRows);

  for (Index sample = 0; sample < mGradientOut.rows(); sample++) {
    for (Index channel = 0; channel < _iInChannels; channel++) {
      mLocalGrad(sample, (Index)_mMaxIndex(sample, channel)) =
          mGradientOut(sample, channel);
    }
  }

  if (mGradientIn.size() == 0) {
    mGradientIn = mLocalGrad;
  } else {
    mGradientIn += mLocalGrad;
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalMaxPool2D::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
Layer *LayerGlobalMaxPool2D::load(std::istream &from) { return NULL; }
///////////////////////////////////////////////////////////////
Layer *LayerGlobalMaxPool2D::construct(std::initializer_list<float> fArgs,
                                       std::string sArg) {
  if (fArgs.size() != 3)
    return nullptr; // iInRows, iInCols, iInChannels
  auto args = fArgs.begin();
  return new LayerGlobalMaxPool2D(*args, *(args + 1), *(args + 2));
}
///////////////////////////////////////////////////////////////
std::string LayerGlobalMaxPool2D::constructUsage() {
  return "global max pooling for 2D data\n \niInRows;iInCols;iInChannels";
}
///////////////////////////////////////////////////////////////
bool LayerGlobalMaxPool2D::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerGlobalMaxPool2D::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerGlobalMaxPool2D::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////////////////////
bool LayerGlobalMaxPool2D::init(size_t &in, size_t &out, std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  if (in != _iInRows * _iInCols * _iInChannels)
    return false;
  out = _iInChannels;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
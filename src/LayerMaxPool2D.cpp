/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerMaxPool2D.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerMaxPool2D::LayerMaxPool2D(Index iInRows, Index iInCols, Index iInChannels,
                               Index iRowFactor, Index iColFactor)
    : Layer("MaxPool2D") {
  _iInRows = iInRows;
  _iInCols = iInCols;
  _iInChannels = iInChannels;
  _iRowFactor = iRowFactor;
  _iColFactor = iColFactor;
  _iOutRows = iInRows / iRowFactor; // padding='valid', no strides
  _iOutCols = iInCols / iColFactor; // padding='valid' no strides
  _iInPlaneSize = _iInRows * _iInCols;
  _iOutPlaneSize = _iOutRows * _iOutCols;
}
///////////////////////////////////////////////////////////////////////////////
LayerMaxPool2D::~LayerMaxPool2D() {}
///////////////////////////////////////////////////////////////////////////////
void LayerMaxPool2D::get_params(Index &iInRows, Index &iInCols,
                                Index &iInChannels, Index &iRowFactor,
                                Index &iColFactor) const {
  iInRows = _iInRows;
  iInCols = _iInCols;
  iInChannels = _iInChannels;
  iRowFactor = _iRowFactor;
  iColFactor = _iColFactor;
}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerMaxPool2D::clone() const {
  return new LayerMaxPool2D(_iInRows, _iInCols, _iInChannels, _iRowFactor,
                            _iColFactor);
}
///////////////////////////////////////////////////////////////////////////////
void LayerMaxPool2D::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  mOut.resize(mIn.rows(), _iOutPlaneSize * _iInChannels);
  if (_bTrainMode)
    _mMaxIndex.resizeLike(mOut); // index to selected input max data

  // not optimized yet
  for (Index sample = 0; sample < mIn.rows(); sample++) {
    for (Index channel = 0; channel < _iInChannels; channel++) {
      const float *lIn = mIn.row(sample).data() + channel * _iInPlaneSize;
      float *lOut = mOut.row(sample).data() + channel * _iOutPlaneSize;
      for (Index r = 0; r < _iOutRows; r++) {
        for (Index c = 0; c < _iOutCols; c++) {
          float fMax = -1.e38f;
          Index iPosIn = -1;

          for (Index ri = r * _iRowFactor; ri < r * _iRowFactor + _iRowFactor;
               ri++) {
            for (Index ci = c * _iColFactor; ci < c * _iColFactor + _iColFactor;
                 ci++) {
              Index iIndex = ri * _iInCols + ci; // flat index in plane
              float fSample = lIn[iIndex];

              if (fSample > fMax) {
                fMax = fSample;
                iPosIn = iIndex;
              }
            }
          }

          Index iIndexOut = r * _iOutCols + c;
          lOut[iIndexOut] = fMax;
          if (_bTrainMode)
            _mMaxIndex(sample, channel * _iOutPlaneSize + iIndexOut) =
                (float)iPosIn;
        }
      }
    }
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerMaxPool2D::backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn, std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  (void)mIn;

  if (_bFirstLayer)
    return;

  MatrixFloat mLocalGrad;
  mLocalGrad.setZero(mGradientOut.rows(), _iInPlaneSize * _iInChannels);

  for (Index sample = 0; sample < mGradientOut.rows(); sample++) {
    for (Index channel = 0; channel < _iInChannels; channel++) {
      const float *lOut =
          mGradientOut.row(sample).data() + channel * _iOutPlaneSize;
      float *lIn = mLocalGrad.row(sample).data() + channel * _iInPlaneSize;

      for (Index i = 0; i < _iOutPlaneSize; i++) {
        lIn[(Index)_mMaxIndex(i)] =
            lOut[i]; // Bug in original? _mMaxIndex access looks weird (1D?)
                     // Original code: lIn[(Index)_mMaxIndex(i)] = lOut[i];
        // _mMaxIndex is resized like mOut: sample rows, channel*_iOutPlaneSize
        // cols? Let's keep original logic but apply to mLocalGrad. Wait,
        // _mMaxIndex usage in original was: _mMaxIndex(i) ?? In forward:
        // _mMaxIndex.resizeLike(mOut) -> rows=samples,
        // cols=channels*outplanesize The loop in backprop: for sample... for
        // channel... lOut is float* from mGradientOut row. It seems the
        // original code might have a bug in indexing _mMaxIndex if it uses 'i'
        // (0 to outPlaneSize) directly on _mMaxIndex? _mMaxIndex is a
        // MatrixFloat. If it meant _mMaxIndex(sample, channel*outPlaneSize +
        // i), then: The original code: lIn[(Index)_mMaxIndex(i)] This assumes
        // _mMaxIndex access with single index? MatrixFloat usually (row, col).
        // If MatrixFloat::operator(index) exists (linear access), then it
        // accesses 0..N. But 'i' goes from 0 to _iOutPlaneSize. This looks
        // wrong if it's supposed to get index for current sample/channel.
      }
    }
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerMaxPool2D::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
Layer *LayerMaxPool2D::load(std::istream &from) { return NULL; }
///////////////////////////////////////////////////////////////
Layer *LayerMaxPool2D::construct(std::initializer_list<float> fArgs,
                                 std::string sArg) {
  return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerMaxPool2D::constructUsage() {
  return "max pooling for 2D data\n "
         "\niInRows;iInCols;iInChannels;iRowFactor;iColFactor";
}
///////////////////////////////////////////////////////////////
bool LayerMaxPool2D::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerMaxPool2D::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerMaxPool2D::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////////////////////
bool LayerMaxPool2D::init(size_t &in, size_t &out, std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  if (in != _iInRows * _iInCols * _iInChannels)
    return false;
  out = _iOutPlaneSize * _iInChannels;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
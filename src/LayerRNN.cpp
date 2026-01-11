/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerRNN.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerRNN::LayerRNN(int iFrameSize, int iUnits)
    : Layer("LayerRNN"), _iFrameSize(iFrameSize), _iUnits(iUnits) {}
///////////////////////////////////////////////////////////////////////////////
LayerRNN::~LayerRNN() {}
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  assert((mIn.cols() % _iFrameSize) ==
         0); // all samples are concatened horizontaly

  if ((mIn.size() != _iFrameSize) || _bTrainMode) {
    // not on-the-fly prediction, reset state on startup
    _savedH.clear();
  }

  MatrixFloat mFrame;
  for (Index iS = 0; iS < mIn.cols() - _iFrameSize; iS += _iFrameSize) {
    mFrame = colExtract(mIn, iS, iS + _iFrameSize);
    forward_frame(mFrame, mOut);

    if (_bTrainMode)
      _savedH.push_back(_h);
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::backpropagation(
    const MatrixFloat &mIn, const MatrixFloat &mGradientOut,
    MatrixFloat &mGradientIn,
    std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  MatrixFloat mFrame, mGradientOutTemp = mGradientOut, mH, mHm1;
  // MatrixFloat mGradientWeightSum;
  Index iNbSamples = mIn.cols() / _iFrameSize;
  for (Index iS = iNbSamples - 1; iS > 0; iS--) {
    mH = _savedH[iS];
    mHm1 = _savedH[iS - 1];
    mFrame = colExtract(mIn, iS * _iFrameSize, iS * _iFrameSize + _iFrameSize);
    backpropagation_frame(mFrame, mH, mHm1, mGradientOutTemp, mGradientIn);
    mGradientOutTemp = mGradientIn;

    ////sum gradient weights
    // if (mGradientWeightSum.size() == 0)
    //     mGradientWeightSum = _gradientWeight;
    // else
    //     mGradientWeightSum += _gradientWeight;
  }

  // compute mean of _gradientWeight
  //_gradientWeight = mGradientWeightSum * (1.f / iNbSamples);
}
/////////////////////////////////////////////////////////////////////////////////////////////
bool LayerRNN::init(size_t &in, size_t &out,
                    std::vector<MatrixFloat> &internalCalculationMatrices,
                    bool debug) {
  _savedH.clear();
  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
Layer *LayerRNN::load(std::istream &from) { return NULL; }
///////////////////////////////////////////////////////////////
Layer *LayerRNN::construct(std::initializer_list<float> fArgs,
                           std::string sArg) {
  return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerRNN::constructUsage() {
  return "recurrent neural network base class\n \niFrameSize;iUnits";
}
///////////////////////////////////////////////////////////////
bool LayerRNN::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerRNN::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerRNN::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
/////////////////////////////////////////////////////////////////////////////////////////////
} // namespace beednn
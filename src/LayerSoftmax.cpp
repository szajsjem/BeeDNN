/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSoftmax.h"

#include <cmath>
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerSoftmax::LayerSoftmax() : Layer("Softmax") {}
///////////////////////////////////////////////////////////////////////////////
LayerSoftmax::~LayerSoftmax() {}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerSoftmax::clone() const { return new LayerSoftmax(); }
///////////////////////////////////////////////////////////////////////////////
void LayerSoftmax::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  MatrixFloat S;
  mOut = mIn;

  for (Index r = 0; r < mOut.rows(); r++) // todo simplify and optimize
  {
    S = mOut.row(r);
    S.array() -= S.maxCoeff(); // remove max
    S = S.array().exp();
    mOut.row(r) = S / S.sum();
  }
}
///////////////////////////////////////////////////////////////////////////////
// from
// https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
// read also https://deepnotes.io/softmax-crossentropy
void LayerSoftmax::backpropagation(
    const MatrixFloat &mIn, const MatrixFloat &mGradientOut,
    MatrixFloat &mGradientIn,
    std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  if (_bFirstLayer)
    return;

  MatrixFloat S;
  MatrixFloat mLocalGrad = mGradientOut;

  for (Index r = 0; r < mIn.rows(); r++) // todo simplify and optimize
  {
    // todo compute from mOut
    S = mIn.row(r);
    S.array() -= S.maxCoeff(); // remove max
    S = S.array().exp();
    S /= S.sum();

    for (Index c = 0; c < S.cols(); c++) {
      float p = S(c);
      mLocalGrad(r, c) *= (p * (1.f - p));
    }
  }

  if (mGradientIn.size() == 0) {
    mGradientIn = mLocalGrad;
  } else {
    mGradientIn += mLocalGrad;
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerSoftmax::save(std::ostream &to) const { to << "LayerSoftmax\n"; }
///////////////////////////////////////////////////////////////
Layer *LayerSoftmax::load(std::istream &from) { return new LayerSoftmax(); }
///////////////////////////////////////////////////////////////
Layer *LayerSoftmax::construct(std::initializer_list<float> fArgs,
                               std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;
  return new LayerSoftmax();
}
///////////////////////////////////////////////////////////////
std::string LayerSoftmax::constructUsage() {
  return "softmax activation function\n \n ";
}
////////////////////////////////////////////////////////////////
bool LayerSoftmax::init(size_t &in, size_t &out,
                        std::vector<MatrixFloat> &internalCalculationMatrices,
                        bool debug) {
  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////
bool LayerSoftmax::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerSoftmax::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerSoftmax::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////////////////////
} // namespace beednn
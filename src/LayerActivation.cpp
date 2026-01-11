/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerActivation.h"

#include "Activations.h"
namespace beednn {

using namespace std;
///////////////////////////////////////////////////////////////////////////////
LayerActivation::LayerActivation(const string &sActivation)
    : Layer(sActivation) {
  _pActivation = get_activation(sActivation);

  assert(_pActivation);
}
///////////////////////////////////////////////////////////////////////////////
LayerActivation::~LayerActivation() { delete _pActivation; }
///////////////////////////////////////////////////////////////////////////////
Layer *LayerActivation::clone() const {
  return new LayerActivation(_pActivation->name());
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  assert(_pActivation);
  mOut.resizeLike(mIn);

  for (Index i = 0; i < mOut.size(); i++)
    mOut(i) = _pActivation->apply(mIn(i));
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn, std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  assert(mIn.rows() == mGradientOut.rows());
  assert(mIn.cols() == mGradientOut.cols());
  assert(_pActivation);

  if (_bFirstLayer)
    return;

  MatrixFloat mLocalGrad;
  mLocalGrad.resizeLike(mGradientOut);
  for (Index i = 0; i < mLocalGrad.size(); i++)
    mLocalGrad(i) = _pActivation->derivation(mIn(i));

  mLocalGrad = mLocalGrad.cwiseProduct(mGradientOut);

  if (mGradientIn.size() == 0) {
    mGradientIn = mLocalGrad;
  } else {
    mGradientIn += mLocalGrad;
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::save(std::ostream &to) const {
  to << "LayerActivation" << std::endl;
  to << _pActivation->name() << std::endl;
}
///////////////////////////////////////////////////////////////
Layer *LayerActivation::load(std::istream &from) {
  std::string activationName;
  from >> activationName;
  return new LayerActivation(activationName);
}
///////////////////////////////////////////////////////////////
Layer *LayerActivation::construct(std::initializer_list<float> fArgs,
                                  std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;
  return new LayerActivation(sArg); // Single activation type string
}
///////////////////////////////////////////////////////////////
std::string LayerActivation::constructUsage() {
  return "applies an activation function\nsActivation\n";
}
////////////////////////////////////////////////////////////////
bool LayerActivation::init(size_t &in, size_t &out, std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////
bool LayerActivation::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
vector<MatrixFloat *> LayerActivation::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
vector<MatrixFloat *> LayerActivation::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
} // namespace beednn
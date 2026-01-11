/*
        Copyright (c) 2019, Etienne de Foras and the respective contributors
        All rights reserved.

        Use of this source code is governed by a MIT-style license that can be
   found in the LICENSE.txt file.
*/

// example of Gated Activation:
//  GLU as in :https://arxiv.org/abs/1612.08083

#include "LayerGatedActivation.h"
#include "Activations.h"

using namespace std;
namespace beednn {
///////////////////////////////////////////////////////////////////////////////
LayerGatedActivation::LayerGatedActivation(const string &sActivation1,
                                           const string &sActivation2)
    : Layer("GatedActivation") {
  _pActivation1 = get_activation(sActivation1);
  _pActivation2 = get_activation(sActivation2);
}
///////////////////////////////////////////////////////////////////////////////
LayerGatedActivation::~LayerGatedActivation() {
  delete _pActivation1;
  delete _pActivation2;
}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerGatedActivation::clone() const {
  return new LayerGatedActivation(_pActivation1->name(), _pActivation2->name());
}
///////////////////////////////////////////////////////////////////////////////
bool LayerGatedActivation::init(size_t &in, size_t &out, std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  if (in & 1)
    return false;
  out = in / 2;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGatedActivation::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  assert(((mIn.cols() & 1) == 0) && "mIn must have an even size");

  Index iNbCols = mIn.cols();
  Index iNbColsHalf = iNbCols / 2;

  mOut.resize(mIn.rows(), iNbColsHalf);
  for (int r = 0; r < mIn.rows(); r++)
    for (int c = 0; c < iNbColsHalf; c++)
      mOut(r, c) = _pActivation1->apply(mIn(r, c)) *
                   _pActivation2->apply(mIn(r, c + iNbColsHalf));
}
///////////////////////////////////////////////////////////////////////////////
void LayerGatedActivation::backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn, std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  if (_bFirstLayer)
    return;

  MatrixFloat mLocalGrad;
  mLocalGrad.resizeLike(mIn);
  Index iNbCols = mIn.cols();
  Index iNbColsHalf = iNbCols / 2;

  // 1st part with activation and 2nd part without activation
  for (int r = 0; r < mIn.rows(); r++)
    for (int c = 0; c < iNbColsHalf; c++) {
      float g = mGradientOut(r, c);
      mLocalGrad(r, c) =
          g * _pActivation2->apply(mIn(r, c + iNbColsHalf)) *
          _pActivation1->derivation(mIn(r, c)); // (dL/dt)*g(y)*f'(x1)*g(x2)
      mLocalGrad(r, c + iNbColsHalf) =
          g * _pActivation1->apply(mIn(r, c)) *
          _pActivation2->derivation(
              mIn(r, c + iNbColsHalf)); // (dL/dt)*f(x1)*g'(x2)
    }

  if (mGradientIn.size() == 0) {
    mGradientIn = mLocalGrad;
  } else {
    mGradientIn += mLocalGrad;
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerGatedActivation::save(std::ostream &to) const {
  to << "LayerGatedActivation" << std::endl;
  to << _pActivation1->name() << ";" << _pActivation2->name() << std::endl;
}
///////////////////////////////////////////////////////////////
Layer *LayerGatedActivation::load(std::istream &from) {
  std::string sArg;
  from >> sArg;
  size_t pos = sArg.find(';');
  if (pos == std::string::npos)
    return nullptr;

  std::string activation1 = sArg.substr(0, pos);
  std::string activation2 = sArg.substr(pos + 1);

  return new LayerGatedActivation(activation1, activation2);
}
///////////////////////////////////////////////////////////////
Layer *LayerGatedActivation::construct(std::initializer_list<float> fArgs,
                                       std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;

  size_t pos = sArg.find(';');
  if (pos == std::string::npos)
    return nullptr;

  std::string activation1 = sArg.substr(0, pos);
  std::string activation2 = sArg.substr(pos + 1);

  return new LayerGatedActivation(activation1, activation2);
}
///////////////////////////////////////////////////////////////
std::string LayerGatedActivation::constructUsage() {
  return "applies gated activation functions\nsActivation1;sActivation2\n ";
}
///////////////////////////////////////////////////////////////
bool LayerGatedActivation::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerGatedActivation::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerGatedActivation::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
} // namespace beednn

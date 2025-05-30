/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSimplestRNN.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerSimplestRNN::LayerSimplestRNN(int iFrameSize) :
    LayerRNN(iFrameSize, iFrameSize)
{
}
///////////////////////////////////////////////////////////////////////////////
LayerSimplestRNN::~LayerSimplestRNN()
{ }
///////////////////////////////////////////////////////////////////////////////
bool LayerSimplestRNN::init(size_t& in, size_t& out, bool debug)
{
    _weight.setRandom(_iUnits, _iUnits); // Todo Xavier init ?

    out = in;
    Layer::init(in, out, debug);
    return true;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerSimplestRNN::clone() const
{
    LayerSimplestRNN* pLayer=new LayerSimplestRNN(_iFrameSize);
	pLayer->_weight = _weight;
    pLayer->_h = _h;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimplestRNN::forward_frame(const MatrixFloat& mInFrame, MatrixFloat& mOut)
{
    if (_h.rows() != mInFrame.rows())  // adapt to batch size
        _h.setZero(mInFrame.rows(), _iUnits);

    MatrixFloat u = _h * _weight + mInFrame;
    _h = u;// tanh(u);
	mOut=_h;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimplestRNN::backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mH, const MatrixFloat& mHm1, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    MatrixFloat mGradU = mH;// oneMinusSquare(mH); // derivative of tanh

    //grad(L/_Whh)=grad(L/U))*h(t-1)
    _gradientWeight = mGradU.transpose() *mHm1;
    _gradientWeight *= (1.f / _gradientWeight.rows());

    if (!_bFirstLayer)
    {
        //grad(L/h(t-1))=grad(L/U))*whh
        mGradientIn = mGradU.transpose()*_weight;
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimplestRNN::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerSimplestRNN::load(std::istream& from) {
    return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerSimplestRNN::construct(std::initializer_list<float> fArgs, std::string sArg) {
    if (fArgs.size() != 1) return nullptr; // iFrameSize
    return new LayerSimplestRNN(*fArgs.begin());
}
///////////////////////////////////////////////////////////////
std::string LayerSimplestRNN::constructUsage() {
    return "minimal recurrent neural network\n \niFrameSize";
}
///////////////////////////////////////////////////////////////
bool LayerSimplestRNN::has_weights() const
{
    return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerSimplestRNN::weights() const
{
    return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerSimplestRNN::gradient_weights() const
{
    return std::vector<MatrixFloat*>();
}
/////////////////////////////////////////////////////////////////////////////////////////////
}
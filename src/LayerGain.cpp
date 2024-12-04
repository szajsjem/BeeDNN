/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGain.h"
namespace beednn {


///////////////////////////////////////////////////////////////////////////////
LayerGain::LayerGain() :
    Layer("Gain")
{
}
///////////////////////////////////////////////////////////////////////////////
LayerGain::~LayerGain()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGain::clone() const
{
    LayerGain* pLayer=new LayerGain();
    pLayer->_weight=_weight;
    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerGain::init(size_t& in, size_t& out, bool debug)
{
	_weight.resize(0,0);

	out = in;
	Layer::init(in, out, debug);
	return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGain::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if(_weight.size()==0)
		_weight.setOnes(1,mIn.cols());
	
    mOut = mIn;

	for (int i = 0; i < mOut.rows(); i++)
		for (int j = 0; j < mOut.cols(); j++)
		{
			mOut(i,j) *= _weight(j);
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerGain::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	_gradientWeight = colWiseMean((mIn.transpose())*mGradientOut);

	if (_bFirstLayer)
		return;

	mGradientIn = mGradientOut;
	for (int i = 0; i < mGradientIn.rows(); i++)
		for (int j = 0; j < mGradientIn.cols(); j++)
		{
			mGradientIn(i, j) *= _weight(j);
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerGain::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerGain::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerGain::construct(std::initializer_list<float> fArgs, std::string sArg) {
	return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerGain::constructUsage() {
	return "error";
}
///////////////////////////////////////////////////////////////
bool LayerGain::has_weights() const
{
	return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerGain::weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerGain::gradient_weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
}
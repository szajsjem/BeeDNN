/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalBias.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerGlobalBias::LayerGlobalBias() :
    Layer("GlobalBias")
{
    _bias.resize(1,1);
	_gradientBias.resize(1, 1);
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalBias::~LayerGlobalBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGlobalBias::clone() const
{
    LayerGlobalBias* pLayer=new LayerGlobalBias();
	pLayer->_bias = _bias;
    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerGlobalBias::init(size_t& in, size_t& out, bool debug)
{
	_bias.setZero();
	out = in;
	Layer::init(in, out, debug);
	return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    mOut = mIn.array() + _bias(0);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;

	_gradientBias(0) = mGradientOut.mean();
	
	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerGlobalBias::load(std::istream& from) {
    return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerGlobalBias::construct(std::initializer_list<float> fArgs, std::string sArg) {
    return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerGlobalBias::constructUsage() {
	return "adds global bias term\n \n ";
}
///////////////////////////////////////////////////////////////
bool LayerGlobalBias::has_weights() const
{
	return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerGlobalBias::weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerGlobalBias::gradient_weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////////////////////
}
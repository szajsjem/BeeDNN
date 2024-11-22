/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerBias.h"
#include "Initializers.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerBias::LayerBias(const string& sBiasInitializer) :
    Layer("Bias")
{
    set_initializer(sBiasInitializer);
}
///////////////////////////////////////////////////////////////////////////////
LayerBias::~LayerBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerBias::clone() const
{
    LayerBias* pLayer=new LayerBias(get_initializer());
	pLayer->_bias = _bias;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerBias::init(size_t& in, size_t& out, bool debug)
{
	_bias.resize(0,0);
    out = in;
    Layer::init(in, out, debug);
    return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerBias::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if(_bias.size()==0)
        Initializers::compute(get_initializer(), _bias, 1, mIn.cols());

    mOut = rowWiseAdd( mIn , _bias);
}
///////////////////////////////////////////////////////////////////////////////
void LayerBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;
	
	_gradientBias = colWiseMean(mGradientOut);

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////
void LayerBias::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerBias::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerBias::construct(std::initializer_list<float> fArgs, std::string sArg) {
	return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerBias::constructUsage() {
	return "error";
}
///////////////////////////////////////////////////////////////
bool LayerBias::has_weights() const
{
	return true;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerBias::weights()
{
	std::vector<MatrixFloat*> v;
	v.push_back(&_bias);
	return v;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerBias::gradient_weights()
{
	std::vector<MatrixFloat*> v;
	v.push_back(&_gradientBias);
	return v;
}
///////////////////////////////////////////////////////////////
}
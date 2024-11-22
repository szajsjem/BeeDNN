/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerRRelu.h"

// as in : https://arxiv.org/pdf/1505.00853.pdf
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerRRelu::LayerRRelu(float alpha1, float alpha2) :
    Layer("RRelu")
{
	_alpha1=alpha1;
	_alpha2=alpha2;
	_invAlpha1=1.f/alpha1;
	_invAlpha2=1.f/alpha2;
	_invAlphaMean=(_invAlpha1+_invAlpha2)*0.5f;
	
}
///////////////////////////////////////////////////////////////////////////////
LayerRRelu::~LayerRRelu()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerRRelu::clone() const
{
    return new LayerRRelu(_alpha1,_alpha2);
}
///////////////////////////////////////////////////////////////////////////////
bool LayerRRelu::init(size_t& in, size_t& out, bool debug)
{
	out = in;
	Layer::init(in, out, debug);
	return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerRRelu::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut = mIn;
	if(_bTrainMode)
	{
		_slopes.resize(mIn.rows(), mIn.cols());
		setRandomUniform(_slopes, _invAlpha1, _invAlpha2);

		for (Index i = 0; i < mOut.size(); i++)
			if (mOut(i) < 0.f)
				mOut(i) *= _slopes(i);
	}
	else
	{
		for (Index i = 0; i < mOut.size(); i++)
			if (mOut(i) < 0.f)
				mOut(i) *= _invAlphaMean;
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerRRelu::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	if (_bFirstLayer)
		return;

	// compute gradientin
	mGradientIn = mGradientOut;
	for (Index i = 0; i < mGradientIn.size(); i++)
	{
		if (mIn(i) < 0.f)
			mGradientIn(i) *= _slopes(i);
	}
}
///////////////////////////////////////////////////////////////
void LayerRRelu::get_params(float& alpha1, float& alpha2) const
{
	alpha1 = _alpha1;
	alpha2 = _alpha2;
}
///////////////////////////////////////////////////////////////////////////////
void LayerRRelu::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerRRelu::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerRRelu::construct(std::initializer_list<float> fArgs, std::string sArg) {
	return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerRRelu::constructUsage() {
	return "error";
}
///////////////////////////////////////////////////////////////
bool LayerRRelu::has_weights() const
{
	return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerRRelu::weights()
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerRRelu::gradient_weights()
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
}
/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGaussianNoise.h"

#include <random>
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerGaussianNoise::LayerGaussianNoise(float fNoise):
    Layer("GaussianNoise"),
    _fNoise(fNoise),
	_distNormal(0.f, fNoise)
{ }
///////////////////////////////////////////////////////////////////////////////
LayerGaussianNoise::~LayerGaussianNoise()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGaussianNoise::clone() const
{
    return new LayerGaussianNoise(_fNoise);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianNoise::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut = mIn;
	if (_bTrainMode && (_fNoise > 0.f) )
	{
		for (Index i = 0; i < mOut.size(); i++)
			mOut(i) += _distNormal(randomEngine());
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianNoise::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;
 
	if (_bFirstLayer)
		return;

	mGradientIn= mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
float LayerGaussianNoise::get_noise() const
{
    return _fNoise;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianNoise::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerGaussianNoise::load(std::istream& from) {
    return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerGaussianNoise::construct(std::initializer_list<float> fArgs, std::string sArg) {
    return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerGaussianNoise::constructUsage() {
    return "error";
}
///////////////////////////////////////////////////////////////
bool LayerGaussianNoise::has_weights() const
{
	return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerGaussianNoise::weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerGaussianNoise::gradient_weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////////////////////
bool LayerGaussianNoise::init(size_t& in, size_t& out, bool debug)
{

	out = in;
	Layer::init(in, out, debug);
	return true;
}
///////////////////////////////////////////////////////////////////////////////
}
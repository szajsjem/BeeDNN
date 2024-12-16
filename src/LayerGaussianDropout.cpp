/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// GaussianDropout as in: https://keras.io/layers/noise/

#include "LayerGaussianDropout.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerGaussianDropout::LayerGaussianDropout(float fProba):
    Layer("GaussianDropout"),
    _fProba(fProba),
	_fStdev( sqrtf(_fProba / (1.f - _fProba))),
	_distNormal(1.f, _fStdev)
{ }
///////////////////////////////////////////////////////////////////////////////
LayerGaussianDropout::~LayerGaussianDropout()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGaussianDropout::clone() const
{
    return new LayerGaussianDropout(_fProba);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianDropout::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if (_bTrainMode)
	{
		_mask.resize(1, mIn.size());

		for (Index i = 0; i < _mask.size(); i++)
			_mask(0, i) = _distNormal(randomEngine());
		
		mOut = mIn * _mask.asDiagonal();
	}
	else
        mOut = mIn;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianDropout::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;
	assert(_bTrainMode);

	if (_bFirstLayer)
		return;

	mGradientIn= mGradientOut*_mask.asDiagonal();
}
///////////////////////////////////////////////////////////////////////////////
float LayerGaussianDropout::get_proba() const
{
    return _fProba;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianDropout::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerGaussianDropout::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerGaussianDropout::construct(std::initializer_list<float> fArgs, std::string sArg) {
	return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerGaussianDropout::constructUsage() {
	return "applies multiplicative gaussian noise\n \nfProbability";
}
///////////////////////////////////////////////////////////////
bool LayerGaussianDropout::init(size_t& in, size_t& out, bool debug) {
	return false;
}
///////////////////////////////////////////////////////////////
bool LayerGaussianDropout::has_weights() const {
	return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerGaussianDropout::weights() const {
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerGaussianDropout::gradient_weights() const {
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////////////////////
}
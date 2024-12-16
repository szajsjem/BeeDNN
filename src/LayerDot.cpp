/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerDot.h"

#include "Initializers.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerDot::LayerDot(Index iInputSize, Index iOutputSize,const string& sWeightInitializer) :
    Layer( "Dot"),
	_iInputSize(iInputSize),
	_iOutputSize(iOutputSize)
{
	set_initializer(sWeightInitializer);
}
///////////////////////////////////////////////////////////////////////////////
LayerDot::~LayerDot()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerDot::clone() const
{
    LayerDot* pLayer=new LayerDot(_iInputSize, _iOutputSize);
    pLayer->_weight=_weight;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerDot::init(size_t& in, size_t& out, bool debug)
{
	if (_iInputSize == 0)
		return false;

	if (_iOutputSize == 0)
		return false;
	if(in!=-1 && _iInputSize!=-1)
		assert(_iInputSize == in);
	else {
		assert(_iInputSize > 0 || in > 0);
	}
	assert(_iOutputSize > 0);//todo return false
	
	if (_iInputSize == -1)
		if (in == -1)
			return false;//please set the input data size to use this
		else {
			_iInputSize = in;
			Initializers::compute(get_initializer(), _weight, in, _iOutputSize);
		}
	else {
		Initializers::compute(get_initializer(), _weight, _iInputSize, _iOutputSize);
		in = _iInputSize;
	}
	out = _iOutputSize;
	Layer::init(in, out, debug);
	return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDot::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	assert(mIn.cols() == _weight.rows());
	mOut = mIn * _weight;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDot::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	// average the gradient as in: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
	_gradientWeight = (mIn.transpose())*mGradientOut*(1.f / mIn.rows());

	if (!_bFirstLayer)
		mGradientIn = mGradientOut * (_weight.transpose());
}
///////////////////////////////////////////////////////////////
Index LayerDot::input_size() const
{
	return _iInputSize;
}
///////////////////////////////////////////////////////////////
Index LayerDot::output_size() const
{
	return _iOutputSize;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDot::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerDot::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerDot::construct(std::initializer_list<float> fArgs, std::string sArg) {
	return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerDot::constructUsage() {
	return "matrix multiplication layer\nsWeightInitializer\niInputSize;iOutputSize";
}
///////////////////////////////////////////////////////////////
bool LayerDot::has_weights() const
{
	return true;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerDot::weights() const
{
	std::vector<MatrixFloat*> v;
	v.push_back((MatrixFloat*) & _weight);
	return v;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerDot::gradient_weights() const
{
	std::vector<MatrixFloat*> v;
	v.push_back((MatrixFloat*)&_gradientWeight);
	return v;
}
///////////////////////////////////////////////////////////////
}
/*
    Copyright (c) 2021, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerTimeDistributedDot.h"
#include "Initializers.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedDot::LayerTimeDistributedDot(int iInFrameSize, int iOutFrameSize, const string& sWeightInitializer) :
    Layer("TimeDistributedDot")
{
	_iInFrameSize=iInFrameSize;
	_iOutFrameSize=iOutFrameSize;

	set_initializer(sWeightInitializer);
}
///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedDot::~LayerTimeDistributedDot()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerTimeDistributedDot::clone() const
{
    LayerTimeDistributedDot* pLayer=new LayerTimeDistributedDot(_iInFrameSize,_iOutFrameSize, get_initializer());
	pLayer->_weight = _weight;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
int LayerTimeDistributedDot::in_frame_size() const
{
	return _iInFrameSize;
}
///////////////////////////////////////////////////////////////////////////////
int LayerTimeDistributedDot::out_frame_size() const
{
	return _iOutFrameSize;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerTimeDistributedDot::init(size_t& in, size_t& out, bool debug)
{
	//Xavier uniform initialization
	Initializers::compute(get_initializer(), _weight, _iInFrameSize, _iOutFrameSize);

	out = in;
	Layer::init(in, out, debug);
	return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	// reshape the input to (x, _iFrameSize), compute, reshape back
	Index iNbFrames = mIn.cols() / _iInFrameSize;
	MatrixFloat mInR = viewResize(mIn, iNbFrames* mIn.rows(), _iInFrameSize);
	mOut = mInR * _weight;
	mOut.resize(mIn.rows(), iNbFrames*_iOutFrameSize);
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	// average the gradient as in: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent

	// reshape the input and gradient to (x, _iFrameSize), compute product, reshape back
	Index iNbFrames = mGradientOut.cols() / _iOutFrameSize;
	MatrixFloat mGradientOutR = viewResize(mGradientOut, iNbFrames * mGradientOut.rows(), _iOutFrameSize);
	MatrixFloat mInR = viewResize(mIn, iNbFrames * mIn.rows(), _iInFrameSize);
	
	_gradientWeight = (mInR.transpose()) * mGradientOutR * (1.f / mIn.rows());

	if (!_bFirstLayer)
	{
		mGradientIn = mGradientOutR * (_weight.transpose());
		mGradientIn.resize(mIn.rows(), iNbFrames * _iInFrameSize);
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerTimeDistributedDot::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerTimeDistributedDot::construct(std::initializer_list<float> fArgs, std::string sArg) {
	return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerTimeDistributedDot::constructUsage() {
	return "error";
}
///////////////////////////////////////////////////////////////
bool LayerTimeDistributedDot::has_weights() const
{
	return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerTimeDistributedDot::weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerTimeDistributedDot::gradient_weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
}
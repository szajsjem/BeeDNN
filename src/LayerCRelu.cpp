/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerCRelu.h"
namespace beednn {


// CRelu as in : https://arxiv.org/pdf/1603.05201.pdf
// Warning: double the output size
// 
///////////////////////////////////////////////////////////////////////////////
LayerCRelu::LayerCRelu() :
    Layer("CRelu")
{	
}
///////////////////////////////////////////////////////////////////////////////
LayerCRelu::~LayerCRelu()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerCRelu::clone() const
{
    return new LayerCRelu();
}
///////////////////////////////////////////////////////////////////////////////
bool LayerCRelu::init(size_t& in, size_t& out, bool debug)
{
	out = in;
	Layer::init(in, out, debug);
	return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerCRelu::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	Index iInSize=mIn.cols();
	mOut.setZero(mIn.rows(),iInSize*2);
	for (Index r = 0; r < mIn.rows(); r++)
		for (Index c = 0; c < iInSize; c++)
		{
			float f=mIn(r,c);
			if(f>=0.)
				mOut(r,c)=f;
			else
				mOut(r,c+iInSize)=-f;
		}			
}
///////////////////////////////////////////////////////////////////////////////
void LayerCRelu::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	if (_bFirstLayer)
		return;

	Index iInSize = mIn.cols();
	mGradientIn.setZero(mIn.rows(), iInSize);
	for (Index r = 0; r < mIn.rows(); r++)
		for (Index c = 0; c < iInSize; c++)
		{
			if (mIn(r, c) > 0.)
				mGradientIn(r, c) = mGradientOut(r, c);
			else
				mGradientIn(r, c) = -mGradientOut(r, c + iInSize);
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerCRelu::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerCRelu::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerCRelu::construct(std::initializer_list<float> fArgs, std::string sArg) {
	if (fArgs.size() != 0) return nullptr;
	return new LayerCRelu();
}
///////////////////////////////////////////////////////////////
std::string LayerCRelu::constructUsage() {
	return "applies concatenated rectified linear unit\n \n ";
}
///////////////////////////////////////////////////////////////
bool LayerCRelu::has_weights() const
{
	return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerCRelu::weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerCRelu::gradient_weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
}
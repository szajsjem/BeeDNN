/*
    Copyright (c) 2020, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// Softmin as in https://pytorch.org/docs/stable/nn.html#torch.nn.Softmin
// use Softmax code with data negation

#include "LayerSoftmin.h"

#include <cmath>
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerSoftmin::LayerSoftmin():
    Layer("Softmin")
{ }
///////////////////////////////////////////////////////////////////////////////
LayerSoftmin::~LayerSoftmin()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerSoftmin::clone() const
{
    return new LayerSoftmin();
}
///////////////////////////////////////////////////////////////////////////////
void LayerSoftmin::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	MatrixFloat S;
	mOut=-mIn;

	for (Index r = 0; r < mOut.rows(); r++)// todo simplify and optimize
	{
		S = mOut.row(r); 
		S.array() -= S.maxCoeff(); //remove max
		S = S.array().exp();
		mOut.row(r) =S/ S.sum();
	}
}
///////////////////////////////////////////////////////////////////////////////
// from https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
void LayerSoftmin::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	if (_bFirstLayer)
		return;

    MatrixFloat S;
	mGradientIn.resizeLike(mGradientOut);

	for (Index r = 0; r < mIn.rows(); r++) // todo simplify and optimize
	{
		S = -mIn.row(r);
		S.array() -= S.maxCoeff(); //remove max
		S = S.array().exp();

		float s = S.sum();
		for (Index c = 0; c < S.cols(); c++)
		{
			float expx = S(c);
			mGradientIn(r, c) = -mGradientOut(r, c)*(expx*(s-expx)) / (s*s);
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerSoftmin::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerSoftmin::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerSoftmin::construct(std::initializer_list<float> fArgs, std::string sArg) {
	return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerSoftmin::constructUsage() {
	return "error";
}
///////////////////////////////////////////////////////////////
bool LayerSoftmin::has_weights() const
{
	return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerSoftmin::weights()
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerSoftmin::gradient_weights()
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////////////////////
bool LayerSoftmin::init(size_t& in, size_t& out, bool debug)
{

	out = in;
	Layer::init(in, out, debug);
	return true;
}
///////////////////////////////////////////////////////////////////////////////
}
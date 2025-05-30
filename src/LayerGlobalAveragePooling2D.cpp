/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalAveragePooling2D.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerGlobalAveragePooling2D::LayerGlobalAveragePooling2D(Index iInRows, Index iInCols, Index iInChannels) :
    Layer("GlobalAveragePooling2D")
{
	_iInRows = iInRows;
	_iInCols = iInCols;
	_iInChannels = iInChannels;
	
	_iInPlaneSize = _iInRows * _iInCols;
	_fInvKernelSize = 1.f / (float)(iInRows * iInCols);
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalAveragePooling2D::~LayerGlobalAveragePooling2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalAveragePooling2D::get_params(Index& iInRows, Index& iInCols, Index & iInChannels) const
{
	iInRows = _iInRows;
	iInCols = _iInCols;
	iInChannels = _iInChannels;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGlobalAveragePooling2D::clone() const
{
    return new LayerGlobalAveragePooling2D(_iInRows, _iInCols, _iInChannels);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalAveragePooling2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut.resize(mIn.rows(), _iInChannels);

	//not optimized yet
	for (Index sample = 0; sample < mIn.rows(); sample++)
	{
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			const float* lIn = mIn.row(sample).data()+ channel * _iInPlaneSize;
			float* lOut = mOut.row(sample).data()+channel;
			float fSum= 0.f;
			
			for (Index ri = 0; ri < _iInRows ; ri++)
			{
				for (Index ci = 0; ci < _iInCols ; ci++)
				{
					Index iIndex = ri * _iInCols + ci; //flat index in plane
					fSum += lIn[iIndex];
				}
			}

			lOut[0] = fSum*_fInvKernelSize;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalAveragePooling2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	if (_bFirstLayer)
		return;

	mGradientIn.resize(mGradientOut.rows(), _iInPlaneSize * _iInChannels);

	//not optimized yet
	for (Index sample = 0; sample < mGradientOut.rows(); sample++)
	{
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			const float* lGradientOut= mGradientOut.row(sample).data() + channel;
			float* lGradientIn = mGradientIn.row(sample).data() + channel * _iInPlaneSize;

			float fGradOut= lGradientOut[0]* _fInvKernelSize;

			for (Index ri = 0 ; ri < _iInRows; ri++)
			{
				for (Index ci = 0; ci < _iInCols; ci++)
				{
					Index iIndexIn = ri * _iInCols + ci; //flat index in plane
					lGradientIn[iIndexIn]= fGradOut;
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalAveragePooling2D::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerGlobalAveragePooling2D::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerGlobalAveragePooling2D::construct(std::initializer_list<float> fArgs, std::string sArg) {
	if (fArgs.size() != 3) return nullptr; // iInRows, iInCols, iInChannels
	auto args = fArgs.begin();
	return new LayerGlobalAveragePooling2D(*args, *(args + 1), *(args + 2));
}
///////////////////////////////////////////////////////////////
std::string LayerGlobalAveragePooling2D::constructUsage() {
	return "global average pooling for 2D data\n \niInRows;iInCols;iInChannels";
}
///////////////////////////////////////////////////////////////
bool LayerGlobalAveragePooling2D::init(size_t& in, size_t& out, bool debug ) {
	return false;
}
///////////////////////////////////////////////////////////////
bool LayerGlobalAveragePooling2D::has_weights() const {
	return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerGlobalAveragePooling2D::weights() const {
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerGlobalAveragePooling2D::gradient_weights() const {
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////////////////////
}
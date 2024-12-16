/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerAveragePooling2D.h"
namespace beednn {


///////////////////////////////////////////////////////////////////////////////
LayerAveragePooling2D::LayerAveragePooling2D(Index iInRows, Index iInCols, Index iInChannels, Index iRowFactor, Index iColFactor) :
    Layer("AveragePooling2D")
{
	_iInRows = iInRows;
	_iInCols = iInCols;
	_iInChannels = iInChannels;
	_iRowFactor = iRowFactor;
	_iColFactor = iColFactor;
	_iOutRows = iInRows/iRowFactor;
	_iOutCols = iInCols/iColFactor;
	_iInPlaneSize = _iInRows * _iInCols;
	_iOutPlaneSize = _iOutRows * _iOutCols;
	_fInvKernelSize = 1.f / (float)(_iRowFactor * _iColFactor);
}
///////////////////////////////////////////////////////////////////////////////
LayerAveragePooling2D::~LayerAveragePooling2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerAveragePooling2D::get_params(Index& iInRows, Index& iInCols, Index & iInChannels, Index& iRowFactor, Index& iColFactor) const
{
	iInRows = _iInRows;
	iInCols = _iInCols;
	iInChannels = _iInChannels;
	iRowFactor = _iRowFactor;
	iColFactor= _iColFactor;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerAveragePooling2D::clone() const
{
    return new LayerAveragePooling2D(_iInRows, _iInCols, _iInChannels, _iRowFactor, _iColFactor);
}
///////////////////////////////////////////////////////////////////////////////
void LayerAveragePooling2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut.resize(mIn.rows(), _iOutPlaneSize*_iInChannels);

	//not optimized yet
	for (Index sample = 0; sample < mIn.rows(); sample++)
	{
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			const float* lIn = mIn.row(sample).data()+ channel * _iInPlaneSize;
			float* lOut = mOut.row(sample).data()+channel* _iOutPlaneSize;
			for (Index r = 0; r < _iOutRows; r++)
			{
				for (Index c = 0; c < _iOutCols; c++)
				{
					float fSum= 0.f;
					
					for (Index ri = r * _iRowFactor; ri < r*_iRowFactor + _iRowFactor; ri++)
					{
						for (Index ci = c * _iColFactor; ci < c*_iColFactor + _iColFactor; ci++)
						{
							Index iIndex = ri * _iInCols + ci; //flat index in plane
							fSum += lIn[iIndex];
						}
					}

					lOut[r * _iOutCols + c] = fSum*_fInvKernelSize;
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerAveragePooling2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	if (_bFirstLayer)
		return;

	mGradientIn.setZero(mGradientOut.rows(), _iInPlaneSize * _iInChannels);

	//not optimized yet
	for (Index sample = 0; sample < mGradientOut.rows(); sample++)
	{
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			const float* lGradientOut= mGradientOut.row(sample).data() + channel * _iOutPlaneSize;
			float* lGradientIn = mGradientIn.row(sample).data() + channel * _iInPlaneSize;

			for (Index r = 0; r < _iOutRows; r++)
			{
				for (Index c = 0; c < _iOutCols; c++)
				{
					float fGradOut= lGradientOut[c+r*_iOutCols]* _fInvKernelSize;

					for (Index ri = r * _iRowFactor; ri < r * _iRowFactor + _iRowFactor; ri++)
					{
						for (Index ci = c * _iColFactor; ci < c * _iColFactor + _iColFactor; ci++)
						{
							Index iIndexIn = ri * _iInCols + ci; //flat index in plane
							lGradientIn[iIndexIn]+= fGradOut;
						}
					}
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerAveragePooling2D::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerAveragePooling2D::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerAveragePooling2D::construct(std::initializer_list<float> fArgs, std::string sArg) {
	return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerAveragePooling2D::constructUsage() {
	return "downsamples input using average pooling\n \niInRows;iInCols;iInChannels;iRowFactor;iColFactor";
}
///////////////////////////////////////////////////////////////
bool LayerAveragePooling2D::init(size_t& in, size_t& out, bool debug) {
	return false;
}
///////////////////////////////////////////////////////////////
bool LayerAveragePooling2D::has_weights() const {
	return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerAveragePooling2D::weights() const {
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerAveragePooling2D::gradient_weights() const {
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
}
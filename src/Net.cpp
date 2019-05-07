/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Net.h"
#include "Layer.h"

#include "Matrix.h"

#include "LayerActivation.h"
#include "LayerSoftmax.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerGlobalGain.h"
#include "LayerPoolAveraging1D.h"

#include <cmath>
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////
Net::Net()
{ 
    _bTrainMode = false;
	_iOutputSize = 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Net::~Net()
{
    clear();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::clear()
{
    for(unsigned int i=0;i<_layers.size();i++)
        delete _layers[i];

    _layers.clear();
    _bTrainMode=false;
	_iOutputSize = 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Net& Net::operator=(const Net& other)
{
    clear();

    for(unsigned int i=0;i<other._layers.size();i++)
        _layers.push_back(other._layers[i]->clone());

	_iOutputSize = other._iOutputSize;

	return *this;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_dropout_layer(int iSize,float fRatio)
{
    _layers.push_back(new LayerDropout(iSize, fRatio));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_activation_layer(string sType)
{
    _layers.push_back(new LayerActivation(sType));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_softmax_layer()
{
    _layers.push_back(new LayerSoftmax());
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_dense_layer(int inSize,int outSize,bool bHasBias)
{
    _layers.push_back(new LayerDense(inSize,outSize, bHasBias));
	_iOutputSize = outSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_globalgain_layer(int inSize, float fGlobalGain)
{
    _layers.push_back(new LayerGlobalGain(inSize,fGlobalGain));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_poolaveraging1D_layer(int inSize, int iOutSize)
{
    _layers.push_back(new LayerPoolAveraging1D(inSize, iOutSize));
	_iOutputSize = iOutSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
    MatrixFloat mTemp=mIn;
    for(unsigned int i=0;i<_layers.size();i++)
    {
        _layers[i]->forward(mTemp,mOut);
        mTemp=mOut; //todo avoid resize
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
int Net::classify(const MatrixFloat& mIn) const
{
    MatrixFloat mOut; //TODO needed?
    forward(mIn,mOut);
    return argmax(mOut);
}
/////////////////////////////////////////////////////////////////////////////////////////////
void Net::classify_all(const MatrixFloat& mIn, MatrixFloat& mClass) const
{
    MatrixFloat mOut;
	forward(mIn, mOut);
	
	if (mOut.cols() != 1)
		rowsArgmax(mOut, mClass);
	else
	{
		mClass.resize(mIn.rows(), 1);
		for (int i = 0; i < mIn.rows(); i++)
			mClass(i, 0) = std::roundf(mOut(0, 0)); //case of "output is a label"
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::set_train_mode(bool bTrainMode)
{
    _bTrainMode = bTrainMode;

    for (unsigned int i = 0; i < _layers.size(); i++)
        _layers[i]->set_train_mode(bTrainMode);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
const vector<Layer*> Net::layers() const
{
    return _layers;
}
/////////////////////////////////////////////////////////////////////////////////////////////
Layer& Net::layer(size_t iLayer)
{
    return *(_layers[iLayer]);
}
/////////////////////////////////////////////////////////////////////////////////////////////
void Net::init()
{
    for(unsigned int i=0;i<_layers.size();i++)
    {
        _layers[i]->init();
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
int Net::output_size() const
{
	return _iOutputSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////

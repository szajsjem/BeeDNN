/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Layer.h"

////////////////////////////////////////////////////////////////
Layer::Layer(int iInSize, int iOutSize,const string& sType):
_iInSize(iInSize),
_iOutSize(iOutSize),
_sType(sType)
{ 
	_bTrainMode = false;
}
////////////////////////////////////////////////////////////////
Layer::~Layer()
{ }
////////////////////////////////////////////////////////////////
void Layer::init()
{ }
///////////////////////////////////////////////////////////////
string Layer::type() const
{
    return _sType;
}
///////////////////////////////////////////////////////////////
int Layer::in_size() const
{
    return _iInSize;
}
///////////////////////////////////////////////////////////////
int Layer::out_size() const
{
    return _iOutSize;
}
///////////////////////////////////////////////////////////////
void Layer::set_train_mode(bool bTrainMode)
{
	_bTrainMode = bTrainMode;
}
///////////////////////////////////////////////////////////////
bool Layer::has_weight()
{
    return false;
}
///////////////////////////////////////////////////////////////
MatrixFloat& Layer::weights()
{
    return _weight;
}
///////////////////////////////////////////////////////////////
MatrixFloat& Layer::gradient_weights()
{
    return _deltaWeight;
}
///////////////////////////////////////////////////////////////

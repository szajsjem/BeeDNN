/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerChannelBias.h"
#include "Initializers.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerChannelBias::LayerChannelBias(Index iNbRows,Index iNbCols,Index iNbChannels, const string& sBiasInitializer) :
    Layer("ChannelBias")
{
	_iNbRows=iNbRows;
	_iNbCols=iNbCols;
	_iNbChannels=iNbChannels;

	set_initializer(sBiasInitializer);
}
///////////////////////////////////////////////////////////////////////////////
LayerChannelBias::~LayerChannelBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerChannelBias::clone() const
{
    LayerChannelBias* pLayer=new LayerChannelBias(_iNbRows,_iNbCols,_iNbChannels, get_initializer());
	pLayer->_bias = _bias;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerChannelBias::init(size_t& in, size_t& out, bool debug)
{
	Initializers::compute(get_initializer(), _bias, 1, _iNbChannels);
	out = in;
	Layer::init(in, out, debug);
	return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::get_params(Index & iRows, Index & iCols, Index & iChannels) const
{
	iRows = _iNbRows;
	iCols = _iNbCols;
	iChannels = _iNbChannels;
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut = mIn;
	channelWiseAdd(mOut, mIn.rows(),_iNbChannels,_iNbRows,_iNbCols, _bias);
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;

	_gradientBias =channelWiseMean(mGradientOut, mGradientOut.rows(),_iNbChannels,_iNbRows,_iNbCols);

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerChannelBias::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerChannelBias::construct(std::initializer_list<float> fArgs, std::string sArg) {
	if (fArgs.size() != 3) return nullptr; // iNbRows, iNbCols, iNbChannels
	auto args = fArgs.begin();
	return new LayerChannelBias(*args, *(args + 1), *(args + 2), sArg);
}
///////////////////////////////////////////////////////////////
std::string LayerChannelBias::constructUsage() {
	return "adds trainable bias per channel\nsBiasInitializer\niNbRows;iNbCols;iNbChannels";
}
///////////////////////////////////////////////////////////////
bool LayerChannelBias::has_weights() const
{
	return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerChannelBias::weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat*> LayerChannelBias::gradient_weights() const
{
	return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////////////////////
}
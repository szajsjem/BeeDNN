/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerActivation.h"

#include "Activations.h"
namespace beednn {

using namespace std;
///////////////////////////////////////////////////////////////////////////////
LayerActivation::LayerActivation(const string& sActivation):
    Layer(sActivation)
{
    _pActivation=get_activation(sActivation);

	assert(_pActivation);
}
///////////////////////////////////////////////////////////////////////////////
LayerActivation::~LayerActivation()
{
	delete _pActivation;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerActivation::clone() const
{
    return new LayerActivation(_pActivation->name());
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    assert(_pActivation);
    mOut.resizeLike(mIn);

    for(Index i=0;i<mOut.size();i++)
        mOut(i)=_pActivation->apply(mIn(i));
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    assert(mIn.rows() == mGradientOut.rows());
    assert(mIn.cols() == mGradientOut.cols());
    assert(_pActivation);

	if (_bFirstLayer)
		return;

    mGradientIn.resizeLike(mGradientOut);
    for(Index i=0;i<mGradientIn.size();i++)
        mGradientIn(i)=_pActivation->derivation(mIn(i));

    mGradientIn = mGradientIn.cwiseProduct(mGradientOut);
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
Layer* LayerActivation::load(std::istream& from) {
    return NULL;
}
///////////////////////////////////////////////////////////////
Layer* LayerActivation::construct(std::initializer_list<float> fArgs, std::string sArg) {
    return NULL;
}
///////////////////////////////////////////////////////////////
std::string LayerActivation::constructUsage() {
    return "error";
}
////////////////////////////////////////////////////////////////
bool LayerActivation::init(size_t& in, size_t& out, bool debug)
{
    out = in;
    Layer::init(in, out, debug);
    return true;
}
///////////////////////////////////////////////////////////////
bool LayerActivation::has_weights() const
{
    return false;
}
///////////////////////////////////////////////////////////////
vector<MatrixFloat*> LayerActivation::weights() const
{
    return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
vector<MatrixFloat*> LayerActivation::gradient_weights() const
{
    return std::vector<MatrixFloat*>();
}
///////////////////////////////////////////////////////////////
}
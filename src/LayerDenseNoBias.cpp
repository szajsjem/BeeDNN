#include "LayerDenseNoBias.h"

#include <cstdlib> // for rand
#include <cmath> // for sqrt

///////////////////////////////////////////////////////////////////////////////
LayerDenseNoBias::LayerDenseNoBias(int iInSize,int iOutSize):
    Layer(iInSize,iOutSize,"DenseNoBias")
{
    _weight.resize((int)_iInSize,(int)_iOutSize);
	
	LayerDenseNoBias::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDenseNoBias::~LayerDenseNoBias()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerDenseNoBias::init()
{
    //Xavier uniform initialisation
    float a =sqrtf(6.f/(_iInSize+_iOutSize));
    for(int i=0;i<_weight.size();i++)
        _weight(i)=((float)rand()/(float)RAND_MAX-0.5f)*2.f*a;
	
	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseNoBias::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    mMatOut= mMatIn*_weight;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseNoBias::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta)
{
    mNewDelta=mDelta*(_weight.transpose());
    _weight-=(mInput.transpose())*(mDelta*fLearningRate);
}
///////////////////////////////////////////////////////////////////////////////
const MatrixFloat& LayerDenseNoBias::weight() const
{
    return _weight;
}
///////////////////////////////////////////////////////////////////////////////

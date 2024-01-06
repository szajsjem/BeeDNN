#include "LayerEmbed.h"

#include "Initializers.h"
using namespace std;
namespace beednn {
	///////////////////////////////////////////////////////////////////////////////
	LayerEmbed::LayerEmbed(Index vocabSize, Index dimensionSize, Index maxPositon) :
		Layer("Normalize"),
		_pPositionSize(maxPositon),
		_pVocabSize(vocabSize),
		_pDimensionSize(dimensionSize)
	{
		LayerEmbed::init();
	}
	///////////////////////////////////////////////////////////////////////////////
	LayerEmbed::~LayerEmbed()
	{
	}
	///////////////////////////////////////////////////////////////////////////////
	Layer* LayerEmbed::clone() const
	{
		LayerEmbed* pLayer = new LayerEmbed(_pVocabSize, _pDimensionSize, _pPositionSize);
		//copy the token embed and position embed
		return pLayer;
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerEmbed::init()
	{

		//init two 2d biases
		// dim*vocab
		// dim*pos

		//idk how to make that

		Layer::init();
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerEmbed::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		//assert(mIn.cols()==1);
		//assert(mIn.maxval()<vocabsize && >0
		//assert(mIn.rows()<possize && >0    
		//or we just scale position and vocab size up

		for (int i = 0; i < mIn.rows(); i++) {
			MatrixFloat encodedToken;
			//encodedToken += vocabBias.col(mIn(i, 0));
			//encodedToken += posBias.col(i);
			//mOut.addCol(encodedToken); //somthing like that \/ below
			mOut.col(mOut.cols()) = encodedToken.col(0);
		}
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerEmbed::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		for (int i = 0; i < mIn.rows(); i++) {
			//vocabBiasGradient.col(mIn(i, 0))+= mGradientOut.row(i);
			//posBiasGradient.col(i) += mGradientOut.row(i);
		}
		//it should be the first layer because tokenizer is static
		//assert(_bFirstLayer);
	}
	///////////////////////////////////////////////////////////////
}

#include "LayerEmbed.h"

#include "Initializers.h"
namespace beednn {
	///////////////////////////////////////////////////////////////////////////////
	LayerEmbed::LayerEmbed(const Index vocabSize,const Index dimensionSize,const Index maxPositon, const std::string& sBiasInitializer) :
		Layer("Normalize"),
		_pPositionSize(maxPositon),
		_pVocabSize(vocabSize),
		_pDimensionSize(dimensionSize)
	{
		set_bias_initializer(sBiasInitializer);
		LayerEmbed::init();
	}
	///////////////////////////////////////////////////////////////////////////////
	LayerEmbed::~LayerEmbed()
	{
	}
	///////////////////////////////////////////////////////////////////////////////
	Layer* LayerEmbed::clone() const
	{
		LayerEmbed* pLayer = new LayerEmbed(_pVocabSize, _pDimensionSize, _pPositionSize,bias_initializer());
		pLayer->_bias = _bias;
		pLayer->_bias2 = _bias2;
		return pLayer;
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerEmbed::init()
	{
		_bias.resize(_pDimensionSize, _pVocabSize);
		Initializers::compute(bias_initializer(), _bias, _pDimensionSize, _pVocabSize);
		_bias2.resize(_pDimensionSize, _pPositionSize);
		Initializers::compute(bias_initializer(), _bias, _pDimensionSize, _pPositionSize);
		Layer::init();
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerEmbed::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		//assert(mIn.cols()==1);
		//assert(mIn.maxval()<vocabsize && >0
		//assert(mIn.rows()<possize && >0    
		//or we just scale position and vocab size up

		for (int i = 0; i < mIn.rows(); i++) {//or mIn.cols()
			MatrixFloat encodedToken;
			encodedToken += _bias.col(mIn(i, 0));
			encodedToken += _bias2.col(i);
			//mOut.addCol(encodedToken); //somthing like that \/ below
			mOut.col(mOut.cols()) = encodedToken.col(0);
		}
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerEmbed::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		for (int i = 0; i < mIn.rows(); i++) {
			_gradientBias.col(mIn(i, 0))+= mGradientOut.row(i);
			_gradientBias2.col(i) += mGradientOut.row(i);
		}
		//it should be the first layer because tokenizer is static
		//assert(_bFirstLayer);
	}

	///////////////////////////////////////////////////////////////
	bool LayerEmbed::has_weights() const
	{
		return false;
	}
	
	///////////////////////////////////////////////////////////////
	bool LayerEmbed::has_biases() const
	{
		return true;
	}
	///////////////////////////////////////////////////////////////
	std::vector<MatrixFloat*> LayerEmbed::biases()
	{
		std::vector<MatrixFloat*> v;
		v.push_back(&_bias);
		v.push_back(&_bias2);
		return v;
	}
	///////////////////////////////////////////////////////////////
	std::vector<MatrixFloat*> LayerEmbed::gradient_biases()
	{
		std::vector<MatrixFloat*> v;
		v.push_back(&_gradientBias);
		v.push_back(&_gradientBias2);
		return v;
	}
	///////////////////////////////////////////////////////////////
}

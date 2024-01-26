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
		_bias.resize(_pVocabSize, _pDimensionSize);
		Initializers::compute(bias_initializer(), _bias, _pVocabSize, _pDimensionSize);
		_bias2.resize(_pPositionSize, _pDimensionSize);
		Initializers::compute(bias_initializer(), _bias2, _pPositionSize, _pDimensionSize);
		Layer::init();
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerEmbed::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		//assert(mIn.cols()==1);
		//assert(mIn.maxval()<vocabsize && >0
		//assert(mIn.rows()<possize && >0    
		//or we just scale position and vocab size up

		mOut.resize(mIn.rows(), _pDimensionSize); 

		for (int i = 0; i < mIn.rows(); i++) {
			mOut.row(i) = _bias.row(mIn(i, 0));
			mOut.row(i) += _bias2.row(i % _pPositionSize);//idk how do you extend position encoding
		}
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerEmbed::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		_gradientBias.resizeLike(_bias);
		_gradientBias2.resizeLike(_bias2);
		for (int i = 0; i < mIn.rows(); i++) {
			_gradientBias.row(mIn(i, 0))+= mGradientOut.row(i);
			_gradientBias2.row(i % _pPositionSize) += mGradientOut.row(i);
		}
		//it should be the first layer because tokenizer can process text into diffrent number of tokens,
		// i don't think you can implement it as layer
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

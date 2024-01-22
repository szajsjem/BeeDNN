#pragma once

#include "Layer.h"
#include "Matrix.h"
namespace beednn {
	class LayerEmbed : public Layer
	{
	public:
		explicit LayerEmbed(const Index vocabSize, const Index dimensionSize, const Index maxPositon, const std::string& sBiasInitializer = "Zeros");
		virtual ~LayerEmbed() override;

		virtual Layer* clone() const override;

		virtual void init() override;

		virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
		virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

		virtual bool has_weights() const;
		virtual bool has_biases() const;
		virtual std::vector<MatrixFloat*> biases();
		virtual std::vector<MatrixFloat*> gradient_biases();
	private:
		Index _pVocabSize, _pPositionSize, _pDimensionSize;
		MatrixFloat _bias2, _gradientBias2;
	};
}
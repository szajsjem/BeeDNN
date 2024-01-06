#pragma once

#include "Layer.h"
#include "Matrix.h"
namespace beednn {
	class LayerEmbed : public Layer
	{
	public:
		explicit LayerEmbed(Index vocabSize, Index dimensionSize, Index maxPositon);
		virtual ~LayerEmbed() override;

		virtual Layer* clone() const override;

		virtual void init() override;

		virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
		virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

	private:
		Index _pVocabSize, _pPositionSize, _pDimensionSize;
	};
}
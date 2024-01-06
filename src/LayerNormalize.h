#pragma once

#include "Layer.h"
#include "Matrix.h"
namespace beednn {
	class LayerNormalize : public Layer
	{
	public:
		explicit LayerNormalize();
		virtual ~LayerNormalize() override;

		virtual Layer* clone() const override;

		virtual void init() override;

		virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
		virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

	private:
	};
}
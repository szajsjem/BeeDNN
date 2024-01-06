#pragma once
#include "Layer.h"
#include "Matrix.h"

namespace beednn {
	class Activation;

	class LayerSelfDot : public Layer
	{
	public:
		explicit LayerSelfDot();
		virtual ~LayerSelfDot() override;

		virtual Layer* clone() const override;

		virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
		virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

	private:
	};
}
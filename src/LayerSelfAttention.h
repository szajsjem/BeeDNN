#pragma once
#include "Layer.h"
#include "Matrix.h"
#include "LayerSelfDot.h"
namespace beednn {
	class Activation;

	class LayerSelfAttention : public LayerSelfDot
	{
	public:
		explicit LayerSelfAttention();

		virtual Layer* clone() const override;

		virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
		virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;
	};
}
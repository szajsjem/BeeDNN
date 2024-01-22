#pragma once
#include "Layer.h"
#include "Matrix.h"

namespace beednn {

	class LayerTranspose : public Layer
	{
	public:
		explicit LayerTranspose();
		virtual ~LayerTranspose() override;

		virtual Layer* clone() const override;

		virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
		virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

	private:
	};
}
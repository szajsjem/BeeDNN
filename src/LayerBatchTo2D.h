#pragma once

#include "Layer.h"
#include "Matrix.h"
namespace beednn {
	class LayerBatchTo2D : public Layer
	{
	public:
		explicit LayerBatchTo2D(const int incolsize, const int outcolsize, const int outrowsize, Layer* l2d);
		virtual ~LayerBatchTo2D() override;

		virtual Layer* clone() const override;

		virtual void init() override;

		virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
		virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

		virtual bool has_weights() const;
		virtual std::vector<MatrixFloat*> weights();
		virtual std::vector<MatrixFloat*> gradient_weights();
		virtual bool has_biases() const;
		virtual std::vector<MatrixFloat*> biases();
		virtual std::vector<MatrixFloat*> gradient_biases();
	private:
		Layer* _l2d;
		int _incolsize;
		int _outcolsize;
		int _outrowsize;
	};
}
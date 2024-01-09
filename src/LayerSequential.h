#pragma once

#include <vector>
#include "Matrix.h"
#include "Layer.h"

namespace beednn {
	class LayerSequential : public Layer
	{
	public:
		explicit LayerSequential(std::vector<Layer*> mSequentialLayers);
		virtual ~LayerSequential() override;

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
		LayerSequential();
		std::vector <Layer*> _Layers;
	};
}
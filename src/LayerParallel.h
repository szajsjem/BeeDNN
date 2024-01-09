#pragma once
#include <vector>
#include "Matrix.h"
#include "Layer.h"
#include "ParallelReduction.h"

namespace beednn {
	class LayerParallel : public Layer
	{
	public:
		explicit LayerParallel(std::vector<Layer*> mParallelLayers, ParallelReduction mReduction);//todo: how to connect many outputs
		virtual ~LayerParallel() override;

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
		LayerParallel();
		std::vector < Layer*> _Layers;
		ParallelReduction _ParallelReduction;
	};
}
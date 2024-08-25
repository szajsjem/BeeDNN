#pragma once
#include <vector>
#include "Matrix.h"
#include "Layer.h"
#include "ParallelReduction.h"

namespace beednn {
	class LayerRouter : public Layer
	{
	public:
		/*
		selectedexperts 0..1 val above which experts are selected
		>=2 selected highest n layers
		RouterLayer shoud have softmax at the end 
		*/
		explicit LayerRouter(Layer* RouterLayer, float selectedexperts, std::vector<Layer*> mExperts, ParallelReduction mReduction);

		virtual ~LayerRouter() override;

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
		LayerRouter();
		Layer* _router;
		std::vector < Layer*> _Layers;
		float computeLayers;
		ParallelReduction _ParallelReduction;
	};
}
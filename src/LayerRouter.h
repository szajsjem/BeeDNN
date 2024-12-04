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

		virtual bool init(size_t& in, size_t& out, bool debug = false) override;

		virtual bool has_weights() const override;
		virtual std::vector<MatrixFloat*> weights() const override;
		virtual std::vector<MatrixFloat*> gradient_weights() const override;

		virtual void save(std::ostream& to)const override;
		static Layer* load(std::istream& from);
		static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
		static std::string constructUsage();

		virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
		virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

	private:
		LayerRouter();
		Layer* _router;
		std::vector < Layer*> _Layers;
		float computeLayers;
		ParallelReduction _ParallelReduction;
	};
	REGISTER_LAYER(LayerRouter, "LayerRouter");
}
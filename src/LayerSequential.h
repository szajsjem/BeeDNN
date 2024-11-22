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

		virtual bool init(size_t& in, size_t& out, bool debug = false) override;

		virtual bool has_weights() const override;
		virtual std::vector<MatrixFloat*> weights() override;
		virtual std::vector<MatrixFloat*> gradient_weights() override;

		virtual void save(std::ostream& to)const override;
		static Layer* load(std::istream& from);
		static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
		static std::string constructUsage();

		virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
		virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

	private:
		LayerSequential();
		std::vector <Layer*> _Layers;
	};
	REGISTER_LAYER(LayerSequential, "LayerSequential");
}
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

		virtual bool init(size_t& in, size_t& out, bool debug = false) override;

		virtual bool has_weights() const override;

		virtual void save(std::ostream& to)const override;
		static Layer* load(std::istream& from);
		static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
		static std::string constructUsage();
	};
	REGISTER_LAYER(LayerSelfAttention, "LayerSelfAttention");
}
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

		virtual bool init(size_t& in, size_t& out, bool debug = false) override;

		virtual bool has_weights() const override;

		virtual void save(std::ostream& to)const override;
		static Layer* load(std::istream& from);
		static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
		static std::string constructUsage();

	private:
	};
	REGISTER_LAYER(LayerTranspose, "LayerTranspose");
}
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

		virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
		virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;


		virtual bool init(size_t& in, size_t& out, bool debug = false) override;

		virtual bool has_weights() const override;
		virtual std::vector<MatrixFloat*> weights() override;
		virtual std::vector<MatrixFloat*> gradient_weights() override;

		virtual void save(std::ostream& to)const override;
		static Layer* load(std::istream& from);
		static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
		static std::string constructUsage();
	private:
		Layer* _l2d;
		int _incolsize;
		int _outcolsize;
		int _outrowsize;
	};
	REGISTER_LAYER(LayerBatchTo2D, "LayerBatchTo2D");
}
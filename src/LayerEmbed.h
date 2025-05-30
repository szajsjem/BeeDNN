#pragma once

#include "Layer.h"
#include "Matrix.h"
namespace beednn {
	class LayerEmbed : public Layer
	{
	public:
		explicit LayerEmbed(const Index vocabSize, const Index dimensionSize, const Index maxPositon, const std::string& sBiasInitializer = "Zeros");
		virtual ~LayerEmbed() override;

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
		Index _pVocabSize, _pPositionSize, _pDimensionSize;
		MatrixFloat _bias2, _gradientBias2;
		MatrixFloat _bias, _gradientBias;
	};
	REGISTER_LAYER(LayerEmbed, "LayerEmbed");
}
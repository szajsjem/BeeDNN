#pragma once

#include <string>
#include "Matrix.h"
#include "Layer.h"
#include "LayerSoftmax.h"
#include "LayerDense.h"
#include "LayerStacked.h"
#include "LayerParallel.h"
#include "LayerSequential.h"
#include "LayerNormalize.h"
#include "LayerSelfAttention.h"
#include "LayerActivation.h"
#include "LayerTranspose.h"

namespace beednn {
	class LayerTransformerHeads : public LayerParallel
	{
	public:
		explicit LayerTransformerHeads(const int iDimmensionSize, const int iHeadVMem, const int iHeadQKMem, const int iNumHeads, const std::string& sWeightInitializer = "GlorotUniform", const std::string& sBiasInitializer = "Zeros");
		static std::string constructUsage();
		static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
	};
	REGISTER_LAYER(LayerTransformerHeads, "LayerTransformerHeads");
}
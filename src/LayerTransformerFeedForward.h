#pragma once

#include <string>
#include "Matrix.h"
#include "Layer.h"

#include "LayerNormalize.h"
#include "LayerDense.h"
#include "LayerActivation.h"
#include "LayerParallel.h"
#include "LayerSequential.h"
#include "LayerActivation.h"

namespace beednn {
	class LayerTransformerFeedForward : public LayerParallel
	{
	public:
		explicit LayerTransformerFeedForward(Index iDimmensionSize, Index iMemorySize, const std::string& sActivation, const std::string& sWeightInitializer = "GlorotUniform", const std::string& sBiasInitializer = "Zeros");
	};
	REGISTER_LAYER(LayerTransformerFeedForward, "LayerTransformerFeedForward");
};
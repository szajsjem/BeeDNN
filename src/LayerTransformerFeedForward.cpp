#include "LayerTransformerFeedForward.h"

namespace beednn {
	LayerTransformerFeedForward::LayerTransformerFeedForward(Index iDimmensionSize, Index iMemorySize, const string& sActivation, const std::string& sWeightInitializer = "GlorotUniform", const std::string& sBiasInitializer = "Zeros")
		:LayerParallel({ new LayerActivation("Identity"),
			new LayerSequential({
				new LayerNormalize(),
				new LayerDense(iDimmensionSize, iMemorySize, sWeightInitializer, sBiasInitializer),
				new LayerActivation(sActivation),
				new LayerDense(iMemorySize, iDimmensionSize, sWeightInitializer, sBiasInitializer)})
			}, SUM) {}
}
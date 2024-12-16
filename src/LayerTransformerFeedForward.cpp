#include "LayerTransformerFeedForward.h"

namespace beednn {
	LayerTransformerFeedForward::LayerTransformerFeedForward(const Index iDimmensionSize, const  Index iMemorySize, const std::string& sActivation, const std::string& sWeightInitializer, const std::string& sBiasInitializer)
		:LayerParallel({ 
			new LayerActivation("Identity"),
			new LayerSequential({
				new LayerNormalize(),
				new LayerDense(iDimmensionSize, iMemorySize, sWeightInitializer, sBiasInitializer),
				new LayerActivation(sActivation),
				new LayerDense(iMemorySize, iDimmensionSize, sWeightInitializer, sBiasInitializer)})
			}, SUM) {}
	std::string LayerTransformerFeedForward::constructUsage() {
		return "transformer feed forward block\nsReduction;sActivation;sWeightInitializer;sBiasInitializer\niDimensionSize;iMemorySize";
	}
}
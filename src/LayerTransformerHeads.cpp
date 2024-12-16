#include "LayerTransformerHeads.h"

namespace beednn {
	LayerTransformerHeads::LayerTransformerHeads(const int iDimmensionSize, const int iHeadVMem, const int iHeadQKMem, const int iNumHeads, const std::string& sWeightInitializer, const std::string& sBiasInitializer)
		:LayerParallel({
			new LayerActivation("Identity"),
			new LayerSequential({
				new LayerNormalize(),
				new LayerStacked({
					new LayerParallel({
						new LayerSequential({
							new LayerStacked({
								new LayerDense(iDimmensionSize,iHeadQKMem, sWeightInitializer, sBiasInitializer)
							},ROWSTACK,2),
							new LayerSelfAttention(),
							new LayerSoftmax()
						}),
						new LayerDense(iDimmensionSize,iHeadVMem, sWeightInitializer, sBiasInitializer),
					},DOT)
				},COLSTACK,iNumHeads),
				new LayerDense(iNumHeads * iHeadVMem, iDimmensionSize, sWeightInitializer, sBiasInitializer)
			})
			}, SUM) {}
	std::string LayerTransformerHeads::constructUsage() {
		return "multi-head attention block\nsReduction;sWeightInitializer;sBiasInitializer\niDimensionSize;iHeadVMem;iHeadQKMem;iNumHeads";
	}
}
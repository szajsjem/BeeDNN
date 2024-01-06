#include "LayerTransformerHeads.h"

namespace beednn {
	LayerTransformerHeads::LayerTransformerHeads(const int iDimmensionSize, const int iHeadMem, const int iNumHeads, const std::string& sWeightInitializer, const std::string& sBiasInitializer)
		:LayerParallel({
			new LayerActivation("Identity"),
			new LayerSequential({
				new LayerNormalize(),
				new LayerStacked({
					new LayerParallel({
						new LayerDense(iDimmensionSize,iHeadMem, sWeightInitializer, sBiasInitializer),
						new LayerSequential({
							new LayerStacked({
								new LayerDense(iDimmensionSize,iHeadMem, sWeightInitializer, sBiasInitializer)
							},None,2),
							new LayerSelfAttention(),
							new LayerSoftmax(),
						})
					},Dot)
				},None,iNumHeads),
				new LayerDense(iNumHeads * iHeadMem, iDimmensionSize, sWeightInitializer, sBiasInitializer)
			})
			}, SUM) {}
}
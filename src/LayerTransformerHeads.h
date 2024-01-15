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

namespace beednn {
	class LayerTransformerHeads : public LayerParallel
	{
	public:
		explicit LayerTransformerHeads(const int iDimmensionSize, const int iHeadMem, const int iNumHeads, const std::string& sWeightInitializer, const std::string& sBiasInitializer);
	};
}
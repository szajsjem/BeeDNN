#pragma once
#include <vector>
#include <algorithm>
#include "Matrix.h"
#include "Layer.h"
#include "ParallelReduction.h"
#include "LayerParallel.h"

namespace beednn {
	class LayerStacked : public LayerParallel
	{
	public:
		explicit LayerStacked(Layer* mStackedLayer, ParallelReduction mReduction, const int num);
	};
	REGISTER_LAYER(LayerStacked, "LayerStacked");
}
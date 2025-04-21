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
		static std::string constructUsage();
		static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
	};
	REGISTER_LAYER(LayerStacked, "LayerStacked");
}
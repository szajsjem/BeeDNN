#pragma once
#include <vector>
#include <algorithm>
#include "Matrix.h"
#include "Layer.h"
#include "LayerSequential.h"

namespace beednn {
	class LayerRepetetive : public LayerSequential
	{
	public:
		explicit LayerRepetetive(Layer* mStackedLayer, const int num);
		static std::string constructUsage();
	};
	REGISTER_LAYER(LayerRepetetive, "LayerRepetetive");
}
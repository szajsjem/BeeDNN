#include "LayerStacked .h"

namespace beednn {
	LayerStacked::LayerStacked(Layer* mStackedLayer, ParallelReduction mReduction, const int num)
		:LayerParallel(std::generate_n(std::vector<Layer*>(), num, [&]() {return mStackedLayer->clone(); }), mReduction) {
		delete mStackedLayer;
	}
}
#include "LayerStacked.h"

namespace std {
	template<class T, class I>
	std::vector<T> generate_v(I num, std::function<T()> func) {
		std::vector<T> out(num);
		std::generate_n(out.begin(), num, func);
		return out;
	}
}

namespace beednn {
	LayerStacked::LayerStacked(Layer* mStackedLayer, ParallelReduction mReduction, const int num)
		:LayerParallel(std::generate_v<Layer*>(num, [&]() {return mStackedLayer->clone(); }), mReduction) {
		delete mStackedLayer;
	}
}
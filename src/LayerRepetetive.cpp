#include "LayerRepetetive.h"

namespace std {
	template<class T, class I>
	std::vector<T> generate_w(I num, std::function<T()> func) {
		std::vector<T> out(num);
		std::generate_n(out.begin(), num, func);
		return out;
	}
}

namespace beednn {
	LayerRepetetive::LayerRepetetive(Layer* mStackedLayer,  const int num)
		:LayerSequential(std::generate_w<Layer*>(num, [&]() {return mStackedLayer->clone(); })) {
		delete mStackedLayer;
	}
	std::string LayerRepetetive::constructUsage() {
		return "stacks multiple copies of a layer in series\nReduction;layer\nNum of copies in parallel";
	}
}
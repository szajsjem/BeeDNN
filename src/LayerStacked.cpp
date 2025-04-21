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
	std::string LayerStacked::constructUsage() {
		return "stacks multiple copies of a layer\nReduction;layer\nNum of copies in parallel";
	}
	Layer* LayerStacked::construct(std::initializer_list<float> fArgs, std::string sArg) {
		if (fArgs.size() != 1) return nullptr; // Number of copies

		// Split into reduction and layer pointer
		size_t pos = sArg.find(';');
		if (pos == std::string::npos) return nullptr;

		std::string reductionStr = sArg.substr(0, pos);
		std::string layerPtr = sArg.substr(pos + 1);

		ParallelReduction reduction = reductionFromString(reductionStr);
		Layer* layer = (Layer*)std::stoull(layerPtr, nullptr, 16);

		return new LayerStacked(layer, reduction, *fArgs.begin());
	}
}
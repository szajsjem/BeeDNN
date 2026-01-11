#include "LayerStacked.h"

namespace std {
template <class T, class I>
std::vector<T> generate_v(I num, std::function<T()> func) {
  std::vector<T> out(num);
  std::generate_n(out.begin(), num, func);
  return out;
}
} // namespace std

namespace beednn {
LayerStacked::LayerStacked(Layer *mStackedLayer, ParallelReduction mReduction,
                           const int num)
    : LayerParallel(std::generate_v<Layer *>(
                        num, [&]() { return mStackedLayer->clone(); }),
                    mReduction) {
  delete mStackedLayer;
}
LayerStacked::LayerStacked(std::vector<Layer *> layers,
                           ParallelReduction mReduction)
    : LayerParallel(layers, mReduction) {}

std::string LayerStacked::constructUsage() {
  return "stacks multiple copies of a layer\nsReduction;sLayerPtr\niNumCopies";
}
Layer *LayerStacked::construct(std::initializer_list<float> fArgs,
                               std::string sArg) {
  if (fArgs.size() != 1)
    return nullptr; // Number of copies

  // Split into reduction and layer pointer
  size_t pos = sArg.find(';');
  if (pos == std::string::npos)
    return nullptr;

  std::string reductionStr = sArg.substr(0, pos);
  std::string layerPtr = sArg.substr(pos + 1);

  ParallelReduction reduction = reductionFromString(reductionStr);
  Layer *layer = (Layer *)std::stoull(layerPtr, nullptr, 16);

  return new LayerStacked(layer, reduction, *fArgs.begin());
}
void LayerStacked::save(std::ostream &to) const {
  to << "LayerStacked" << std::endl;
  to << reductionToString(_ParallelReduction) << std::endl;
  to << _Layers.size() << std::endl;
  for (auto layer : _Layers) {
    layer->save(to);
  }
}
Layer *LayerStacked::load(std::istream &from) {
  std::string sReduction;
  from >> sReduction;
  ParallelReduction reduction = reductionFromString(sReduction);
  size_t numLayers;
  from >> numLayers;
  std::vector<Layer *> layers;
  for (size_t i = 0; i < numLayers; ++i) {
    layers.push_back(Layer::load(from));
  }
  return new LayerStacked(layers, reduction);
}
} // namespace beednn
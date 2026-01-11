#include "LayerRepetetive.h"

namespace std {
template <class T, class I>
std::vector<T> generate_w(I num, std::function<T()> func) {
  std::vector<T> out(num);
  std::generate_n(out.begin(), num, func);
  return out;
}
} // namespace std

namespace beednn {
LayerRepetetive::LayerRepetetive(Layer *mStackedLayer, const int num)
    : LayerSequential(std::generate_w<Layer *>(
          num, [&]() { return mStackedLayer->clone(); })) {
  delete mStackedLayer;
}
LayerRepetetive::LayerRepetetive(std::vector<Layer *> layers)
    : LayerSequential(layers) {}

std::string LayerRepetetive::constructUsage() {
  return "stacks multiple copies of a layer in series\nsLayerPtr\niNumCopies";
}
Layer *LayerRepetetive::construct(std::initializer_list<float> fArgs,
                                  std::string sArg) {
  if (fArgs.size() != 1)
    return nullptr; // Number of copies

  Layer *layer = (Layer *)std::stoull(sArg, nullptr, 16);
  return new LayerRepetetive(layer, *fArgs.begin());
}

void LayerRepetetive::save(std::ostream &to) const {
  to << "LayerRepetetive" << std::endl;
  to << _Layers.size() << std::endl;
  for (auto layer : _Layers) {
    layer->save(to);
  }
}

Layer *LayerRepetetive::load(std::istream &from) {
  size_t numLayers;
  from >> numLayers;
  std::vector<Layer *> layers;
  for (size_t i = 0; i < numLayers; ++i) {
    layers.push_back(Layer::load(from));
  }
  return new LayerRepetetive(layers);
}
} // namespace beednn
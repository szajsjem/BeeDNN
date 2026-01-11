#include "LayerTransformerFeedForward.h"

namespace beednn {
LayerTransformerFeedForward::LayerTransformerFeedForward(
    const Index iDimmensionSize, const Index iMemorySize,
    const std::string &sActivation, const std::string &sWeightInitializer,
    const std::string &sBiasInitializer)
    : LayerParallel(
          {new LayerActivation("Identity"),
           new LayerSequential(
               {new LayerNormalize(),
                new LayerDense(iDimmensionSize, iMemorySize, sWeightInitializer,
                               sBiasInitializer),
                new LayerActivation(sActivation),
                new LayerDense(iMemorySize, iDimmensionSize, sWeightInitializer,
                               sBiasInitializer)})},
          SUM) {}

LayerTransformerFeedForward::LayerTransformerFeedForward(
    std::vector<Layer *> layers, ParallelReduction reduction)
    : LayerParallel(layers, reduction) {}

std::string LayerTransformerFeedForward::constructUsage() {
  return "transformer feed forward "
         "block\nsReduction;sActivation;sWeightInitializer;"
         "sBiasInitializer\niDimensionSize;iMemorySize";
}
Layer *
LayerTransformerFeedForward::construct(std::initializer_list<float> fArgs,
                                       std::string sArg) {
  if (fArgs.size() != 2)
    return nullptr; // iDimensionSize, iMemorySize
  auto args = fArgs.begin();
  Index iDimensionSize = (Index)*args;
  Index iMemorySize = (Index) * (args + 1);

  // Parse strings: sReduction;sActivation;sWeightInitializer;sBiasInitializer
  std::vector<std::string> sParams;
  std::stringstream ss(sArg);
  std::string item;
  while (std::getline(ss, item, ';')) {
    sParams.push_back(item);
  }

  if (sParams.size() < 4)
    return nullptr; // Expecting 4 string params

  // sParams[0] is reduction, but constructor doesn't take it (defaults to SUM)
  // We ignore it for now or enforce it matches SUM?
  // Given user request to implement construct, we'll map to existing
  // constructor.

  return new LayerTransformerFeedForward(iDimensionSize, iMemorySize,
                                         sParams[1], sParams[2], sParams[3]);
}
void LayerTransformerFeedForward::save(std::ostream &to) const {
  to << "LayerTransformerFeedForward" << std::endl;
  to << reductionToString(_ParallelReduction) << std::endl;
  to << _Layers.size() << std::endl;
  for (auto layer : _Layers) {
    layer->save(to);
  }
}
Layer *LayerTransformerFeedForward::load(std::istream &from) {
  std::string sReduction;
  from >> sReduction;
  ParallelReduction reduction = reductionFromString(sReduction);
  size_t numLayers;
  from >> numLayers;
  std::vector<Layer *> layers;
  for (size_t i = 0; i < numLayers; ++i) {
    layers.push_back(Layer::load(from));
  }
  return new LayerTransformerFeedForward(layers, reduction);
}
} // namespace beednn
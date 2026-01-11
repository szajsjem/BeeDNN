#include "LayerTransformerHeads.h"

namespace beednn {
LayerTransformerHeads::LayerTransformerHeads(
    const int iDimmensionSize, const int iHeadVMem, const int iHeadQKMem,
    const int iNumHeads, const std::string &sWeightInitializer,
    const std::string &sBiasInitializer)
    : LayerParallel(
          {new LayerActivation("Identity"),
           new LayerSequential(
               {new LayerNormalize(),
                new LayerStacked(
                    new LayerParallel(
                        {
                            new LayerSequential(
                                {new LayerStacked(
                                     new LayerDense(iDimmensionSize, iHeadQKMem,
                                                    sWeightInitializer,
                                                    sBiasInitializer),
                                     ROWSTACK, 2),
                                 new LayerSelfAttention(), new LayerSoftmax()}),
                            new LayerDense(iDimmensionSize, iHeadVMem,
                                           sWeightInitializer,
                                           sBiasInitializer),
                        },
                        DOT),
                    COLSTACK, iNumHeads),
                new LayerDense(iNumHeads * iHeadVMem, iDimmensionSize,
                               sWeightInitializer, sBiasInitializer)})},
          SUM) {}
std::string LayerTransformerHeads::constructUsage() {
  return "multi-head attention "
         "block\nsReduction;sWeightInitializer;"
         "sBiasInitializer\niDimensionSize;iHeadVMem;iHeadQKMem;iNumHeads";
}
Layer *LayerTransformerHeads::construct(std::initializer_list<float> fArgs,
                                        std::string sArg) {
  if (fArgs.size() != 4)
    return nullptr; // dimSize, headVMem, headQKMem, numHeads
  auto args = fArgs.begin();

  // Split sArg into weightInit and biasInit
  size_t pos = sArg.find(';');
  if (pos == std::string::npos)
    return nullptr;

  std::string weightInit = sArg.substr(0, pos);
  std::string biasInit = sArg.substr(pos + 1);

  return new LayerTransformerHeads(*args, *(args + 1), *(args + 2), *(args + 3),
                                   weightInit, biasInit);
}

LayerTransformerHeads::LayerTransformerHeads(std::vector<Layer *> layers,
                                             ParallelReduction reduction)
    : LayerParallel(layers, reduction) {}

void LayerTransformerHeads::save(std::ostream &to) const {
  to << "LayerTransformerHeads" << std::endl;
  to << reductionToString(_ParallelReduction) << std::endl;
  to << _Layers.size() << std::endl;
  for (auto layer : _Layers) {
    layer->save(to);
  }
}
Layer *LayerTransformerHeads::load(std::istream &from) {
  std::string sReduction;
  from >> sReduction;
  ParallelReduction reduction = reductionFromString(sReduction);
  size_t numLayers;
  from >> numLayers;
  std::vector<Layer *> layers;
  for (size_t i = 0; i < numLayers; ++i) {
    layers.push_back(Layer::load(from));
  }
  return new LayerTransformerHeads(layers, reduction);
}
} // namespace beednn
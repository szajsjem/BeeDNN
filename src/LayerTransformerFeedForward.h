#pragma once

#include "Layer.h"
#include "Matrix.h"
#include <string>


#include "LayerActivation.h"
#include "LayerDense.h"
#include "LayerNormalize.h"
#include "LayerParallel.h"
#include "LayerSequential.h"


namespace beednn {
class LayerTransformerFeedForward : public LayerParallel {
public:
  explicit LayerTransformerFeedForward(
      Index iDimmensionSize, Index iMemorySize, const std::string &sActivation,
      const std::string &sWeightInitializer = "GlorotUniform",
      const std::string &sBiasInitializer = "Zeros");
  explicit LayerTransformerFeedForward(std::vector<Layer *> layers,
                                       ParallelReduction reduction);
  static std::string constructUsage();
  static Layer *construct(std::initializer_list<float> fArgs, std::string sArg);

  void save(std::ostream &to) const override;
  static Layer *load(std::istream &from);
};
REGISTER_LAYER(LayerTransformerFeedForward, "LayerTransformerFeedForward");
}; // namespace beednn
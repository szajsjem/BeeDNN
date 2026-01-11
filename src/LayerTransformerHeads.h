#pragma once

#include "Layer.h"
#include "LayerActivation.h"
#include "LayerDense.h"
#include "LayerNormalize.h"
#include "LayerParallel.h"
#include "LayerSelfAttention.h"
#include "LayerSequential.h"
#include "LayerSoftmax.h"
#include "LayerStacked.h"
#include "LayerTranspose.h"
#include "Matrix.h"
#include <string>


namespace beednn {
class LayerTransformerHeads : public LayerParallel {
public:
  explicit LayerTransformerHeads(
      const int iDimmensionSize, const int iHeadVMem, const int iHeadQKMem,
      const int iNumHeads,
      const std::string &sWeightInitializer = "GlorotUniform",
      const std::string &sBiasInitializer = "Zeros");
  explicit LayerTransformerHeads(std::vector<Layer *> layers,
                                 ParallelReduction reduction);
  static std::string constructUsage();
  static Layer *construct(std::initializer_list<float> fArgs, std::string sArg);

  void save(std::ostream &to) const override;
  static Layer *load(std::istream &from);
};
REGISTER_LAYER(LayerTransformerHeads, "LayerTransformerHeads");
} // namespace beednn
#pragma once
#include "Layer.h"
#include "LayerParallel.h"
#include "Matrix.h"
#include "ParallelReduction.h"
#include <algorithm>
#include <vector>


namespace beednn {
class LayerStacked : public LayerParallel {
public:
  explicit LayerStacked(Layer *mStackedLayer, ParallelReduction mReduction,
                        const int num);
  explicit LayerStacked(std::vector<Layer *> layers,
                        ParallelReduction mReduction);
  static std::string constructUsage();
  static Layer *construct(std::initializer_list<float> fArgs, std::string sArg);

  void save(std::ostream &to) const override;
  static Layer *load(std::istream &from);
};
REGISTER_LAYER(LayerStacked, "LayerStacked");
} // namespace beednn
#pragma once
#include "Layer.h"
#include "LayerSequential.h"
#include "Matrix.h"
#include <algorithm>
#include <vector>


namespace beednn {
class LayerRepetetive : public LayerSequential {
public:
  explicit LayerRepetetive(Layer *mStackedLayer, const int num);
  explicit LayerRepetetive(std::vector<Layer *> layers);
  static std::string constructUsage();
  static Layer *construct(std::initializer_list<float> fArgs, std::string sArg);

  void save(std::ostream &to) const override;
  static Layer *load(std::istream &from);
};
REGISTER_LAYER(LayerRepetetive, "LayerRepetetive");
} // namespace beednn
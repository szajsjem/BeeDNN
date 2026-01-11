/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "LayerSequential.h"
#include "Matrix.h"

namespace beednn {
class LayerDense : public LayerSequential {
  explicit LayerDense(std::vector<Layer *> layers);
public:
  explicit LayerDense(Index iInputSize, Index iOutputSize,
                      const std::string &sWeightInitializer = "GlorotUniform",
                      const std::string &sBiasInitializer = "Zeros");

  void save(std::ostream &to) const override;
  static Layer *load(std::istream &from);

  static std::string constructUsage();
  static Layer *construct(std::initializer_list<float> fArgs, std::string sArg);
};
REGISTER_LAYER(LayerDense, "LayerDense");
} // namespace beednn

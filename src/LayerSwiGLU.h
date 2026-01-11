/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "LayerGatedActivation.h"
namespace beednn {
class LayerSwiGLU : public LayerGatedActivation {
public:
  explicit LayerSwiGLU();
  static std::string constructUsage();
  static Layer *construct(std::initializer_list<float> fArgs, std::string sArg);

  void save(std::ostream &to) const override;
  static Layer *load(std::istream &from);
};
REGISTER_LAYER(LayerSwiGLU, "LayerSwiGLU");
} // namespace beednn

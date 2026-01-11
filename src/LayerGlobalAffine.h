/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

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
class LayerGlobalAffine : public LayerSequential {
public:
  explicit LayerGlobalAffine();
  explicit LayerGlobalAffine(std::vector<Layer *> layers);
  static std::string constructUsage();
  static Layer *construct(std::initializer_list<float> fArgs, std::string sArg);

  void save(std::ostream &to) const override;
  static Layer *load(std::istream &from);
};
REGISTER_LAYER(LayerGlobalAffine, "LayerGlobalAffine");
} // namespace beednn

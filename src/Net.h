/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/
#pragma once

#include "Matrix.h"
#include <vector>

namespace beednn {
class Layer;

class Net {
public:
  Net();
  virtual ~Net();
  Net &operator=(const Net &other);

  void clear();
  size_t init(size_t inputDataSize = -1, bool debug = false);

  // add a layer, take the ownership of the layer
  void add(Layer *l);

  // replace a layer, take the ownership of the layer
  void replace(size_t iLayer, Layer *l);

  const std::vector<Layer *> layers() const;
  Layer &layer(size_t iLayer);
  const Layer &layer(size_t iLayer) const;
  size_t size() const;

  void set_classification_mode(bool bClassificationMode); // true by default
  bool is_classification_mode() const;

  void predict(const MatrixFloat &mIn, MatrixFloat &mOut) const;
  void predict_classes(const MatrixFloat &mIn, MatrixFloat &mClass) const;

  void set_train_mode(bool bTrainMode); // set to true if training, set to false
                                        // if testing (default)

  std::vector<MatrixFloat> &internalCalculationMatrices() {
    return _internalCalculationMatrices;
  }
  int startICM(int i) {
      return _layerStarticm[i];
  }

private:
  bool _bTrainMode;
  std::vector<Layer *> _layers;
  std::vector<MatrixFloat> _internalCalculationMatrices;
  std::vector<size_t> _layerStarticm;
  bool _bClassificationMode;
};
} // namespace beednn

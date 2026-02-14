/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Net.h"
#include "Layer.h"

#include "Matrix.h"

#include <cmath>
namespace beednn {

/////////////////////////////////////////////////////////////////////////////////////////////////
Net::Net() {
  _bTrainMode = false;
  _bClassificationMode = true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Net::~Net() { clear(); }
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::clear() {
  for (unsigned int i = 0; i < _layers.size(); i++)
    delete _layers[i];

  _layers.clear();
  _layerStarticm.clear();
  _internalCalculationMatrices.clear();
  _bTrainMode = false;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Net &Net::operator=(const Net &other) {
  clear();

  for (size_t i = 0; i < other._layers.size(); i++)
    _layers.push_back(other._layers[i]->clone());

  _bClassificationMode = other._bClassificationMode;

  return *this;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
// add the layer, take the ownership of the layer
void Net::add(Layer *l) { _layers.push_back(l); }
/////////////////////////////////////////////////////////////////////////////////////////////////
// replace a layer, take the ownership of the layer
void Net::replace(size_t iLayer, Layer *l) {
  assert(iLayer < _layers.size());

  delete _layers[iLayer];
  _layers[iLayer] = l;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::predict(const MatrixFloat &mIn, MatrixFloat &mOut) const {
  // todo cut in mini batches so save memory
  MatrixFloat mTemp = mIn;
  for (unsigned int i = 0; i < _layers.size(); i++) {
    _layers[i]->forward(mTemp, mOut);
    mTemp = mOut; // todo avoid resize
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////
void Net::set_classification_mode(bool bClassificationMode) {
  _bClassificationMode = bClassificationMode;
}
/////////////////////////////////////////////////////////////////////////////////////////////
bool Net::is_classification_mode() const { return _bClassificationMode; }
/////////////////////////////////////////////////////////////////////////////////////////////
void Net::predict_classes(const MatrixFloat &mIn, MatrixFloat &mClass) const {
  MatrixFloat mOut;
  predict(mIn, mOut);

  if (mOut.cols() != 1)
    rowsArgmax(mOut, mClass); // one hot case
  else {
    mClass.resize(mOut.rows(), 1);
    for (int i = 0; i < mOut.rows(); i++)
      mClass(i, 0) = std::roundf(mOut(i, 0)); // categorical case
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::set_train_mode(bool bTrainMode) {
  _bTrainMode = bTrainMode;

  for (unsigned int i = 0; i < _layers.size(); i++)
    _layers[i]->set_train_mode(bTrainMode);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
const std::vector<Layer *> Net::layers() const { return _layers; }
/////////////////////////////////////////////////////////////////////////////////////////////
Layer &Net::layer(size_t iLayer) { return *(_layers[iLayer]); }
/////////////////////////////////////////////////////////////////////////////////////////////
const Layer &Net::layer(size_t iLayer) const { return *(_layers[iLayer]); }
/////////////////////////////////////////////////////////////////////////////////////////////
size_t Net::size() const { return _layers.size(); }
/////////////////////////////////////////////////////////////////////////////////////////////
size_t Net::init(size_t inputDataSize, bool debug) {
  // Clear previous initialization state for safe re-initialization
  _layerStarticm.clear();
  _internalCalculationMatrices.clear();

  size_t in = inputDataSize, out = -1;

  for (unsigned int i = 0; i < _layers.size(); i++) {
    _layerStarticm.push_back(_internalCalculationMatrices.size());
    _layers[i]->init(in, out, _internalCalculationMatrices, debug);
    in = out;
    out = -1;
  }
  return in;
}
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
// Distributed Training Implementation
std::vector<float> Net::get_params() const {
  std::vector<float> params;
  for (const auto *l : _layers) {
    if (l->has_weights()) {
      for (const auto *w : l->weights()) {
        if (w) {
          for (size_t i = 0; i < w->size(); ++i)
            params.push_back((*w)(i));
        }
      }
    }
  }
  return params;
}

void Net::set_params(const std::vector<float> &params) {
  size_t idx = 0;
  for (auto *l : _layers) {
    if (l->has_weights()) {
      for (auto *w : l->weights()) {
        if (w) {
          for (size_t i = 0; i < w->size(); ++i) {
            if (idx < params.size())
              (*w)(i) = params[idx++];
          }
        }
      }
    }
  }
}

void Net::mix_params(const std::vector<float> &other_params, float theta) {
  if (theta == 1.0f) {
    set_params(other_params);
    return;
  }
  if (theta == 0.0f) {
    return;
  }

  size_t idx = 0;
  for (auto *l : _layers) {
    if (l->has_weights()) {
      for (auto *w : l->weights()) {
        if (w) {
          for (size_t i = 0; i < w->size(); ++i) {
            if (idx < other_params.size()) {
              float current = (*w)(i);
              float other = other_params[idx++];
              (*w)(i) = (1.0f - theta) * current + theta * other;
            }
          }
        }
      }
    }
  }
}

std::vector<float> Net::get_param_gradients() const {
  std::vector<float> grads;
  for (const auto *l : _layers) {
    if (l->has_weights()) {
      for (const auto *g : l->gradient_weights()) {
        if (g) {
          for (size_t i = 0; i < g->size(); ++i)
            grads.push_back((*g)(i));
        }
      }
    }
  }
  return grads;
}

void Net::accumulate_weight_diff_to_grad(
    const std::vector<float> &received_weights) {
  // Adds (CurrentWeight - ReceivedWeight) to Gradient
  size_t idx = 0;
  for (auto *l : _layers) {
    if (l->has_weights()) {
      const auto &weights = l->weights();
      const auto &grads = l->gradient_weights();

      // Assuming weights and grads are aligned (same count, same dims)
      for (size_t matIdx = 0; matIdx < weights.size(); ++matIdx) {
        MatrixFloat *w = weights[matIdx];
        MatrixFloat *g = grads[matIdx];

        if (w && g) {
          for (size_t i = 0; i < w->size(); ++i) {
            if (idx < received_weights.size()) {
              float diff = (*w)(i)-received_weights[idx++];
              (*g)(i) += diff;
            }
          }
        }
      }
    }
  }
}
} // namespace beednn

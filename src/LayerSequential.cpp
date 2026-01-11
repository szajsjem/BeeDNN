#include "LayerSequential.h"

namespace beednn {
LayerSequential::LayerSequential(std::vector<Layer *> mSequentialLayers)
    : Layer("LayerSequential") {
  for (auto layer : mSequentialLayers) {
    _Layers.push_back(layer);
  }
}
///////////////////////////////////////////////////////////////////////////////
LayerSequential::LayerSequential() : Layer("LayerSequential") {}
///////////////////////////////////////////////////////////////////////////////
LayerSequential::~LayerSequential() {
  for (auto x : _Layers) {
    delete x;
  }
}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerSequential::clone() const {
  LayerSequential *pLayer = new LayerSequential();
  for (auto x : _Layers) {
    pLayer->_Layers.push_back(x->clone());
  }
  return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerSequential::init(
    size_t &in, size_t &out,
    std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {

  // Preallocate buffers for intermediate outputs (N-1 buffers for N layers)
  if (_Layers.size() > 1) {
    for (size_t i = 0; i < _Layers.size(); ++i) {
      internalCalculationMatrices.emplace_back();
    }
  }

  size_t tmp = in, tmpo;
  _offsetTable.clear();
  for (const auto &l : _Layers) {
    _offsetTable.push_back((int)internalCalculationMatrices.size());
    tmpo = -1;
    if (!l->init(tmp, tmpo, internalCalculationMatrices, debug))
      return false;
    if (in == -1)
      in = tmp;
    tmp = tmpo;
  }
  if (out != -1)
    if (tmp != out)
      return false;
  out = tmp;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSequential::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  MatrixFloat mf = mIn;
  for (auto x : _Layers) {
    x->forward(mf, mOut);
    mf = mOut;
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerSequential::backpropagation(
    const MatrixFloat &mIn, const MatrixFloat &mGradientOut,
    MatrixFloat &mGradientIn,
    std::vector<MatrixFloat> &internalCalculationMatrices, int start) {

  // 1. Recompute forward pass to populate intermediate buffers
  // Buffers start at _internalMatrixStart.
  // internalCalculationMatrices[_internalMatrixStart + i] stores output of
  // _Layers[i]

  const MatrixFloat *pInput = &mIn;
  if (_Layers.size() > 1) {
    for (size_t i = 0; i < _Layers.size() - 1; ++i) {
      MatrixFloat &buffer = internalCalculationMatrices[start + i];
      _Layers[i]->forward(*pInput, buffer);
      pInput = &buffer;
    }
  }

  // Backpropagate through layers in reverse order
  MatrixFloat grad = mGradientOut;
  MatrixFloat tempGrad; // We could optimize this too?
  // User asked "internal matrices ... used in backpropagation".
  // Optimizing tempGrad (gradient between layers) would require N-1 gradient
  // buffers? Current implementation reuses one `tempGrad`. If we want to
  // preallocate it, we can push one extra matrix in init. Let's stick to the
  // requested vector usage for now.

  // Iterate from last layer down to second layer (internal gradients)
  if (_Layers.size() > 1) {
    for (size_t i = _Layers.size() - 1; i > 0; --i) {
      // Input to Layer i is output of Layer i-1 (Buffer i-1)
      const MatrixFloat &layerInput =
          internalCalculationMatrices[start + i - 1];

      tempGrad.resize(
          0, 0); // Force assignment/overwrite behavior in accumulating layers
      _Layers[i]->backpropagation(layerInput, grad, tempGrad,
                                  internalCalculationMatrices, _offsetTable[i]);
      grad = tempGrad;
    }
  }

  // Final backprop to the first layer (which writes to mGradientIn)
  if (_Layers.size() > 0) {
    // Input to Layer 0 is mIn
    _Layers[0]->backpropagation(mIn, grad, mGradientIn,
                                internalCalculationMatrices, _offsetTable[0]);
  }
}
///////////////////////////////////////////////////////////////
bool LayerSequential::has_weights() const {
  for (auto layer : _Layers)
    if (layer->has_weights())
      return true;
  return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerSequential::weights() const {
  std::vector<MatrixFloat *> v;
  for (auto layer : _Layers)
    if (layer->has_weights()) {
      auto vi = layer->weights();
      if (vi.size() > 0) {
        v.insert(v.end(), vi.begin(), vi.end());
      }
    }
  return v;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerSequential::gradient_weights() const {
  std::vector<MatrixFloat *> v;
  for (auto layer : _Layers)
    if (layer->has_weights()) {
      auto vi = layer->gradient_weights();
      if (vi.size() > 0) {
        v.insert(v.end(), vi.begin(), vi.end());
      }
    }
  return v;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSequential::save(std::ostream &to) const {
  to << "LayerSequential" << std::endl;
  to << _Layers.size() << std::endl;
  for (auto layer : _Layers) {
    layer->save(to);
  }
}
///////////////////////////////////////////////////////////////
Layer *LayerSequential::load(std::istream &from) {
  size_t numLayers;
  from >> numLayers;
  std::vector<Layer *> layers;
  for (size_t i = 0; i < numLayers; ++i) {
    layers.push_back(Layer::load(from));
  }
  return new LayerSequential(layers);
}
///////////////////////////////////////////////////////////////
Layer *LayerSequential::construct(std::initializer_list<float> fArgs,
                                  std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;

  // Parse layer pointers string - split on commas
  std::vector<Layer *> layers;
  std::string::size_type start = 0;
  std::string::size_type end;

  while ((end = sArg.find(',', start)) != std::string::npos) {
    Layer *layer =
        (Layer *)std::stoull(sArg.substr(start, end - start), nullptr, 16);
    layers.push_back(layer);
    start = end + 1;
  }
  // Get last layer
  if (start < sArg.length()) {
    Layer *layer = (Layer *)std::stoull(sArg.substr(start), nullptr, 16);
    layers.push_back(layer);
  }

  return new LayerSequential(layers);
}
///////////////////////////////////////////////////////////////
std::string LayerSequential::constructUsage() {
  return "sequential layer container\nlayers\niNumLayers";
}
///////////////////////////////////////////////////////////////
} // namespace beednn
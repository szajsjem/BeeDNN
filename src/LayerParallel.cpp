#include "LayerParallel.h"

namespace beednn {
LayerParallel::LayerParallel(std::vector<Layer *> mParallelLayers,
                             ParallelReduction mReduction)
    : Layer("LayerParallel") {
  for (auto layer : mParallelLayers) {
    _Layers.push_back(layer);
  }
  _ParallelReduction = mReduction;
}
///////////////////////////////////////////////////////////////////////////////
LayerParallel::LayerParallel() : Layer("LayerParallel") {}
///////////////////////////////////////////////////////////////////////////////
LayerParallel::~LayerParallel() {
  for (auto x : _Layers) {
    delete x;
  }
}
///////////////////////////////////////////////////////////////////////////////
Layer *LayerParallel::clone() const {
  LayerParallel *pLayer = new LayerParallel();
  for (auto x : _Layers) {
    pLayer->_Layers.push_back(x->clone());
  }
  pLayer->_ParallelReduction = _ParallelReduction;
  return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerParallel::init(size_t &in, size_t &out,
                         std::vector<MatrixFloat> &internalCalculationMatrices,
                         bool debug) {
  // 1. Buffer for mLocalGradientIn
  internalCalculationMatrices.emplace_back();
  // 2. Buffer for temporaries (mf)
  internalCalculationMatrices.emplace_back();

  if (_ParallelReduction != COLSTACK) {
    _offsetTable.clear();
    for (auto l : _Layers) {
      _offsetTable.push_back((int)internalCalculationMatrices.size());
      if (!l->init(in, out, internalCalculationMatrices, debug))
        return false;
    }
  } else {
    size_t clout = 0;
    _offsetTable.clear();
    for (auto l : _Layers) {
      _offsetTable.push_back((int)internalCalculationMatrices.size());
      size_t tout = -1;
      if (!l->init(in, tout, internalCalculationMatrices, debug))
        return false;
      if (tout == -1 || clout == -1)
        clout = -1;
      else
        clout += tout;
    }
    if (out == -1)
      out = clout;
    if (out != clout)
      return false;
  }
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
///////////////////////////////////////////////////////////////////////////////
void LayerParallel::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  mOut.resize(0, 0); // todo save output sizes of diffrent layers
  for (auto x : _Layers) {
    MatrixFloat mf;
    x->forward(mIn, mf);
    if (mOut.size() == 0) {
      mOut = mf;
    } else if (_ParallelReduction == SUM) {
      mOut += mf;
    } else if (_ParallelReduction == DOT) {
      mOut *= mf;
    } else if (_ParallelReduction == ROWSTACK) {
      mOut = concatenateRows(mOut, mf);
    } else if (_ParallelReduction == COLSTACK) {
      mOut = concatenateCols(mOut, mf);
    }
  }
}
///////////////////////////////////////////////////////////////////////////////
void LayerParallel::backpropagation(
    const MatrixFloat &mIn, const MatrixFloat &mGradientOut,
    MatrixFloat &mGradientIn,
    std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
  MatrixFloat &mLocalGradientIn = internalCalculationMatrices[start];
  MatrixFloat &mf = internalCalculationMatrices[start + 1];

  // Initialize accumulators
  mLocalGradientIn.resizeLike(mIn);
  mLocalGradientIn.setZero();

  if (_ParallelReduction == ROWSTACK) {
    Index start = 0, n = mGradientOut.rows();
    for (size_t i = 0; i < _Layers.size(); ++i) {
      auto x = _Layers[i];
      MatrixFloat temp; // Forward still uses local temp? forward signature
                        // wasn't changed.
      x->forward(mIn, temp);
      n = temp.rows();

      // assert(start+n<=mGradientOut.rows())
      mf.resize(0, 0);
      x->backpropagation(mIn, mGradientOut.middleRows(start, n), mf,
                         internalCalculationMatrices, _offsetTable[i]);
      mLocalGradientIn += mf;

      start += n;
    }
  }
  if (_ParallelReduction == COLSTACK) {
    Index start = 0, n = mGradientOut.cols();
    for (size_t i = 0; i < _Layers.size(); ++i) {
      auto x = _Layers[i];
      MatrixFloat temp;
      x->forward(mIn, temp);
      n = temp.cols();

      // assert(start+n<=mGradientOut.rows())
      mf.resize(0, 0);
      x->backpropagation(mIn, mGradientOut.middleCols(start, n), mf,
                         internalCalculationMatrices, _offsetTable[i]);
      mLocalGradientIn += mf;

      start += n;
    }
  }
  if (_ParallelReduction == SUM) {
    for (size_t i = 0; i < _Layers.size(); ++i) {
      auto x = _Layers[i];
      mf.resize(0, 0);
      x->backpropagation(mIn, mGradientOut, mf, internalCalculationMatrices,
                         _offsetTable[i]);
      mLocalGradientIn += mf;
    }
  }
  if (_ParallelReduction == DOT) {
    std::vector<MatrixFloat> mOuts(_Layers.size());
    for (int i = 0; i < _Layers.size(); i++)
      _Layers[i]->forward(mIn, mOuts[i]);

    MatrixFloat mG = mGradientOut;
    for (int i = _Layers.size() - 1; i >= 0; i--) {
      MatrixFloat mD;
      mf.resize(0, 0);
      if (i > 0) {
        mD = mOuts[0];
        for (int j = 1; j < i; j++) {
          mD *= mOuts[j];
        }
        _Layers[i]->backpropagation(mIn, mD.transpose() * mG, mf,
                                    internalCalculationMatrices,
                                    _offsetTable[i]); // todo check mD*mG
        mG *= mOuts[i].transpose();
      } else {
        _Layers[i]->backpropagation(mIn, mG, mf, internalCalculationMatrices,
                                    _offsetTable[i]); // todo check mD*mG
      }
      // mG *= mOuts[i].inverse();//inverse symbol not found
      mLocalGradientIn += mf;
    }
  }
  mLocalGradientIn /= _Layers.size();

  if (mGradientIn.size() == 0) {
    mGradientIn = mLocalGradientIn;
  } else {
    mGradientIn += mLocalGradientIn;
  }
}
///////////////////////////////////////////////////////////////
bool LayerParallel::has_weights() const {
  for (auto layer : _Layers)
    if (layer->has_weights())
      return true;
  return false;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerParallel::weights() const {
  std::vector<MatrixFloat *> v;
  for (auto layer : _Layers)
    if (layer->has_weights()) {
      auto vi = layer->weights();
      if (vi.size() > 0) { // this probably not needed
        v.insert(v.end(), vi.begin(), vi.end());
      }
    }
  return v;
}
///////////////////////////////////////////////////////////////
std::vector<MatrixFloat *> LayerParallel::gradient_weights() const {
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
void LayerParallel::save(std::ostream &to) const {
  to << "LayerParallel" << std::endl;
  to << reductionToString(_ParallelReduction) << std::endl;
  to << _Layers.size() << std::endl;
  for (auto layer : _Layers) {
    layer->save(to);
  }
}
///////////////////////////////////////////////////////////////
Layer *LayerParallel::load(std::istream &from) {
  std::string sReduction;
  from >> sReduction;
  ParallelReduction reduction = reductionFromString(sReduction);
  size_t numLayers;
  from >> numLayers;
  std::vector<Layer *> layers;
  for (size_t i = 0; i < numLayers; ++i) {
    layers.push_back(Layer::load(from));
  }
  return new LayerParallel(layers, reduction);
}
///////////////////////////////////////////////////////////////
Layer *LayerParallel::construct(std::initializer_list<float> fArgs,
                                std::string sArg) {
  if (fArgs.size() != 0)
    return nullptr;

  // Split into reduction type and layer pointers
  size_t pos = sArg.find(';');
  if (pos == std::string::npos)
    return nullptr;

  std::string reductionStr = sArg.substr(0, pos);
  std::string layerPtrsStr = sArg.substr(pos + 1);

  // Convert reduction string to enum
  ParallelReduction reduction = reductionFromString(reductionStr);

  // Parse layer pointers string - split on commas
  std::vector<Layer *> layers;
  std::string::size_type start = 0;
  std::string::size_type end;

  while ((end = layerPtrsStr.find(',', start)) != std::string::npos) {
    // Convert hex string to pointer
    Layer *layer = (Layer *)std::stoull(layerPtrsStr.substr(start, end - start),
                                        nullptr, 16);
    layers.push_back(layer);
    start = end + 1;
  }
  // Get last layer
  if (start < layerPtrsStr.length()) {
    Layer *layer =
        (Layer *)std::stoull(layerPtrsStr.substr(start), nullptr, 16);
    layers.push_back(layer);
  }

  return new LayerParallel(layers, reduction);
}
///////////////////////////////////////////////////////////////
std::string LayerParallel::constructUsage() {
  return "parallel layer combination\nsReduction;layers\n";
}
///////////////////////////////////////////////////////////////
} // namespace beednn
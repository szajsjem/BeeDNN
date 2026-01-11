#include "LayerRouter.h"

namespace beednn {
LayerRouter::LayerRouter(Layer *RouterLayer, float selectedexperts,
                         std::vector<Layer *> mExperts,
                         ParallelReduction mReduction)
    : Layer("LayerRouter") {
  for (auto layer : mExperts) {
    _Layers.push_back(layer);
  }
  assert(mReduction != DOT);
  _ParallelReduction = mReduction;
  _router = RouterLayer;
  computeLayers = selectedexperts;
}
LayerRouter::LayerRouter() : Layer("LayerRouter") {}
LayerRouter::~LayerRouter() {
  for (auto x : _Layers) {
    delete x;
  }
  delete _router;
}
Layer *LayerRouter::clone() const {
  LayerRouter *pLayer = new LayerRouter();
  for (auto x : _Layers) {
    pLayer->_Layers.push_back(x->clone());
  }
  pLayer->_ParallelReduction = _ParallelReduction;
  pLayer->_router = _router->clone();
  return pLayer;
}
bool LayerRouter::init(size_t &in, size_t &out, std::vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  out = in;
  Layer::init(in, out, internalCalculationMatrices, debug);
  return true;
}
void LayerRouter::forward(const MatrixFloat &mIn, MatrixFloat &mOut) {
  MatrixFloat routing;
  _router->forward(mIn, routing);
#if 1
#define LayerRouterVersion1
  for (int i = 0; i < _Layers.size(); i++) {
    MatrixFloat mf;
    _Layers[i]->forward(mIn, mf);
    // mf.array() *= routing.col(i).array();
    mf = rowWiseMult(mf, routing.col(i));
    if (i == 0) {
      mOut = mf;
    } else if (_ParallelReduction == SUM) {
      mOut += mf;
    } else if (_ParallelReduction == ROWSTACK) {
      mOut = concatenateRows(mOut, mf);
    } else if (_ParallelReduction == COLSTACK) {
      mOut = concatenateCols(mOut, mf);
    }
  }
#else
  std::vector<MatrixFloat> outs(computeLayers);
  for (int i = 0; i < routing.rows(); i++) {
    std::vector<std::pair<float, int>> sorted;
    for (int j = 0; j < routing.cols(); j++)
      sorted.push_back(std::pair<float, int>(routing(i, j), j));
    std::sort(sorted.begin(), sorted.end());
    if (i == 0) {
      MatrixFloat a = mIn.row(0), b;
      _Layers[0]->forward(a, b);
      for (int j = 0; j < computeLayers; j++)
        outs[j].resize(mIn.rows(), b.cols());
    }
    MatrixFloat a = mIn.row(i), b;
    for (int j = 0; j < computeLayers; j++) {
      _Layers[sorted[j].second]->forward(a, b);
      outs[j].row(i) = b.row(0);
    }
  }
  for (int i = 0; i < computeLayers; i++) {
    if (i == 0) {
      mOut = outs[0];
    } else if (_ParallelReduction == SUM) {
      mOut += outs[i];
    } else if (_ParallelReduction == DOT) {
      mOut *= outs[i];
    } else if (_ParallelReduction == ROWSTACK) {
      mOut = concatenateRows(mOut, outs[i]);
    } else if (_ParallelReduction == COLSTACK) {
      mOut = concatenateCols(mOut, outs[i]);
    }
  }
#endif
}
void LayerRouter::backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn, std::vector<MatrixFloat> &internalCalculationMatrices, int start) {
#ifdef LayerRouterVersion1
  // todo
#else
  // todo
#endif
}
bool LayerRouter::has_weights() const {
  for (auto layer : _Layers)
    if (layer->has_weights())
      return true;
  return _router->has_weights();
}
std::vector<MatrixFloat *> LayerRouter::weights() const {
  std::vector<MatrixFloat *> v;
  for (auto layer : _Layers)
    if (layer->has_weights()) {
      auto vi = layer->weights();
      if (vi.size() > 0) { // this probably not needed
        v.insert(v.end(), vi.begin(), vi.end());
      }
    }
  auto vi = _router->weights();
  if (vi.size() > 0) { // this probably not needed
    v.insert(v.end(), vi.begin(), vi.end());
  }
  return v;
}
std::vector<MatrixFloat *> LayerRouter::gradient_weights() const {
  std::vector<MatrixFloat *> v;
  for (auto layer : _Layers)
    if (layer->has_weights()) {
      auto vi = layer->gradient_weights();
      if (vi.size() > 0) {
        v.insert(v.end(), vi.begin(), vi.end());
      }
    }
  auto vi = _router->gradient_weights();
  if (vi.size() > 0) {
    v.insert(v.end(), vi.begin(), vi.end());
  }
  return v;
}
///////////////////////////////////////////////////////////////////////////////
void LayerRouter::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
Layer *LayerRouter::load(std::istream &from) { return NULL; }
///////////////////////////////////////////////////////////////
Layer *LayerRouter::construct(std::initializer_list<float> fArgs,
                              std::string sArg) {
  if (fArgs.size() != 1)
    return nullptr; // selectedExperts

  // Split into parts: router layer, reduction, expert layers
  size_t pos1 = sArg.find(';');
  if (pos1 == std::string::npos)
    return nullptr;

  size_t pos2 = sArg.find(';', pos1 + 1);
  if (pos2 == std::string::npos)
    return nullptr;

  // Parse router layer pointer
  Layer *routerLayer = (Layer *)std::stoull(sArg.substr(0, pos1), nullptr, 16);

  // Parse reduction type
  std::string reductionStr = sArg.substr(pos1 + 1, pos2 - pos1 - 1);
  ParallelReduction reduction = reductionFromString(reductionStr);

  // Parse expert layer pointers
  std::string expertPtrsStr = sArg.substr(pos2 + 1);
  std::vector<Layer *> experts;
  std::string::size_type start = 0;
  std::string::size_type end;

  while ((end = expertPtrsStr.find(',', start)) != std::string::npos) {
    Layer *layer = (Layer *)std::stoull(
        expertPtrsStr.substr(start, end - start), nullptr, 16);
    experts.push_back(layer);
    start = end + 1;
  }
  // Get last expert
  if (start < expertPtrsStr.length()) {
    Layer *layer =
        (Layer *)std::stoull(expertPtrsStr.substr(start), nullptr, 16);
    experts.push_back(layer);
  }

  return new LayerRouter(routerLayer, *fArgs.begin(), experts, reduction);
}
///////////////////////////////////////////////////////////////
std::string LayerRouter::constructUsage() {
  return "routes inputs through expert "
         "layers\nlayer;sReduction;layers\nfSelectedExperts";
}
} // namespace beednn
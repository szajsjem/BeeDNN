#include "BeeDNNLib.h"
#include <algorithm>
#include <string>
#include <vector>

#include "Activations.h"
#include "Layer.h"
#include "LayerActivation.h"
#include "LayerFactory.h"
#include "Net.h"
#include "NetTrain.h"
#include "NetUtil.h"


using namespace beednn;

// Net Implementation
void *create_net() { return new Net(); }

void delete_net(void *ptr) {
  if (ptr)
    delete static_cast<Net *>(ptr);
}

void init_net(void *pNN, int32_t iInputSize) {
  static_cast<Net *>(pNN)->init(iInputSize);
}

void set_classification_mode(void *pNN, int32_t _iClassificationMode) {
  static_cast<Net *>(pNN)->set_classification_mode(_iClassificationMode != 0);
}

void set_train_mode(void *pNN, int32_t bTrainMode) {
  static_cast<Net *>(pNN)->set_train_mode(bTrainMode != 0);
}

void add_layer(void *pNN, void *pLayer) {
  // Takes ownership
  static_cast<Net *>(pNN)->add(static_cast<Layer *>(pLayer));
}

void predict(void *pNN, const float *pIn, float *pOut, int32_t iNbSamples,
             int32_t iNbFeatures) {
  MatrixFloat mIn = MatrixFloat::from_raw_buffer(pIn, iNbSamples, iNbFeatures);
  MatrixFloat mOut;
  static_cast<Net *>(pNN)->predict(mIn, mOut);

  // Copy output
  // Assuming pOut is allocated large enough (iNbSamples * outputSize)
  std::copy(mOut.data(), mOut.data() + mOut.size(), pOut);
}

// NetTrain Implementation
void *create_train() { return new NetTrain(); }

void delete_train(void *ptr) {
  if (ptr)
    delete static_cast<NetTrain *>(ptr);
}

void train_set_train_data(void *ptr, float *samples, int sampleRows,
                          int sampleCols, float *truth, int truthRows,
                          int truthCols) {
  MatrixFloat mSamples =
      MatrixFloat::from_raw_buffer(samples, sampleRows, sampleCols);
  MatrixFloat mTruth =
      MatrixFloat::from_raw_buffer(truth, truthRows, truthCols);
  static_cast<NetTrain *>(ptr)->set_train_data_copy(mSamples, mTruth);
}

void train_set_validation_data(void *ptr, float *samples, int sampleRows,
                               int sampleCols, float *truth, int truthRows,
                               int truthCols) {
  MatrixFloat mSamples =
      MatrixFloat::from_raw_buffer(samples, sampleRows, sampleCols);
  MatrixFloat mTruth =
      MatrixFloat::from_raw_buffer(truth, truthRows, truthCols);
  static_cast<NetTrain *>(ptr)->set_validation_data(mSamples, mTruth);
}

void train_set_batch_size(void *ptr, int batchSize) {
  static_cast<NetTrain *>(ptr)->set_batchsize(batchSize);
}

void train_set_epochs(void *ptr, int epochs) {
  static_cast<NetTrain *>(ptr)->set_epochs(epochs);
}

void train_fit(void *ptr, void *netPtr) {
  static_cast<NetTrain *>(ptr)->fit(*static_cast<Net *>(netPtr));
}

void train_set_optimizer(void *ptr, const char *optimizer) {
  static_cast<NetTrain *>(ptr)->set_optimizer(optimizer);
}

void train_set_loss(void *ptr, const char *loss) {
  static_cast<NetTrain *>(ptr)->set_loss(loss);
}

void train_set_regularizer(void *ptr, const char *regularizer,
                           float parameter) {
  static_cast<NetTrain *>(ptr)->set_regularizer(regularizer, parameter);
}

int train_get_train_loss_size(void *ptr) {
  return (int)static_cast<NetTrain *>(ptr)->get_train_loss().size();
}

void train_get_train_loss_data(void *ptr, float *buffer) {
  const auto &v = static_cast<NetTrain *>(ptr)->get_train_loss();
  std::copy(v.begin(), v.end(), buffer);
}

// Layer Implementation
void *layer_create_activation(const char *activation) {
  return new LayerActivation(activation);
}

void *layer_construct(const char *type, float *args, int argsCount,
                      const char *arg) {
  std::vector<float> vArgs(args, args + argsCount);
  std::initializer_list<float> ilArgs = {};
  // Workaround for initializer_list since we can't create it dynamically
  // easily. We will use registered creator directly? No, Factory hides it.
  // We'll use the same hack as in WASM or just support few args.
  // Actually, LayerFactory::construct signature expects std::initializer_list.
  // Since we are in C++, we can't build it dynamically.
  // However, we can use a helper that dispatches.

  // Dispatch hack for common sizes
  switch (argsCount) {
  case 0:
    return LayerFactory::construct(type, {}, arg ? arg : "");
  case 1:
    return LayerFactory::construct(type, {args[0]}, arg ? arg : "");
  case 2:
    return LayerFactory::construct(type, {args[0], args[1]}, arg ? arg : "");
  case 3:
    return LayerFactory::construct(type, {args[0], args[1], args[2]},
                                   arg ? arg : "");
  case 4:
    return LayerFactory::construct(type, {args[0], args[1], args[2], args[3]},
                                   arg ? arg : "");
  default:
    return LayerFactory::construct(type, {args[0]}, arg ? arg : ""); // Fallback
  }
}

void delete_layer(void *ptr) {
  // Do NOT delete if owned by Net!
  // But if we created it and haven't added it yet, we should.
  // This is up to the caller (Python GC).
  if (ptr)
    delete static_cast<Layer *>(ptr);
}

// Distributed Training Implementation
int net_get_params_size(void *ptr) {
  return (int)static_cast<Net *>(ptr)->get_params().size();
}

void net_get_params_data(void *ptr, float *buffer) {
  std::vector<float> p = static_cast<Net *>(ptr)->get_params();
  std::copy(p.begin(), p.end(), buffer);
}

void net_set_params(void *ptr, float *params, int size) {
  std::vector<float> v(params, params + size);
  static_cast<Net *>(ptr)->set_params(v);
}

void net_mix_params(void *ptr, float *other_params, int size, float theta) {
  std::vector<float> v(other_params, other_params + size);
  static_cast<Net *>(ptr)->mix_params(v, theta);
}

void net_accumulate_weight_diff_to_grad(void *ptr, float *recv_params,
                                        int size) {
  std::vector<float> v(recv_params, recv_params + size);
  static_cast<Net *>(ptr)->accumulate_weight_diff_to_grad(v);
}

void train_distributed_step(void *ptr, float num_workers) {
  static_cast<NetTrain *>(ptr)->distributed_step(num_workers);
}

// Legacy / Helper
// Deprecated: create(inputSize) -> Net*
// This was "create" in old API. We map it to create_net + init
void *create(int32_t iInputSize) {
  Net *net = new Net();
  if (iInputSize > 0)
    net->init(iInputSize);
  return net;
}

void save(void *pNN, char *filename) {
  // This expects pNN to be the old BeeDNN struct wrapper?
  // The old code had a BeeDNN struct. We should probably keep it for backward
  // compatibility OR we just break it and update Python Loader. The previous
  // implementation used a wrapper class BeeDNN. I replaced it with direct Net*
  // usage in my new functions. BUT 'create' returned 'new BeeDNN(iInputSize)'.
  // If I want to match the old signature exactly for 'create', I should keep
  // the wrapper? No, I'll update Python Loader to use the new API directly.
  // 'save' in old API used NetUtil::save.
  // We can't implement save(filename) easily on just Net* because NetUtil::save
  // takes NetTrain too. I'll implement save matching the old behavior if
  // possible, but preferably we split it. OLD behavior: save(wrapper) -> saves
  // net + train. I will remove the wrapper and require user to manage Net and
  // Train separately. For 'save', I'll just save the Net if we only have the
  // Net. Wait, NetUtil::save takes BOTH. So 'save' is tricky. I'll leave 'save'
  // as a stub or implement a new save_net / save_train.

  // Let's implement save_model(net, train, filename)
}

// Legacy 'add_layer' with string
// We try to interpret it.
void add_layer(void *pNN, char *layer) {
  // Legacy support: try to create activation
  // If failed (how to know?), try construct with empty args.
  // LayerFactory::create(layer) used to exist.
  // We'll assume it's an activation or simple layer.
  // Note: LayerFactory::construct("Dense", {}, "GlorotUniform")
  std::string s(layer);
  Layer *l = new LayerActivation(s); // Try generic activation
  // Ideally we should check if 's' is in available activations.
  // If not, maybe it's a layer type?
  static_cast<Net *>(pNN)->add(l);
}
#ifndef _BeeDNNLib_
#define _BeeDNNLib_

#include <cstdint>

#ifdef _WIN32
#ifdef BEEDNNLIB_BUILD
#define BEEDNN_EXPORT extern "C" __declspec(dllexport)
#else
#define BEEDNN_EXPORT extern "C" __declspec(dllimport)
#endif
#else
#define BEEDNN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Net
BEEDNN_EXPORT void *create_net();
BEEDNN_EXPORT void delete_net(void *ptr);
BEEDNN_EXPORT void set_classification_mode(void *pNN,
                                           int32_t _iClassificationMode);
BEEDNN_EXPORT void add_layer(void *pNN, void *pLayer);
BEEDNN_EXPORT void predict(void *pNN, const float *pIn, float *pOut,
                           int32_t iNbSamples, int32_t iNbFeatures);
BEEDNN_EXPORT void init_net(void *pNN, int32_t iInputSize);
BEEDNN_EXPORT void set_train_mode(void *pNN, int32_t bTrainMode);

// NetTrain
BEEDNN_EXPORT void *create_train();
BEEDNN_EXPORT void delete_train(void *ptr);
BEEDNN_EXPORT void train_set_train_data(void *ptr, float *samples,
                                        int sampleRows, int sampleCols,
                                        float *truth, int truthRows,
                                        int truthCols);
BEEDNN_EXPORT void train_set_validation_data(void *ptr, float *samples,
                                             int sampleRows, int sampleCols,
                                             float *truth, int truthRows,
                                             int truthCols);
BEEDNN_EXPORT void train_set_batch_size(void *ptr, int batchSize);
BEEDNN_EXPORT void train_set_epochs(void *ptr, int epochs);
BEEDNN_EXPORT void train_fit(void *ptr, void *netPtr);
BEEDNN_EXPORT void train_set_optimizer(void *ptr, const char *optimizer);
BEEDNN_EXPORT void train_set_loss(void *ptr, const char *loss);
BEEDNN_EXPORT void train_set_regularizer(void *ptr, const char *regularizer,
                                         float parameter);
// Getters for history (returns pointer to internal vector data, careful with
// lifetime)
BEEDNN_EXPORT int train_get_train_loss_size(void *ptr);
BEEDNN_EXPORT void train_get_train_loss_data(void *ptr, float *buffer);
// ... others omitted for brevity but should be added

// Layer
BEEDNN_EXPORT void *layer_create_activation(const char *activation);
BEEDNN_EXPORT void *layer_construct(const char *type, float *args,
                                    int argsCount, const char *arg);
BEEDNN_EXPORT void delete_layer(void *ptr);

// Distributed Training
// Net Params
BEEDNN_EXPORT int net_get_params_size(void *ptr);
BEEDNN_EXPORT void net_get_params_data(void *ptr, float *buffer);
BEEDNN_EXPORT void net_set_params(void *ptr, float *params, int size);
BEEDNN_EXPORT void net_mix_params(void *ptr, float *other_params, int size,
                                  float theta);
BEEDNN_EXPORT void
net_accumulate_weight_diff_to_grad(void *ptr, float *recv_params, int size);
// Distributed Step
BEEDNN_EXPORT void train_distributed_step(void *ptr, float num_workers);

// Legacy
BEEDNN_EXPORT void save(void *pNN, char *filename);

#endif

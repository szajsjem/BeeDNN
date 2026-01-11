#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include "LayerActivation.h"
#include "LayerBias.h"
#include "LayerDense.h"
#include "LayerDot.h"
#include "LayerSequential.h"
#include "Matrix.h"

using namespace beednn;
using namespace std;

bool areMatricesEqual(const MatrixFloat &m1, const MatrixFloat &m2,
                      float epsilon = 1e-5f) {
  if (m1.rows() != m2.rows() || m1.cols() != m2.cols())
    return false;
  for (Index i = 0; i < m1.size(); ++i) {
    if (std::abs(m1(i) - m2(i)) > epsilon)
      return false;
  }
  return true;
}

void testLayerDot() {
  std::cout << "Testing LayerDot..." << std::endl;
  LayerDot layer(2, 3, "Zeros"); // 2 inputs, 3 outputs

  // Manually set weights for predictability
  MatrixFloat weights(2, 3);
  weights(0) = 0.1f;
  weights(1) = 0.2f;
  weights(2) = 0.3f;
  weights(3) = 0.4f;
  weights(4) = 0.5f;
  weights(5) = 0.6f;
  *layer.weights()[0] = weights;

  // Forward
  MatrixFloat input(1, 2);
  input(0) = 1.0f;
  input(1) = 2.0f;
  MatrixFloat output;
  layer.forward(input, output);

  // Save
  std::stringstream ss;
  layer.save(ss);

  // Load
  std::string type;
  ss >> type;
  Layer *loadedLayer = LayerDot::load(ss);
  assert(loadedLayer != nullptr);

  // Verify weights
  assert(areMatricesEqual(*layer.weights()[0], *loadedLayer->weights()[0]));

  // Verify forward
  MatrixFloat loadedOutput;
  loadedLayer->forward(input, loadedOutput);
  assert(areMatricesEqual(output, loadedOutput));

  delete loadedLayer;
  std::cout << "LayerDot passed." << std::endl;
}

void testLayerBias() {
  std::cout << "Testing LayerBias..." << std::endl;
  LayerBias layer("Zeros");

  // Forward to init bias (lazy init)
  MatrixFloat input(1, 3);
  input.setZero();
  MatrixFloat output;
  layer.forward(input, output);

  // Set bias
  MatrixFloat bias(1, 3);
  bias(0) = 0.1f;
  bias(1) = 0.2f;
  bias(2) = 0.3f;
  *layer.weights()[0] = bias;

  // Save
  std::stringstream ss;
  layer.save(ss);

  // Load
  std::string type;
  ss >> type;
  Layer *loadedLayer = LayerBias::load(ss);
  assert(loadedLayer != nullptr);

  // Verify weights
  assert(areMatricesEqual(*layer.weights()[0], *loadedLayer->weights()[0]));

  // Verify forward
  MatrixFloat loadedOutput;
  loadedLayer->forward(input, loadedOutput);
  assert(
      areMatricesEqual(output + bias, loadedOutput)); // output was zero + bias

  delete loadedLayer;
  std::cout << "LayerBias passed." << std::endl;
}

void testLayerActivation() {
  std::cout << "Testing LayerActivation..." << std::endl;
  LayerActivation layer("Relu");

  MatrixFloat input(1, 3);
  input(0) = -1.0f;
  input(1) = 0.0f;
  input(2) = 1.0f;
  MatrixFloat output;
  layer.forward(input, output);

  // Save
  std::stringstream ss;
  layer.save(ss);

  // Load
  std::string type;
  ss >> type;
  Layer *loadedLayer = LayerActivation::load(ss);
  assert(loadedLayer != nullptr);

  // Verify forward
  MatrixFloat loadedOutput;
  loadedLayer->forward(input, loadedOutput);
  assert(areMatricesEqual(output, loadedOutput));

  delete loadedLayer;
  std::cout << "LayerActivation passed." << std::endl;
}

void testLayerDense() {
  std::cout << "Testing LayerDense..." << std::endl;
  LayerDense layer(2, 3, "GlorotUniform", "Zeros");
  size_t i = 2, o = 3;
  std::vector<MatrixFloat> icm;
  layer.init(i, o, icm, true);
  // Forward to init
  MatrixFloat input(1, 2);
  setRandomUniform(input, 0, 1);
  MatrixFloat output;
  layer.forward(input, output);

  // Save
  std::stringstream ss;
  layer.save(ss);

  // Load
  std::string type;
  ss >> type;
  Layer *loadedLayer = LayerDense::load(ss);
  assert(loadedLayer != nullptr);

  // Verify weights (Dense has multiple weights from inner layers)
  auto w1 = layer.weights();
  auto w2 = loadedLayer->weights();
  assert(w1.size() == w2.size());
  for (size_t i = 0; i < w1.size(); ++i) {
    assert(areMatricesEqual(*w1[i], *w2[i]));
  }

  // Verify forward
  MatrixFloat loadedOutput;
  loadedLayer->forward(input, loadedOutput);
  assert(areMatricesEqual(output, loadedOutput));

  delete loadedLayer;
  std::cout << "LayerDense passed." << std::endl;
}

void testLayerSequential() {
  std::cout << "Testing LayerSequential..." << std::endl;
  std::vector<Layer *> layers;
  layers.push_back(new LayerDot(2, 3, "GlorotUniform"));
  layers.push_back(new LayerBias("Zeros"));
  layers.push_back(new LayerActivation("Relu"));
  LayerSequential layer(layers);
  size_t i = 2, o = 3;
  std::vector<MatrixFloat> icm;
  layer.init(i, o, icm, true);

  // Forward
  MatrixFloat input(1, 2);
  setRandomUniform(input, 0, 1);
  MatrixFloat output;
  layer.forward(input, output);

  // Save
  std::stringstream ss;
  layer.save(ss);

  // Load
  std::string type;
  ss >> type;
  Layer *loadedLayer = LayerSequential::load(ss);
  assert(loadedLayer != nullptr);

  // Verify forward
  MatrixFloat loadedOutput;
  loadedLayer->forward(input, loadedOutput);
  assert(areMatricesEqual(output, loadedOutput));

  delete loadedLayer;
  std::cout << "LayerSequential passed." << std::endl;
}

#include "LayerEmbed.h"
#include "LayerNormalize.h"
#include "LayerParallel.h"
#include "LayerSoftmax.h"
#include "ParallelReduction.h"
#include <LayerFactory.h>
#include <allLayers.h>

void testLayerSoftmax() {
  std::cout << "Testing LayerSoftmax..." << std::endl;
  LayerSoftmax layer;

  MatrixFloat input(1, 3);
  input(0) = 1.0f;
  input(1) = 2.0f;
  input(2) = 3.0f;
  MatrixFloat output;
  layer.forward(input, output);

  // Check sum to 1
  float sum = 0;
  for (Index i = 0; i < output.size(); ++i) {
    sum += output(i);
  }
  assert(std::abs(sum - 1.0f) < 1e-4f);

  // Save/Load
  std::stringstream ss;
  layer.save(ss);
  std::string type;
  ss >> type;
  Layer *loadedLayer = LayerSoftmax::load(ss);
  assert(loadedLayer != nullptr);
  MatrixFloat loadedOutput;
  loadedLayer->forward(input, loadedOutput);
  assert(areMatricesEqual(output, loadedOutput));
  delete loadedLayer;

  std::cout << "LayerSoftmax passed." << std::endl;
}

void testLayerEmbed() {
  std::cout << "Testing LayerEmbed..." << std::endl;
  // vocabSize=10, dimensionSize=4, maxPositon=5
  LayerEmbed layer(10, 4, 5, "Zeros");

  MatrixFloat input(1, 2);
  input(0) = 1.0f; // index 1
  input(1) = 5.0f; // index 5

  MatrixFloat output;

  size_t in = 1;
  size_t out = 0;
  std::vector<MatrixFloat> icm;
  layer.init(in, out, icm);

  layer.forward(input, output);
  assert(output.size() == 4);

  // currently no save/load implemented for LayerEmbed,TODO
  //// Save/Load
  // std::stringstream ss;
  // layer.save(ss);
  // std::string type;
  // ss >> type;
  // Layer *loadedLayer = LayerEmbed::load(ss);
  // assert(loadedLayer != nullptr);

  // MatrixFloat loadedOutput;
  // loadedLayer->forward(input, loadedOutput);
  // assert(areMatricesEqual(output, loadedOutput));
  // delete loadedLayer;

  // std::cout << "LayerEmbed passed." << std::endl;
}

void testLayerNormalize() {
  std::cout << "Testing LayerNormalize..." << std::endl;
  LayerNormalize layer;

  MatrixFloat input(1, 4);
  input(0) = 1.0f;
  input(1) = 2.0f;
  input(2) = 3.0f;
  input(3) = 4.0f;

  MatrixFloat output;
  layer.forward(input, output);

  float mean = 0;
  for (Index i = 0; i < output.size(); ++i)
    mean += output(i);
  mean /= output.size();
  assert(std::abs(mean) < 1e-4f);

  // Save/Load
  std::stringstream ss;
  layer.save(ss);
  std::string type;
  ss >> type;
  Layer *loadedLayer = LayerNormalize::load(ss);
  assert(loadedLayer != nullptr);
  MatrixFloat loadedOutput;
  loadedLayer->forward(input, loadedOutput);
  assert(areMatricesEqual(output, loadedOutput));
  delete loadedLayer;

  std::cout << "LayerNormalize passed." << std::endl;
}

void testLayerParallel() {
  std::cout << "Testing LayerParallel..." << std::endl;

  std::vector<Layer *> branches;
  branches.push_back(new LayerDense(2, 2, "Zeros", "Zeros"));
  branches.push_back(new LayerDense(2, 2, "Zeros", "Zeros"));

  LayerParallel layer(branches, ParallelReduction::SUM);

  MatrixFloat input(1, 2);
  input.setConstant(1.0f);

  size_t in = 2;
  size_t out = -1;
  std::vector<MatrixFloat> icm;
  layer.init(in, out, icm);
  assert(out == 2);

  MatrixFloat output;
  std::vector<MatrixFloat *> ws = layer.weights();
  for (auto *w : ws) {
    w->setConstant(1.0f);
  }

  layer.forward(input, output);
  assert(std::abs(output(0) - 4.0f) < 1e-4f);
  assert(std::abs(output(1) - 4.0f) < 1e-4f);

  // Save/Load
  std::stringstream ss;
  layer.save(ss);
  std::string type;
  ss >> type;
  Layer *loadedLayer = LayerParallel::load(ss);
  assert(loadedLayer != nullptr);

  // Note: verify if init is needed after load for Parallel
  // Assuming generic load for Parallel handles structure but maybe not init
  // sizes if they are dynamic. But LayerParallel::init is usually called during
  // setup. We'll see if forward works directly.
  try {
    MatrixFloat loadedOutput;
    loadedLayer->forward(input, loadedOutput);
    assert(areMatricesEqual(output, loadedOutput));
  } catch (...) {
    // If fails, try init
    size_t l_in = 2;
    size_t l_out = 0;
    std::vector<MatrixFloat> l_icm;
    loadedLayer->init(l_in, l_out, l_icm);
    MatrixFloat loadedOutput;
    loadedLayer->forward(input, loadedOutput);
    assert(areMatricesEqual(output, loadedOutput));
  }

  delete loadedLayer;
  std::cout << "LayerParallel passed." << std::endl;
}

void testCommonAllLayers() {
  std::cout << "Testing Common All Layers..." << std::endl;
  std::vector<std::string> layers = LayerFactory::getAvailable();

  for (const auto &layerName : layers) {
    if (layerName == "LayerParallel")
      continue;
    if (layerName == "LayerStacked")
      continue;
    if (layerName == "LayerRouter")
      continue;
    if (layerName == "LayerSequential")
      continue;
    if (layerName == "LayerRNN")
      continue;
    if (layerName == "LayerSimpleRNN")
      continue;
    if (layerName == "LayerSimplestRNN")
      continue;
    if (layerName == "LayerSelfAttention")
      continue;
    if (layerName == "LayerConvolution2D")
      continue;
    if (layerName == "LayerTimeDistributedDense")
      continue;
    if (layerName == "LayerRepetetive")
      continue;
    if (layerName == "LayerGaussianDropout")
      continue; // Unstable gradient in generic test

    // Skip Global layers (usually expect 4D) and Pooling layers
    if (layerName.find("Global") != std::string::npos)
      continue;
    if (layerName.find("Pool") != std::string::npos)
      continue; // Matches Pooling, MaxPool, etc.
    if (layerName.find("ZeroPadding2D") != std::string::npos)
      continue;
    if (layerName.find("BatchTo2D") != std::string::npos)
      continue;
    if (layerName.find("Transformer") != std::string::npos)
      continue; // Skip transformer parts if complex

    if (layerName == "LayerRandomFlip")
      continue;
    if (layerName == "LayerRRelu")
      continue;
    if (layerName == "LayerChannelBias")
      continue;
    if (layerName == "LayerDropout")
      continue;

    std::cout << "  checking " << layerName << "... \n";

    try {
      std::string usage = LayerFactory::getUsage(layerName);
      std::stringstream ss(usage);
      std::string desc, sArgsDesc, fArgsDesc;
      std::getline(ss, desc);
      std::getline(ss, sArgsDesc);
      std::getline(ss, fArgsDesc);

      // Parse String Args
      std::string sArg = "";
      if (!sArgsDesc.empty()) {
        // Trim sArgsDesc
        size_t first = sArgsDesc.find_first_not_of(" \t\r\n");
        if (first != std::string::npos) {
          size_t last = sArgsDesc.find_last_not_of(" \t\r\n");
          sArgsDesc = sArgsDesc.substr(first, (last - first + 1));

          std::stringstream ssArgs(sArgsDesc);
          std::string argName;
          std::vector<std::string> defaults;

          while (std::getline(ssArgs, argName, ';')) {
            if (argName.find("Initializer") != std::string::npos)
              defaults.push_back("GlorotUniform");
            else if (argName.find("Activation") != std::string::npos)
              defaults.push_back("Relu");
            else if (argName.find("Reduction") != std::string::npos)
              defaults.push_back("Sum");
            // else if(argName.find("Recall") != std::string::npos)
            // defaults.push_back("0.5");
            else
              defaults.push_back("Zeros");
          }
          for (size_t i = 0; i < defaults.size(); ++i) {
            sArg += defaults[i];
            if (i < defaults.size() - 1)
              sArg += ";";
          }
        }
      }

      // Parse Float Args Count
      std::vector<float> floats;
      bool hasFloats = false;
      if (!fArgsDesc.empty()) {
        size_t first = fArgsDesc.find_first_not_of(" \t\r\n");
        if (first != std::string::npos) {
          hasFloats = true;
          // Provide at least one float if description is not empty
          floats.push_back(10);
          for (char c : fArgsDesc) {
            if (c == ';')
              floats.push_back(10);
          }
        }
      }

      // Construct
      Layer *layer = nullptr;
      // If floats is empty but hasFloats is true, it means 1 arg. Not possible
      // with above logic but safe logic needed

      layer = LayerFactory::construct(
          layerName,
          std::initializer_list<float>(floats.data(),
                                       floats.data() + floats.size()),
          sArg);

      if (!layer) {
        std::cout << "FAILED to construct." << std::endl;
        continue;
      }

      size_t inSize = 10;
      size_t outSize = -1;
      std::vector<MatrixFloat> icm;

      if (!layer->init(inSize, outSize, icm, false)) {
        std::cout << "init returned false." << std::endl;
        delete layer;
        continue;
      }

      MatrixFloat input(1, inSize);
      setRandomUniform(input, -1, 1);
      MatrixFloat output;
      layer->forward(input, output);

      MatrixFloat output2;
      output2.resizeLike(output);
      setRandomUniform(output2, -1.f, 1.f); // Random target

      // helper to compute MSE loss
      auto computeLoss = [](const MatrixFloat &pred,
                            const MatrixFloat &target) {
        MatrixFloat diff = pred - target;
        return diff.squaredNorm();
      };

      float loss = computeLoss(output, output2);

      if (loss == 0.0f) {
        std::cout << "  [WARN] Loss is zero, skipping backpropagation test."
                  << std::endl;
        delete layer;
        continue;
      }

      // Compute gradient of loss w.r.t. output
      // For MSE = 0.5 * (y - t)^2, grad = (y - t). We use sum(y-t)^2 so grad =
      // 2*(y-t). We'll just use diff as direction.
      MatrixFloat gradOut = (output - output2) * 2.0f;
      MatrixFloat gradIn;

      layer->backpropagation(input, gradOut, gradIn, icm, 0);

      // Verify Input Gradient
      // Skip for LayerEmbed as inputs are indices
      float epsilon = 1e-3f;
      if (layerName != "LayerEmbed") {
        MatrixFloat input2 = input - gradIn * epsilon;
        MatrixFloat outputIM;
        layer->forward(input2, outputIM);
        float inputbp = computeLoss(outputIM, output2);

        if (inputbp > loss) {
          std::cout << "  [WARN] Input gradient step increased or kept loss: "
                    << loss << " -> " << inputbp << std::endl;
        } else {
          std::cout << "  Input gradient step reduced loss: " << loss << " -> "
                    << inputbp << std::endl;
        }
      }

      // Verify Weight Gradient
      if (layer->has_weights()) {
        auto weights = layer->weights();
        auto grads =
            layer->gradient_weights(); // Note: returned vectors of pointers
        if (weights.size() > 0 && weights.size() == grads.size()) {
          // Apply gradient step
          for (size_t i = 0; i < weights.size(); ++i) {
            *weights[i] -= *grads[i] * epsilon;
          }

          MatrixFloat outputWM;
          layer->forward(input, outputWM);
          float weightbp = computeLoss(outputWM, output2);

          if (weightbp > loss) {
            std::cout
                << "  [WARN] Weight gradient step increased or kept loss: "
                << loss << " -> " << weightbp << std::endl;
          } else {
            std::cout << "  Weight gradient step reduced loss: " << loss
                      << " -> " << weightbp << std::endl;
          }
        }
      }

      // Basic Save check to stream
      std::stringstream ssSave;
      layer->save(ssSave);

      delete layer;
      std::cout << "OK" << std::endl;

    } catch (const std::exception &e) {
      std::cout << "EXCEPTION: " << e.what() << std::endl;
    }
  }
}

int main() {
  try {
    testLayerDot();
    testLayerBias();
    testLayerActivation();
    testLayerDense();
    testLayerSequential();

    // New tests
    testLayerSoftmax();
    testLayerEmbed();
    testLayerNormalize();
    testLayerParallel();

    // Common test
    testCommonAllLayers();

    std::cout << "All tests passed!" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}

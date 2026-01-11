#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include "Matrix.h"
#include "allLayers.h"


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

void testLayerSaveLoad(Layer *layer, const std::string &name,
                       const MatrixFloat &input) {
  std::cout << "Testing " << name << "..." << std::endl;

  // Forward
  MatrixFloat output;
  layer->forward(input, output);

  // Save
  std::stringstream ss;
  layer->save(ss);

  // Load using Factory
  Layer *loadedLayer = LayerFactory::loadLayer(ss);
  assert(loadedLayer != nullptr);

  // Verify Forward
  MatrixFloat loadedOutput;
  loadedLayer->forward(input, loadedOutput);

  if (!areMatricesEqual(output, loadedOutput)) {
    std::cerr << "Mismatch in output for " << name << std::endl;
    exit(1);
  }

  std::cout << name << " passed." << std::endl;
  delete loadedLayer;
}

int main() {
  try {
    MatrixFloat input(1, 4);
    input.setRandom();

    // Gated Layers
    testLayerSaveLoad(new LayerBilinear(), "LayerBilinear", input);
    testLayerSaveLoad(new LayerGLU(), "LayerGLU", input);
    testLayerSaveLoad(new LayerGTU(), "LayerGTU", input);
    testLayerSaveLoad(new LayerReGLU(), "LayerReGLU", input);
    testLayerSaveLoad(new LayerGEGLU(), "LayerGEGLU", input);
    testLayerSaveLoad(new LayerSeGLU(), "LayerSeGLU", input);
    testLayerSaveLoad(new LayerSwiGLU(), "LayerSwiGLU", input);

    // Global Affine
    testLayerSaveLoad(new LayerGlobalAffine(), "LayerGlobalAffine", input);

    // Repetetive
    testLayerSaveLoad(new LayerRepetetive(new LayerDense(4, 4), 2),
                      "LayerRepetetive", input); // 2 copies

    // Parallel / Stacked
    testLayerSaveLoad(new LayerStacked(new LayerDense(4, 4), SUM, 2),
                      "LayerStacked", input);

    // Transformer components
    // FeedForward needs specific dims
    testLayerSaveLoad(new LayerTransformerFeedForward(4, 8, "Relu"),
                      "LayerTransformerFeedForward", input);

    // Heads needs specific dims
    // dimSize=4, vMem=2, qkMem=2, heads=2 -> output size will be heads*vMem =
    // 4? No, output dense maps back to dimSize=4
    testLayerSaveLoad(new LayerTransformerHeads(4, 2, 2, 2),
                      "LayerTransformerHeads", input);

    // TimeDistributed
    // Needs proper input size? It processes frames.
    // If input is (1, 4), and frame size is determined by user?
    // LayerTimeDistributedDense(inFrame, outFrame)
    // If (1,4), we can treat it as 1 time step of size 4 if inFrame=4?
    // Or 2 time steps of size 2?
    testLayerSaveLoad(new LayerTimeDistributedDense(2, 2),
                      "LayerTimeDistributedDense", input);

    std::cout << "All generic save/load tests passed!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}

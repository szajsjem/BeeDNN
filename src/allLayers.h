#pragma once

// Base Layer and Factory
#include "Layer.h"
#include "LayerFactory.h"

// Basic Neural Network Layers
#include "LayerDot.h"
#include "LayerDense.h"
#include "LayerBias.h"
#include "LayerActivation.h"
#include "LayerDropout.h"
#include "LayerGain.h"
#include "LayerAffine.h"

// Composite Layers
#include "LayerSequential.h"
#include "LayerParallel.h"
#include "LayerRouter.h"
#include "LayerStacked.h"

// Convolutional and Pooling Layers
#include "LayerConvolution2D.h"
#include "LayerMaxPool2D.h"
#include "LayerAveragePooling2D.h"
#include "LayerGlobalMaxPool2D.h"
#include "LayerGlobalAveragePooling2D.h"
#include "LayerZeroPadding2D.h"
#include "LayerBatchTo2D.h"

// Normalization and Regularization Layers
#include "LayerNormalize.h"
#include "LayerGaussianNoise.h"
#include "LayerGaussianDropout.h"
#include "LayerUniformNoise.h"
#include "LayerRandomFlip.h"

// Advanced Activation Layers
#include "LayerPRelu.h"
#include "LayerRRelu.h"
#include "LayerPELU.h"
#include "LayerCRelu.h"
#include "LayerTERELU.h"

// Gated Activation Layers
#include "LayerGatedActivation.h"
#include "LayerGLU.h"
#include "LayerGTU.h"
#include "LayerReGLU.h"
#include "LayerGEGLU.h"
#include "LayerSeGLU.h"
#include "LayerSwiGLU.h"
#include "LayerBilinear.h"

// Specialized Layers
#include "LayerSoftmax.h"
#include "LayerSoftmin.h"
#include "LayerEmbed.h"

// Global Layers
#include "LayerGlobalBias.h"
#include "LayerGlobalGain.h"
#include "LayerGlobalAffine.h"
#include "LayerChannelBias.h"

// RNN and Attention Layers
#include "LayerRNN.h"
#include "LayerSimpleRNN.h"
#include "LayerSimplestRNN.h"
#include "LayerSelfDot.h"
#include "LayerSelfAttention.h"

// Time Distributed Layers
#include "LayerTimeDistributedDot.h"
#include "LayerTimeDistributedBias.h"
#include "LayerTimeDistributedDense.h"

// Transformer Layers
#include "LayerTranspose.h"
#include "LayerTransformerHeads.h"
#include "LayerTransformerFeedForward.h"
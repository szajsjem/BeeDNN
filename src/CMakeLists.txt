set(BEEDNN_FILES
	Activations.cpp Activations.h
	CIFAR10Reader.cpp CIFAR10Reader.h
	Metrics.cpp Metrics.h
	CsvFileReader.cpp CsvFileReader.h
	DataSource.cpp DataSource.h
	Initializers.cpp Initializers.h
	JsonFile.cpp JsonFile.h
	KMeans.cpp KMeans.h
	KMeansTrain.cpp KMeansTrain.h
	Layer.cpp Layer.h
	LayerActivation.cpp LayerActivation.h
	LayerAffine.cpp LayerAffine.h
	LayerAveragePooling2D.cpp LayerAveragePooling2D.h
	LayerBias.cpp LayerBias.h
	LayerBilinear.cpp LayerBilinear.h
	LayerChannelBias.cpp LayerChannelBias.h
	LayerConvolution2D.cpp LayerConvolution2D.h
	LayerCRelu.cpp LayerCRelu.h
	LayerDense.cpp LayerDense.h
	LayerDot.cpp LayerDot.h
	LayerDropout.cpp LayerDropout.h
	LayerFactory.cpp LayerFactory.h
	LayerGain.cpp LayerGain.h
	LayerGaussianDropout.cpp LayerGaussianDropout.h
	LayerGaussianNoise.cpp LayerGaussianNoise.h
	LayerGatedActivation.cpp LayerGatedActivation.h
	LayerGEGLU.cpp LayerGEGLU.h
	LayerGlobalAffine.cpp LayerGlobalAffine.h
	LayerGlobalAveragePooling2D.cpp LayerGlobalAveragePooling2D.h
	LayerGlobalBias.cpp LayerGlobalBias.h
	LayerGlobalGain.cpp LayerGlobalGain.h
	LayerGlobalMaxPool2D.cpp LayerGlobalMaxPool2D.h
	LayerGLU.cpp LayerGLU.h
	LayerGTU.cpp LayerGTU.h
	LayerMaxPool2D.cpp LayerMaxPool2D.h
	LayerPELU.cpp LayerPELU.h
	LayerPRelu.cpp LayerPRelu.h
	LayerRandomFlip.cpp LayerRandomFlip.h
	LayerReGLU.cpp LayerReGLU.h	
	LayerRNN.cpp LayerRNN.h
	LayerRRelu.cpp LayerRRelu.h
	LayerSeGLU.cpp LayerSeGLU.h
	LayerSimpleRNN.cpp LayerSimpleRNN.h
	LayerSimplestRNN.cpp LayerSimplestRNN.h
	LayerSoftmax.cpp LayerSoftmax.h
	LayerSoftmin.cpp LayerSoftmin.h
	LayerSwiGLU.cpp LayerSwiGLU.h
	LayerTERELU.cpp LayerTERELU.h
	LayerTimeDistributedBias.cpp LayerTimeDistributedBias.h
	LayerTimeDistributedDense.cpp LayerTimeDistributedDense.h
	LayerTimeDistributedDot.cpp LayerTimeDistributedDot.h
	LayerUniformNoise.cpp LayerUniformNoise.h
	LayerZeroPadding2D.cpp LayerZeroPadding2D.h
	LayerEmbed.cpp LayerEmbed.h
	LayerNormalize.cpp LayerNormalize.h
	LayerParallel.cpp LayerParallel.h
	LayerSelfAttention.cpp LayerSelfAttention.h
	LayerSelfDot.cpp LayerSelfDot.h
	LayerSequential.cpp LayerSequential.h
	LayerStacked.cpp LayerStacked.h
	LayerTransformerFeedForward.cpp LayerTransformerFeedForward.h
	LayerTransformerHeads.cpp LayerTransformerHeads.h
	Loss.cpp Loss.h
	Matrix.cpp Matrix.h
	MetaOptimizer.cpp MetaOptimizer.h
	MinMaxScaler.cpp MinMaxScaler.h
	MNISTReader.cpp MNISTReader.h
	Net.cpp Net.h
	NetTrain.cpp NetTrain.h
	NetUtil.cpp NetUtil.h
	Optimizer.cpp Optimizer.h
	Regularizer.cpp Regularizer.h
	StandardScaler.cpp StandardScaler.h
	LayerTranspose.cpp LayerTranspose.h
	LayerBatchTo2D.cpp LayerBatchTo2D.h
	LayerRouter.h LayerRouter.cpp)
include_directories(.)
add_library( libBeeDNN STATIC ${BEEDNN_FILES})

# Make the library available for installation
set_target_properties(libBeeDNN PROPERTIES
    PUBLIC_HEADER "${BEEDNN_FILES}"
    OUTPUT_NAME "BeeDNN"
)

# Set include directories for the library
target_include_directories(libBeeDNN
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Export the library for use by other projects
install(TARGETS libBeeDNN
    EXPORT BeeDNNTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/BeeDNN
)

# Generate and install the export targets
install(EXPORT BeeDNNTargets
    FILE BeeDNNTargets.cmake
    NAMESPACE BeeDNN::
    DESTINATION lib/cmake/BeeDNN
)
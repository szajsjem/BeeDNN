TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

DEFINES+= "USE_EIGEN_NO"

INCLUDEPATH+=..
INCLUDEPATH+=$$(EIGEN_PATH)

SOURCES += test_xor.cpp \
    ../Net.cpp \
    ../NetTrainLearningRate.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../MatrixUtil.cpp \
    ../LayerActivation.cpp \
    ../LayerDenseWithoutBias.cpp \
    ../LayerDenseWithBias.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrainLearningRate.h \
    ../MatrixUtil.h \
    ../LayerActivation.h \
    ../LayerDenseWithoutBias.h \
    ../LayerDenseWithBias.h

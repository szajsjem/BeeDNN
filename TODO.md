# BeeDNN TODO List

This document tracks planned improvements, optimizations, and known issues for the BeeDNN project.

## Critical Fixes
- [ ] **LayerRNN Backpropagation**: Fix potential index issue in `LayerRNN::backpropagation` (currently `iS > 0`, check if it should cover 0).
- [ ] **Serialization**:
    - Implement `LayerSimplestRNN` serialization (currently marked as TODO in `NetUtil.cpp`).
    - Implement `LayerEmbed` serialization (missing in `test_layers.cpp`).
- [ ] **Error Handling**: Replace asserts with proper failure returns (e.g., in `LayerDot.cpp`).
- [ ] **Thread Safety**: Add locks in `MetaOptimizer.cpp`.

## Optimizations
- [ ] **Matrix Operations** (`Matrix.h/cpp`):
    - Optimize `channelWiseAdd` and `channelWiseMean`.
    - Improve `colExtract` performance (replace with strided matrix views).
    - Optimize `colWiseMin` and `colWiseMax` for empty matrices.
    - Vectorize operations in `Matrix.h` (e.g., squaring, cubing).
    - Ensure performance gains from `USE_EIGEN` are fully realized.
- [ ] **Layer Specific**:
    - Simplify and optimize `LayerSoftmax` and `LayerSoftmin`.
    - Avoid unnecessary resizes in `Net.cpp` (`mTemp = mOut`).
    - Remove redundant copies in `NetTrain.cpp` (`mSampleShuffled`).
- [ ] **Loss Functions**: Vectorize loss calculations in `Loss.cpp`.
- [ ] **Data Processing**:
    - Optimize `StandardScaler` variance calculation.
    - Optimize `MinMaxScaler` (compute directly A and B such as Y=A*x+B).

## Features & Enhancements
- [ ] **Time Series**: Complete the Time Series implementation (currently WIP).
- [ ] **Network Architecture**:
    - Support cutting into mini-batches to save memory in `Net.cpp`.(partial complete thanks to gradient accumulation, needs testing)
    - Implement Xavier initialization for `LayerSimplestRNN` and `LayerSimpleRNN`.
    - Improve `LayerParallel`: solve "how to connect many outputs".
    - Implement position extending(or rotary encoder) for `LayerEmbed`.
- [ ] **Bindings**: Fix issues in `wasm_binding` layer construction.
- [ ] **API**: Call optimizer callback with `trainT` as arg in `MetaOptimizer.cpp`.

## Code Quality & Documentation
- [ ] **Refactoring**:
    - Review `LayerRNN::backpropagation` loop bounds and variable naming.
    - Merge `Matrix` assignment operators.
    - Standardize index handling (e.g., use `Matrix<index>` in `LayerGlobalMaxPool2D`).
- [ ] **Cleanup**: Remove commented-out code in `NetUtil.cpp` and `Matrix.cpp`.
- [ ] **Documentation**: Add more detailed examples for advanced layers.
- [ ] **Testing**: Add unit tests for edge cases (empty matrices, boundary conditions).

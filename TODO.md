# BeeDNN TODO List

This document tracks planned improvements, optimizations, and known issues for the BeeDNN project.

## Critical Fixes
- [ ] **LayerRNN Backpropagation**: Fix potential index issue in `LayerRNN::backpropagation` (currently `iS > 0`, check if it should cover 0 or if logic is correct).
- [ ] **Serialization**: Implement `LayerSimplestRNN` serialization (currently marked as TODO in `NetUtil.cpp`).

## Optimizations
- [ ] **Matrix Operations**:
    - Optimize `channelWiseAdd` in `Matrix.cpp` (currently marked as "todo optimize a lot").
    - Optimize `channelWiseMean` in `Matrix.cpp`.
    - Improve `colExtract` performance (currently "slow for now").
    - Optimize `colWiseMin` and `colWiseMax` for empty matrices.
- [ ] **Eigen Integration**: Ensure full compatibility and performance gains when `USE_EIGEN` is defined.

## Features & Enhancements
- [ ] **Time Series**: Complete the Time Series implementation (currently marked as WIP in documentation).
- [ ] **Testing**: Add unit tests for edge cases, such as empty matrices in `colWiseMin`/`Max`.
- [ ] **Documentation**: Add more detailed examples for advanced layers.

## Code Quality
- [ ] **Refactoring**: Review `LayerRNN::backpropagation` loop bounds and variable naming.
- [ ] **Cleanup**: Remove commented-out code in `NetUtil.cpp` and `Matrix.cpp`.

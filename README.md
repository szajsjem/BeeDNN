# BeeDNN

**BeeDNN** is a lightweight, high-performance Deep Learning library written in C++. It is designed to be simple, dependency-free, and easy to integrate into any C++ project.

## Key Features

*   **Zero Dependencies**: Written in pure C++ (STL only). No external libraries required.
*   **High Performance**: Optimized for speed. Can optionally use [Eigen](http://eigen.tuxfamily.org) for accelerated matrix operations.
*   **Cross-Platform**: Works on Windows, Linux, and macOS. Compatible with Visual Studio and CMake.
*   **Flexible API**:
    *   Decoupled layers and activations.
    *   Support for various initializers (Glorot, He, Lecun).
    *   Wide range of layers (Dense, Conv2D, RNN, LSTM, etc.).
    *   Extensive list of activation functions (ReLU, Gelu, Swish, Mish, etc.).
*   **Serialization**: Save and load models, weights, and training parameters to simple JSON files.
*   **Bindings**: Includes Python and Java bindings (check subdirectories).

## Quick Start

### Prerequisites
*   C++ Compiler (C++11 or later)
*   CMake (optional, for building with CMake)
*   Visual Studio 2019+ (optional, for Windows)

### Building
You can build BeeDNN using CMake or the provided Visual Studio solution.

**Using CMake:**
```bash
mkdir build
cd build
cmake ..
make
```

**Using Visual Studio:**
1.  Open `src/all.sln`.
2.  Select your configuration (Debug/Release).
3.  Build the solution.

### Example: XOR Classification
See `samples/sample_classification_xor` for a simple example.

```cpp
// Pseudo-code snippet
Net net;
net.add(new LayerDense(2, 10));
net.add(new LayerActivation(new Relu()));
net.add(new LayerDense(10, 1));
net.add(new LayerActivation(new Sigmoid()));

NetTrain trainer;
trainer.train(net, inputData, targetData);
```

## Project Structure

*   `src/`: Core library source code.
*   `samples/`: Example programs demonstrating various features (XOR, MNIST, etc.).
*   `tests/`: Unit tests.
*   `python/`: Python bindings and examples.
*   `java_binding/`: Java bindings.
*   `res/`: Resources.

## Documentation

*   **[HOWTO.md](HOWTO.md)**: Detailed guide on compiling, running samples, and using the library.
*   **[TODO.md](TODO.md)**: List of planned features, fixes, and optimizations.

## Contributing

Contributions are welcome! Please check `TODO.md` for ideas or open an issue on GitHub.

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.
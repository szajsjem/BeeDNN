# BeeDNN

**BeeDNN** is a lightweight, high-performance Deep Learning library written in C++. It is designed to be simple, dependency-free, and easy to integrate into any project, with extensive support for modern language bindings and distributed training.

## Key Features

*   **Zero Dependencies**: Written in pure C++ (STL only). No external libraries required.
*   **High Performance**: Optimized for speed. Can optionally use [Eigen](http://eigen.tuxfamily.org) for accelerated matrix operations.
*   **Cross-Platform & Web**: Works on Windows, Linux, macOS, and **Web Browsers (WASM)**.
*   **Multi-Language Bindings**: Full-featured bindings for **Python**, **Java**, and **WebAssembly/JavaScript**.
*   **Distributed Training**: Support for Federated Learning, weight sync, gradient aggregation, and param mixing.
*   **Flexible API**:
    *   Decoupled layers and activations.
    *   Support for various initializers (Glorot, He, Lecun).
    *   Wide range of layers (Dense, Conv2D, RNN, LSTM, etc.).
    *   Extensive list of activation functions (ReLU, Gelu, Swish, Mish, etc.).
*   **Serialization**: Save and load models, weights, and training parameters.

## Quick Start

### Prerequisites
*   C++ Compiler (C++11 or later)
*   CMake (optional, for building with CMake)
*   Emscripten (optional, for WASM bindings)
*   JDK & Python (optional, for bindings)

### Building
You can build BeeDNN using CMake or the provided Visual Studio solution.

**Using CMake:**
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

**For WebAssembly (WASM):**
Check `wasm_binding/compile.bat` or use Emscripten via CMake.

### Example: Python
```python
from beednn.BeeDNNLoader import BeeDNNLoader
import numpy as np

loader = BeeDNNLoader("BeeDNNLib.dll")
net = loader.create_net()
loader.add_dense(net, 2, 8, "Relu")
loader.add_dense(net, 8, 1, "Sigmoid")

# Predict
output = loader.predict(net, np.array([[0, 1]], dtype=np.float32))
```

## Project Structure

*   `src/`: Core library source code.
*   `wasm_binding/`: WebAssembly/JS bindings and web examples.
*   `python_binding/`: C-API and Python loader integration.
*   `java_binding/`: JNI bindings for Java (Self-contained JAR).
*   `python/`: Higher-level Python API.
*   `samples/`: Example programs (C++).
*   `tests/`: Unit tests and verification scripts.

## Documentation

*   **[HOWTO.md](HOWTO.md)**: Detailed guide on compiling and using the library.
*   **[TODO.md](TODO.md)**: Roadmap and planned optimizations.
*   **[save_load_walkthrough.md](save_load_walkthrough.md)**: Guide on model serialization.

## Contributing

Contributions are welcome! Please check `TODO.md` for ideas or open an issue on GitHub.

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.

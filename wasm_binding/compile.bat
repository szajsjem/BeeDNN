@echo off
echo Compiling BeeDNN WebAssembly...
mkdir build
cd build
emcmake cmake ..
emmake make
echo Done. Check build/beednn.js and build/beednn.wasm
pause

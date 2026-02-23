cmake_minimum_required(VERSION 3.18.0)
project(BeeDNNWasm)

# Assuming EMSCRIPTEN env is set, or run with emcmake
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(../src)

# Source files
file(GLOB SOURCES_CORE ../src/*.cpp)
set(SOURCES
    src/BeeDNNWasm.cpp
    ${SOURCES_CORE}
)

# Create executable (which emscripten turns into .js + .wasm)
add_executable(beednn ${SOURCES})

# Emscripten specific settings
target_compile_options(beednn PRIVATE "-fexceptions")
target_link_options(beednn PRIVATE 
    "-lembind" 
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=BeeDNNModule"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sENVIRONMENT=web,worker"
    "-fexceptions" 
)

# Output beednn.js and beednn.wasm
set_target_properties(beednn PROPERTIES SUFFIX ".js")

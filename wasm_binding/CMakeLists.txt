cmake_minimum_required(VERSION 3.18.0)
project(BeeDNNWasm)

# Assuming EMSCRIPTEN env is set, or run with emcmake
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(../src)

# Source files
set(SOURCES
    src/BeeDNNWasm.cpp
)

# Create executable (which emscripten turns into .js + .wasm)
add_executable(beednn ${SOURCES})

# Link BeeDNN static library
target_link_libraries(beednn libBeeDNN)

# Emscripten specific settings
target_compile_options(beednn PRIVATE "-fexceptions")
target_link_options(beednn PRIVATE 
    "-lembind" 
    "-s" "MODULARIZE=1"
    "-s" "EXPORT_NAME='BeeDNNModule'"
    "-s" "ALLOW_MEMORY_GROWTH=1"
    "-s" "environment='web,worker'"
    "-fexceptions" 
)

# Output beednn.js and beednn.wasm
set_target_properties(beednn PROPERTIES SUFFIX ".js")

cmake_minimum_required(VERSION 3.18.0)

# Include directories
include_directories(../src)

# Create shared library for Python ctypes
add_library(BeeDNNLib SHARED BeeDNNLib.cpp)

# Link against BeeDNN static library
target_link_libraries(BeeDNNLib PRIVATE libBeeDNN)

target_compile_definitions(BeeDNNLib PRIVATE BEEDNNLIB_BUILD)

# Output name
set_target_properties(BeeDNNLib PROPERTIES 
    OUTPUT_NAME "BeeDNNLib"
    PREFIX ""
)

if(WIN32)
    set_target_properties(BeeDNNLib PROPERTIES SUFFIX ".dll")
elseif(APPLE)
    set_target_properties(BeeDNNLib PROPERTIES SUFFIX ".dylib")
else()
    set_target_properties(BeeDNNLib PROPERTIES SUFFIX ".so")
endif()

find_package(JNI REQUIRED)
find_package(Java REQUIRED)
include(UseJava)

# Add JNI headers to include path
include_directories(${JNI_INCLUDE_DIRS})

# Create the native library
add_library(BeeDNNJava SHARED
    src/BeeDNNJava.cpp
)

# Link against the main BeeDNN library
target_link_libraries(BeeDNNJava 
    PRIVATE 
        libBeeDNN
        ${JNI_LIBRARIES}
)

# Set output name based on platform
if(WIN32)
    set_target_properties(BeeDNNJava PROPERTIES 
        OUTPUT_NAME "BeeDNNJava"
        PREFIX ""
        SUFFIX ".dll"
    )
    set(NATIVE_LIB_NAME "BeeDNNJava.dll")
    set(NATIVE_LIB_PATH "win32-x86-64")
elseif(APPLE)
    set_target_properties(BeeDNNJava PROPERTIES 
        OUTPUT_NAME "libBeeDNNJava"
        SUFFIX ".dylib"
    )
    set(NATIVE_LIB_NAME "libBeeDNNJava.dylib")
    set(NATIVE_LIB_PATH "darwin")
else()
    set_target_properties(BeeDNNJava PROPERTIES 
        OUTPUT_NAME "libBeeDNNJava"
        SUFFIX ".so"
    )
    set(NATIVE_LIB_NAME "libBeeDNNJava.so")
    set(NATIVE_LIB_PATH "linux-x86-64")
endif()

# Create jar file from Java sources
set(CMAKE_JAVA_COMPILE_FLAGS "-source" "8" "-target" "8")
add_jar(BeeDNNJar
    SOURCES
        java/com/beednn/Net.java
        java/com/beednn/NetTrain.java
        java/com/beednn/Layer.java
    RESOURCE
        $<TARGET_FILE:BeeDNNJava> # sadly, doesnt put the dll inside
    OUTPUT_NAME
        BeeDNN
)

add_dependencies(BeeDNNJar BeeDNNJava)
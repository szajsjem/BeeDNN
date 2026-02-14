find_package(JNI REQUIRED)
find_package(Java REQUIRED)
include(UseJava)

# Add JNI headers to include path
include_directories(${JNI_INCLUDE_DIRS} ../src)

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
    OUTPUT_NAME
        BeeDNN
)

add_dependencies(BeeDNNJar BeeDNNJava)

# Bundle native library into JAR
if(WIN32)
    set(LIB_PLATFORM "win32-x86-64")
    set(LIB_NAME "BeeDNNJava.dll")
elseif(APPLE)
    set(LIB_PLATFORM "darwin")
    set(LIB_NAME "libBeeDNNJava.dylib")
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
else()
    set(LIB_PLATFORM "linux-x86-64")
    set(LIB_NAME "libBeeDNNJava.so")
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

# Staged location inside JAR: /native/<platform>/<libname>
set(NATIVE_DIR "native/${LIB_PLATFORM}")
set(STAGING_DIR "${CMAKE_CURRENT_BINARY_DIR}/staging")

add_custom_command(TARGET BeeDNNJar POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${STAGING_DIR}/${NATIVE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:BeeDNNJava> ${STAGING_DIR}/${NATIVE_DIR}/${LIB_NAME}
    COMMAND "${Java_JAR_EXECUTABLE}" uf "${CMAKE_CURRENT_BINARY_DIR}/BeeDNN.jar" -C "${STAGING_DIR}" .
    COMMENT "Bundling native library ${LIB_NAME} into JAR at ${NATIVE_DIR}"
)
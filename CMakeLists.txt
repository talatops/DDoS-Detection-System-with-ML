cmake_minimum_required(VERSION 3.10)
project(DDoSGpuDetector VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBPCAP REQUIRED libpcap)

# OpenCL
find_package(OpenCL REQUIRED)

# Boost
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Python3 for ML inference
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Include directories
include_directories(
    ${LIBPCAP_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    src/
)

# Source files
set(INGEST_SOURCES
    src/ingest/pcap_reader.cpp
    src/ingest/window_manager.cpp
)

set(DETECTOR_SOURCES
    src/detectors/entropy_cpu.cpp
    src/detectors/cusum_detector.cpp
    src/detectors/pca_detector.cpp
    src/detectors/decision_engine.cpp
)

set(OPENCL_SOURCES
    src/opencl/host.cpp
    src/opencl/gpu_detector.cpp
)

set(BLOCKING_SOURCES
    src/blocking/rtbh_controller.cpp
    src/blocking/pcap_filter.cpp
    src/blocking/iptables_simulator.cpp
)

set(UTILS_SOURCES
    src/utils/metrics_collector.cpp
    src/utils/resource_monitor.cpp
    src/utils/logger.cpp
    src/utils/simple_json.cpp
    src/utils/config_loader.cpp
)

set(ML_SOURCES
    src/ml/inference_engine.cpp
    src/ml/feature_builder.cpp
)

set(ALL_SOURCES
    ${INGEST_SOURCES}
    ${DETECTOR_SOURCES}
    ${OPENCL_SOURCES}
    ${BLOCKING_SOURCES}
    ${UTILS_SOURCES}
    ${ML_SOURCES}
)

# Executable
add_executable(detector
    src/main.cpp
    ${ALL_SOURCES}
)

# Link libraries
target_link_libraries(detector
    ${LIBPCAP_LIBRARIES}
    ${OpenCL_LIBRARIES}
    ${Boost_LIBRARIES}
    ${Python3_LIBRARIES}
    pthread
)

# Compiler flags
target_compile_options(detector PRIVATE
    -Wall -Wextra -O3
    ${LIBPCAP_CFLAGS_OTHER}
)

# OpenCL kernel compilation (if needed)
# add_custom_command(
#     OUTPUT ${CMAKE_BINARY_DIR}/kernels/entropy.cl.bin
#     COMMAND ${CMAKE_COMMAND} -E copy
#         ${CMAKE_SOURCE_DIR}/src/opencl/kernels/entropy.cl
#         ${CMAKE_BINARY_DIR}/kernels/entropy.cl.bin
#     DEPENDS src/opencl/kernels/entropy.cl
# )

# Install
install(TARGETS detector DESTINATION bin)


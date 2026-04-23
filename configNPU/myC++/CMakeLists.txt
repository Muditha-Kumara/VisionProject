cmake_minimum_required(VERSION 3.10)
project(luckfox_benchmark)

set(CMAKE_CXX_STANDARD 11)

# --- Path Configuration ---
set(SDK_PATH "/home/muditha/VisionProject/configNPU/luckfox-pico")
set(EXAMPLE_PATH "/home/muditha/VisionProject/configNPU/luckfox_pico_rkmpi_example")

# --- RKNN runtime for Luckfox Pico (RV1106, armhf) ---
set(RKNN_LIB "${EXAMPLE_PATH}/lib/uclibc/librknnmrt.so")
set(RGA_LIB "${EXAMPLE_PATH}/lib/uclibc/librga.so")

include_directories(
    ${EXAMPLE_PATH}/include
    ${EXAMPLE_PATH}/include/rknn
    ${EXAMPLE_PATH}/lib/uclibc/include/opencv4
    ${SDK_PATH}/sysdrv/source/mpp/include
)

add_executable(luckfox_benchmark main.cc)

# --- Linking Strategy ---
target_link_libraries(luckfox_benchmark
    ${EXAMPLE_PATH}/lib/uclibc/lib/libopencv_highgui.a
    ${EXAMPLE_PATH}/lib/uclibc/lib/libopencv_imgproc.a
    ${EXAMPLE_PATH}/lib/uclibc/lib/libopencv_core.a
    ${RKNN_LIB}
    ${RGA_LIB}
    pthread
    dl
)

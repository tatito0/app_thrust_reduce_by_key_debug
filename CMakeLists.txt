set(CMAKE_EXPORT_COMPILE_COMMANDS)
cmake_minimum_required(VERSION 2.8)
project(app_thrust_reduce_by_key_debug)

# Overide with cmake -DCMAKE_BUILD_TYPE=Debug {dir}
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release )
endif()
message("Build type: ${CMAKE_BUILD_TYPE}")

#Compiler setup
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

find_package(CUDA REQUIRED)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "CUDA RELEASE mode enabled")
    list(APPEND CUDA_NVCC_FLAGS "-O3")
else()
    message(STATUS "CUDA DEBUG mode enabled")
    list(APPEND CUDA_NVCC_FLAGS "-O0" "-DTHRUST_DEBUG")
endif()

list(APPEND LINK_LIBS ${CUDA_LIBRARIES})

cuda_add_executable(app_thrust_reduce_by_key_debug main.cu utils.h picojson.h)
target_link_libraries(app_thrust_reduce_by_key_debug ${LINK_LIBS})

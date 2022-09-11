# CMakeList.txt : CMake project for OpenCLSieve, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("OpenCLSieve")

find_package(OpenCL REQUIRED)


# Add source to this project's executable.
add_executable (${PROJECT_NAME} "OpenCLSieve.cpp" "OpenCLSieve.h")


target_include_directories(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)

message(STATUS "Using OpenCL ${OpenCL_INCLUDE_DIRS}")

set_target_properties(${PROJECT_NAME} PROPERTIES C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=100)
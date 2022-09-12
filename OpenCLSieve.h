#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <cstdint>
#include <tuple>
#include <chrono>
static void compact_results(std::vector<uint32_t>& buffer, std::vector<uint32_t>& results, uint32_t& index);

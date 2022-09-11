#pragma once

#include <iostream>
#include <fstream>
// OpenCL includes
#include <CL/cl.hpp>
#include <cstdint>
#include <tuple>

static void compact_results(std::vector<uint32_t>& buffer, std::vector<uint32_t>& results, uint32_t& index);

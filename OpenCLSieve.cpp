// OpenCLSieve.cpp : Defines the entry point for the application.
//

#include "OpenCLSieve.h"

#define CHECK_CALL(call) do{cl_int res = (call); if(res != CL_SUCCESS){ fprintf(stderr,"OpenCL call failed in line %u with code %u", __LINE__, -res); exit(1);}} while(0);
static size_t WORK_QUEUE_SIZE = 64;

static std::tuple<cl::Device,cl::Platform> get_device() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Device gpu_device{}, cpu_device{};
    bool gpu_found = false, cpu_found = false;
    cl::STRING_CLASS cpu_device_name, gpu_device_name;
    cl::Platform gpu_platform, cpu_platform;

    for (auto platform : platforms) {
        std::vector<cl::Device> devices; 
        cl::STRING_CLASS platform_name, platform_vendor;

        std::cout << "Platform \"" << platform.getInfo<CL_PLATFORM_NAME>() << "\" by vendor \"" << platform.getInfo<CL_PLATFORM_VENDOR>() << "\" found! Devices: \n";

        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (auto device : devices) {
            cl::STRING_CLASS device_name;
            auto device_type = device.getInfo<CL_DEVICE_TYPE>();
            device_name = device.getInfo<CL_DEVICE_NAME>();
            std::cout << device_name;
            if (device_type == CL_DEVICE_TYPE_GPU) {
                std::cout << " (GPU)";
                gpu_device = device;
                gpu_found = true;
                gpu_device_name = device_name;
                gpu_platform = platform;
            }
            else if (device_type == CL_DEVICE_TYPE_CPU) {
                std::cout << " (CPU)";
                cpu_device = device;
                cpu_found = true;
                cpu_device_name = device_name;
                cpu_platform = platform;
            }
            else {
                std::cout << "(unknown)";
            }
            std::cout << std::endl;

        }

    }

    if (gpu_found) {
        std::cout << "Using last GPU device \"" << gpu_device_name << "\"!\n";
        WORK_QUEUE_SIZE = gpu_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        return { gpu_device,gpu_platform };
    }
    if (cpu_found) {
        std::cout << "Using last CPU device \"" << cpu_device_name << "\"!\n";
        WORK_QUEUE_SIZE = cpu_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        return { cpu_device, cpu_platform };
    }
    std::cerr << "No OpenCL device found!\n";
    exit(1);
}


void compact_results(std::vector<uint32_t>& buffer, std::vector<uint32_t>& results, uint32_t& index)
{
    // kernel skips these special cases
    buffer[0] = buffer[1] = 1;

    for (uint32_t i = 0; i < buffer.size(); i++) {
        results[index] = i;
        index += (buffer[i] == 0);
    }

    results.resize(index);

}

static std::tuple<std::vector<uint32_t>,uint32_t> run_sieve_kernel(cl::Device& device, cl::Platform& platform, const uint32_t max_number) {
    std::string src="#define __CL_ENABLE_EXCEPTIONS\n" 
        "void kernel prime_sieve(__global uint * primes, const uint N) {\n"
        "uint start_offset = 0;\n"
        "uint end_offset = N;\n"
        "uint factor = 2 + get_global_id(0);\n"
            "for (uint offset = start_offset + 2 * factor; offset < end_offset; offset += factor) {\n"
            "    primes[offset] = 1;\n"
            "}\n"
        "}\n";
    std::vector<cl::Device> devices;
    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    cl::Context context(device);
    cl::Program program(context, sources);

    auto err = program.build(devices, "-Werror");

    if (err!=CL_SUCCESS) {
        std::cerr << "Error building kernel " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }


    cl::Buffer mem_buf(context, CL_MEM_READ_WRITE, max_number * sizeof(uint32_t));
    cl::Kernel kernel(program, "prime_sieve", &err);

    if (err != CL_SUCCESS) {
        std::cerr << "Error instanciating kernel " << -err << "\n";
        exit(1);
    }

    CHECK_CALL(kernel.setArg(0, mem_buf));
    CHECK_CALL(kernel.setArg(1, max_number));

    cl::CommandQueue queue(context, device);

    cl::Event kernel_finished;

    CHECK_CALL(queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(max_number - 2), cl::NullRange,nullptr,&kernel_finished));
    
    CHECK_CALL(kernel_finished.wait());

    std::vector<uint32_t> buffer(max_number);
    CHECK_CALL(queue.enqueueReadBuffer(mem_buf, CL_TRUE, 0, buffer.size() * sizeof(uint32_t), buffer.data()));
    CHECK_CALL(queue.finish());

    std::vector<uint32_t> results(max_number);
    uint32_t index = 0;

    compact_results(buffer, results, index);

    return { results,index };
}

static int64_t get_smallest_non_marked(std::vector<uint32_t>& vector) {
    auto first = std::find(vector.begin(), vector.end(), 0);
    if (first == vector.end()) {
        return -1;
    }
    return std::distance(vector.begin(), first);
}

static void mark_edge_cases(std::vector<uint32_t>& vector) {
    // edge cases - 1 is neither even nor prime
    vector[1] = 1;
    // mark all even numbers
    for (int index = 0; index < vector.size(); index+=2) {
        vector[index] = 1;
    }
}

static void extend_primes(std::vector<uint32_t>& primes) {
    const size_t old_size = primes.size();
    const size_t new_size = old_size + WORK_QUEUE_SIZE - (old_size % WORK_QUEUE_SIZE);
    // extend with 2 (which is a valid prime) so the kernel does not have to deal with the edge case
    for (size_t current = 0; current < new_size - old_size; current++) {
        primes.push_back(2);
    }
}

static int64_t run_square_kernel(cl::Device& device, cl::Platform& platform, const uint32_t max_number,  const std::vector<uint32_t>& primes, const std::string& src) {
    // should be even, since primes was padded earlier
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::steady_clock::time_point end;
    std::vector<cl::Device> devices;
    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    cl::Context context(device);
    cl::Program program(context, sources);

    auto err = program.build(devices, "-Werror");

    if (err != CL_SUCCESS) {
        std::cerr << "Error building kernel " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    // temporary buffer for the kernel to store marks in, 0-initialized
    cl::Buffer mem_buf(context, CL_MEM_READ_WRITE, max_number * sizeof(uint32_t)), prime_buf(context, CL_MEM_READ_WRITE, primes.size() * sizeof(uint32_t));
    cl::Kernel kernel(program, "square_sieve", &err);

    if (err != CL_SUCCESS) {
        std::cerr << "Error instanciating kernel " << -err << "\n";
        exit(1);
    }

    CHECK_CALL(kernel.setArg(0, prime_buf));
    CHECK_CALL(kernel.setArg(1, mem_buf));
    CHECK_CALL(kernel.setArg(2, max_number));



    cl::CommandQueue queue(context, device);

    cl::Event kernel_finished;

    CHECK_CALL(queue.enqueueWriteBuffer(prime_buf, CL_TRUE, 0, primes.size() * sizeof(uint32_t), primes.data()));

    // one work item (i.e. GPU task) is responsible for one prime, iterating all factors
    CHECK_CALL(queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(primes.size()), cl::NDRange(WORK_QUEUE_SIZE), nullptr, &kernel_finished));

    CHECK_CALL(kernel_finished.wait());

    std::vector<uint32_t> buffer(max_number);
    CHECK_CALL(queue.enqueueReadBuffer(mem_buf, CL_TRUE, 0, buffer.size() * sizeof(uint32_t), buffer.data()));
    CHECK_CALL(queue.finish());

    mark_edge_cases(buffer);

    int64_t result = get_smallest_non_marked(buffer);
    end = std::chrono::high_resolution_clock::now();

    std::cout << "Kernel invocation took " << std::chrono::duration<double, std::ratio<1>>(end - start).count() << " seconds" << std::endl;

    return result;
}

static int64_t run_original_goldbach_kernel(cl::Device& device, cl::Platform& platform, const uint32_t maxNumber, const std::vector<uint32_t>& primes) {
    const std::string src = "#define __CL_ENABLE_EXCEPTIONS\n"
        "void kernel square_sieve(__global uint * primes, __global uint* outputs, const uint number_outputs) {\n"
            "uint start_offset = 1;\n"
            "uint end_offset = number_outputs;\n"
            "uint prime_index = get_global_id(0);\n"
            "const uint prime = primes[prime_index];\n"
            "uint square_number = 1;\n"
            "outputs[prime] = 1;\n"
            "uint result;\n"
            "while(true){\n"
                "result  = prime + 2 * square_number * square_number;\n"
                "square_number ++;\n"
                "if(result < number_outputs){\n"
                    "outputs[result] = 1;\n"
                "}\n"
                "else {\n"
                    "break;\n"
                "}\n"
            "}\n"
        "}\n";
    return run_square_kernel(device, platform, maxNumber, primes, src);

}

static int64_t run_variation_goldbach_kernel(cl::Device& device, cl::Platform& platform, const uint32_t maxNumber, const std::vector<uint32_t>& primes) {
    const std::string src = "#define __CL_ENABLE_EXCEPTIONS\n"
        "void kernel square_sieve(__global uint * primes, __global uint* outputs, const uint number_outputs) {\n"
            "uint start_offset = 1;\n"
            "uint end_offset = number_outputs;\n"
            "const uint prime_index = get_global_id(0);\n"
            "const uint prime = primes[prime_index];\n"
            "uint square_number_one = 1;\n"
            "outputs[prime] = 1;\n"
            "uint result;\n"
            "while(square_number_one * square_number_one + 2 < number_outputs - prime){\n"
                "uint square_number_two = 1;\n"
                "while(true){\n"
                    "result  = prime + square_number_one * square_number_one + square_number_two * square_number_two;\n"
                    "square_number_two ++;\n"
                    "if(result < number_outputs){\n"
                        "outputs[result] = 1;\n"
                    "}\n"
                    "else{\n"
                        "break;\n"
                    "}\n"
                "}\n"
                "square_number_one ++;\n"
            "}\n"
        "}\n";
    return run_square_kernel(device, platform, maxNumber, primes, src);
}

int main(int argc, const char *argv[])
{
    const int default_n = 10000;
    int N = default_n;
    int number_primes;
    int64_t smallest_square, smallest_square_varied;
    std::vector<uint32_t> primes;
    std::vector<uint32_t> non_reachable_single_square;

    cl::Device device;
    cl::Platform platform;

    if (argc > 1) {
        N = std::atoi(argv[1]);
        // 0 passed or conversion failed
        if (!N) {
            std::cerr << "Usage: " << argv[0] << "[ max_number, default " << default_n << "]\n";
            return 1;
        }
    }
    
    // make sure that global work queue size is divisible by size of local work queue, saving edge cases
    N += N % WORK_QUEUE_SIZE;

    std::cout << "Limiting search to " << N << "\n";

    std::tie(device,platform) = get_device();
    std::tie(primes, number_primes) = run_sieve_kernel(device,platform,N);

    extend_primes(primes);

    std::cout << "Found " << number_primes << " primes below " << N << std::endl;
    smallest_square = run_original_goldbach_kernel(device, platform, N, primes);

    std::cout << "Smallest number not representable by sum of a prime and double of a square number: ";
    if (smallest_square != -1) {
         std::cout << smallest_square;
    }
    else {
        std::cout << "not found!";
    }
    std::cout << std::endl;

    smallest_square_varied = run_variation_goldbach_kernel(device, platform, N, primes);

    std::cout << "Smallest number not representable by sum of a prime and two square numbers: ";
    if (smallest_square_varied != -1) {
        std::cout << smallest_square_varied;
    }
    else {
        std::cout << "not found!";
    }
    std::cout << std::endl;

	return 0;
}

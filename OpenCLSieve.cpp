// OpenCLSieve.cpp : Defines the entry point for the application.
//

#include "OpenCLSieve.h"

#define CHECK_CALL(call) do{cl_int res = (call); if(res != CL_SUCCESS){ fprintf(stderr,"OpenCL call failed in line %u with code %u", __LINE__, -res); exit(1);}} while(0);

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
        return { gpu_device,gpu_platform };
    }
    if (cpu_found) {
        std::cout << "Using last CPU device \"" << cpu_device_name << "\"!\n";
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

static std::tuple<std::vector<uint32_t>,uint32_t> run_sieve_kernel(cl::Device& device, cl::Platform& platform, const uint32_t maxNumber) {
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


    cl::Buffer memBuf(context, CL_MEM_READ_WRITE, maxNumber * sizeof(uint32_t));
    cl::Kernel kernel(program, "prime_sieve", &err);

    if (err != CL_SUCCESS) {
        std::cerr << "Error instanciating kernel " << -err << "\n";
        exit(1);
    }

    CHECK_CALL(kernel.setArg(0, memBuf));
    CHECK_CALL(kernel.setArg(1, maxNumber));

    cl::CommandQueue queue(context, device);

    cl::Event kernel_finished;

    CHECK_CALL(queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(maxNumber - 2), cl::NullRange,nullptr,&kernel_finished));
    
    CHECK_CALL(kernel_finished.wait());

    std::vector<uint32_t> buffer(maxNumber);
    CHECK_CALL(queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, buffer.size() * sizeof(uint32_t), buffer.data()));
    CHECK_CALL(queue.finish());

    std::vector<uint32_t> results(maxNumber);
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

static int64_t run_square_kernel(cl::Device& device, cl::Platform& platform, const uint32_t maxNumber,  const std::vector<uint32_t>& primes) {
    std::string src = "#define __CL_ENABLE_EXCEPTIONS\n"
        "void kernel square_sieve(__global uint * primes, __global uint* outputs, const uint number_outputs) {\n"
        "uint start_offset = 1;\n"
        "uint end_offset = number_outputs;\n"
        "const uint prime = primes[get_global_id(0)];\n"
        "uint square_number = 1;\n"
        "outputs[prime] = 1;\n"
        "uint result;\n"
        "do{\n"
            "result  = prime + 2 * square_number * square_number;\n"
            "square_number ++;\n"
            "outputs[result] = 1;\n"
        "} while(result < number_outputs);\n"
        "}\n";
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


    cl::Buffer memBuf(context, CL_MEM_READ_WRITE, maxNumber * sizeof(uint32_t)), prime_buf(context, CL_MEM_READ_WRITE, primes.size() * sizeof(uint32_t));
    cl::Kernel kernel(program, "square_sieve", &err);

    if (err != CL_SUCCESS) {
        std::cerr << "Error instanciating kernel " << -err << "\n";
        exit(1);
    }

    CHECK_CALL(kernel.setArg(0, prime_buf));
    CHECK_CALL(kernel.setArg(1, memBuf));
    CHECK_CALL(kernel.setArg(2, maxNumber));


    cl::CommandQueue queue(context, device);

    cl::Event kernel_finished;

    CHECK_CALL(queue.enqueueWriteBuffer(prime_buf, CL_TRUE, 0, primes.size() * sizeof(uint32_t), primes.data()));

    CHECK_CALL(queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(primes.size()), cl::NullRange, nullptr, &kernel_finished));

    CHECK_CALL(kernel_finished.wait());

    std::vector<uint32_t> buffer(maxNumber);
    CHECK_CALL(queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, buffer.size() * sizeof(uint32_t), buffer.data()));
    CHECK_CALL(queue.finish());

    mark_edge_cases(buffer);

    return get_smallest_non_marked(buffer);
}




int main()
{
    const int N = 10000;
    int numberPrimes;
    int64_t smallest_square;
    std::vector<uint32_t> primes;
    std::vector<uint32_t> non_reachable_single_square;

    cl::Device device;
    cl::Platform platform;
    std::tie(device,platform) = get_device();
    std::tie(primes, numberPrimes) = run_sieve_kernel(device,platform,N);

    std::cout << "Found " << numberPrimes << " primes below " << N << std::endl;
    smallest_square = run_square_kernel(device, platform, N, primes);

    std::cout << "Smallest number not representable by sum of a prime and double of a square number: ";
    if (smallest_square != -1) {
         std::cout << smallest_square;
    }
    else {
        std::cout << "not found!";
    }
    std::cout << std::endl;

	return 0;
}

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    // ???????? GPU ??
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess) {
        std::cerr << "?? GPU ??, CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "??? " << deviceCount << " ? GPU?" << std::endl;
    std::cout << "=========================================" << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        // ?? API:??? i ? GPU ?????
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device ID     : " << i << std::endl;
        std::cout << "????      : " << prop.name << std::endl;
        std::cout << "????      : SM " << prop.major << "." << prop.minor 
                  << " (?? 9.0 ?? Hopper, 10.0 ?? Blackwell)" << std::endl;
        
        // 1. ?? SM (Streaming Multiprocessor) ??
        std::cout << "NUM_SMS       : " << prop.multiProcessorCount << " ? SMs" << std::endl;
        
        // 2. ?? L2 Cache ??
        std::cout << "L2_SIZE       : " << prop.l2CacheSize << " Bytes (" 
                  << prop.l2CacheSize / (1024.0 * 1024.0) << " MiB)" << std::endl;
        
        // 3. ????????? (?????? MAX_DATA_VOLUME ?????)
        std::cout << "VRAM_TOTAL    : " << prop.totalGlobalMem << " Bytes (" 
                  << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GiB)" << std::endl;
        
        std::cout << "=========================================" << std::endl;
    }

    return 0;
}
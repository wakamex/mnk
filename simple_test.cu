#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    printf("Found %d devices, error: %s\n", count, cudaGetErrorString(err));
    if (count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
    }
    return 0;
}

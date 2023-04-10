#include "cuda/gpu_utils.h"

namespace CUDA
{
    bool check_for_gpu()
    {
        cudaError_t err0;
        int nb_devices;
        err0 = cudaGetDeviceCount(&nb_devices);

        if (err0 != cudaSuccess || nb_devices == 0)
        {
            return false;
        }else{
            return true;
        }
    }

    int get_device_count()
    {
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        return nDevices;
    }

    cudaDeviceProp get_device_properties(size_t devID)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, devID);

        return prop;
    }
}

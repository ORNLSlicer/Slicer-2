#ifdef NVCC_FOUND

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <driver_types.h>

/*!
    \brief CUDA functions that provide access to device information
*/
namespace CUDA
{
    //! \brief Checks if the host has a CUDA GPU installed
    bool check_for_gpu();

    //! \brief Returns the amount of CUDA GPUs in a system
    //! \return the number of the device
    int get_device_count();

    //! \brief Returns the a struct that contains GPU information
    cudaDeviceProp get_device_properties(size_t devID);

}
#endif // NVCC_FOUND
#endif // GPU_UTILS_H

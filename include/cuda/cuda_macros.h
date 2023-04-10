#ifndef CUDA_MACROS
#define CUDA_MACROS

/*!
  \title CUDA compiler definitions and macros
  */
#ifdef __CUDACC__
//! \def NVCC compilation
    #define CUDA_COMPILATION

    //! \macro Device code compilation
    #ifdef __CUDA_ARCH__
      #define DEVICE_CODE_COMPILATION
    #endif

    //! \macro Device code that can be launched from host code
    #define GPU_LAUNCHABLE  __global__

    //! \macro Device code used inside the lambda
    #define GPU_LAMBDA      __device__

    //! \macro Device code
    #define GPU_ONLY        __device__

    //! \macro Device and host code
    #define GPU_CPU_CODE    __host__ __device__

    //! \macro Host code
    #define CPU_ONLY        __host__
#else
//! \macro Device code that can be launched from host code
#define GPU_LAUNCHABLE

//! \macro Device code used inside the lambda
#define GPU_LAMBDA

//! \macro Device code
#define GPU_ONLY

//! \macro Device and host code
#define GPU_CPU_CODE

//! \macro Host code
#define CPU_ONLY
#endif

#endif

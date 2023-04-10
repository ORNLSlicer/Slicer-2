#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H

// Locals
#include "cuda/cuda_macros.h"

// Qt
#include <QSharedPointer>
#include <QVector>

// CUDA
#ifdef NVCC_FOUND
#include <driver_types.h>
#endif

namespace ORNL
{
    //! \brief Define for easy access to this singleton.
    #define GPU GPUManager::getInstance()

    // \brief the minimum required version of CUDA compute
    #define MAJOR_VERSION 5
    #define MINOR_VERSION 2

    /*!
     * \class GPUManager
     * \brief provides safe access to GPU
     */
    class GPUManager
    {
    public:
        //! \brief Get the singleton instance of this object.
        static QSharedPointer<GPUManager> getInstance();

        //! \brief should the GPU be used
        //! \note based on settings, hardware available and what Slicer 2 was compiled with
        //! \return if the GPU should be used
        bool use();

        //! \brief gets if gpu support is available
        //! \return if Slicer 2 was compiled with support for CUDA
        bool hasSupport();

        //! \brief how many GPU are installed in thsi system
        //! \return the number of CUDA capable GPUs
        size_t count();

        #ifdef NVCC_FOUND
        //! \brief get device information for a GPU
        //! \param id: the id of the GPU installed in the system
        //! \return information on the GPU
        cudaDeviceProp getDeviceInformation(size_t id);
        #endif

        //! \brief the ids of compatible GPUs installed in the system
        //! \return a list of CUDA device ids
        QVector<int> getDeviceIds();

    private:
        //! \brief Constructor
        GPUManager();

        //! \brief Singleton pointer.
        static QSharedPointer<GPUManager> m_singleton;

        //! \brief if Slicer 2 was compiled with support for CUDA
        bool m_has_nvcc = false;

        //! \brief ids of compatible GPUs
        QVector<int> m_gpu_ids;
    };
}


#endif //GPU_MANAGER_H

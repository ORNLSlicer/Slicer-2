#include "managers/gpu_manager.h"

// Locals
#include "managers/settings/settings_manager.h"

// CUDA Locals
#ifdef NVCC_FOUND
#include "cuda/gpu_utils.h"
#endif

namespace ORNL
{
    QSharedPointer<GPUManager> GPUManager::m_singleton =
            QSharedPointer<GPUManager>();

    QSharedPointer<GPUManager> GPUManager::getInstance()
    {
        if (m_singleton.isNull()) {
            m_singleton.reset(new GPUManager());
        }
        return m_singleton;
    }

    GPUManager::GPUManager()
    {
        // Set flag if NVCC for NVCC
        #ifdef NVCC_FOUND
        m_has_nvcc = true;

        // Count the number of installed GPUs
        if(::CUDA::check_for_gpu())
            for(int i = 0, end = ::CUDA::get_device_count(); i < end; ++i)
            {
                if(getDeviceInformation(i).computeMode == cudaComputeModeDefault)
                {
                    if(getDeviceInformation(i).major == MAJOR_VERSION)
                    {
                        if (getDeviceInformation(i).minor >= MINOR_VERSION)
                            m_gpu_ids.push_back(i);
                    }
                    else if(getDeviceInformation(i).major > MAJOR_VERSION)
                    {
                        m_gpu_ids.push_back(i);
                    }
                }
            }
        #endif

    }

    bool GPUManager::use()
    {
        bool accel_enabled = GSM->getGlobal()->setting<bool>(Constants::ProfileSettings::Optimizations::kEnableGPU);
        return (m_has_nvcc && count() > 0 && accel_enabled);
    }

    bool GPUManager::hasSupport()
    {
    #ifdef NVCC_FOUND
        return true;
    #else
        return false;
    #endif
    }

    #ifdef NVCC_FOUND
    cudaDeviceProp GPUManager::getDeviceInformation(size_t id)
    {
        return ::CUDA::get_device_properties(id);
    }
    #endif

    size_t GPUManager::count()
    {
        return m_gpu_ids.count();
    }

    QVector<int> GPUManager::getDeviceIds()
    {
        return m_gpu_ids;
    }

}

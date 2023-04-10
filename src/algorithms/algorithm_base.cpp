#include <algorithms/algorithm_base.h>

//Local
#include "managers/gpu_manager.h"

namespace ORNL
{
    AlgorithmBase::AlgorithmBase(){}

    void AlgorithmBase::execute()
    {
        if(GPU->use() && !m_override_enable_gpu)
        {
            this->executeGPU();
        }
        else
        {
            this->executeCPU();
        }
    }
}

#ifndef ALGORITHM_H
#define ALGORITHM_H

//Qt
#include <QSharedPointer>

namespace ORNL
{
    /*!
     * \class AlgorithmBase
     *
     * \brief Abstract base class for algorithms. Provides a foundation for algorithms to implement in a uniform
     * manor ensure extensability and reusability. It also primarly handles the automatic selection of GPU and CPU implementations
     * based on conpiler and user availiblity.
     */
    class AlgorithmBase
    {
    public:
        //! \brief Constructor. Also fetches our global settings.
        AlgorithmBase();

        //! \brief Executes either a CPU or GPU implementation of an algorithm depending on
        //!         1. If the NVCC compiler is available
        //!         2. If there is a CUDA capable GPU installed in the host system
        //!         3. If the user has enabled it in the settings
        void execute();

        //! \brief Destructor
        virtual ~AlgorithmBase() {}

    protected:

        //! \brief Every algorithm must have a CPU implementation
        virtual void executeCPU() = 0;

        //! \brief Every algorithm may have a GPU implementation
        //! \note In the event there is no GPU implementation then point this function to the CPU
        //! \note All code that calls CUDA functions MUST BE WRAPPED IN:
        //!     #ifdef NVCC_FOUND
        //!     call_to_cuda();
        //!     #endif
        virtual void executeGPU() = 0;

        //! \brief A flag for if we are using the GPU to compute
        bool m_isGPUCompute;

        //! \brief Allows the child class to override using the GPU
        bool m_override_enable_gpu = false;

    };
}

#endif // ALGORITHM_H

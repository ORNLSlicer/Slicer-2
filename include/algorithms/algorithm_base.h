#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <QSharedPointer>

namespace ORNL {
/*!
 * \class AlgorithmBase
 *
 * \brief Abstract base class for algorithms. Provides a foundation for algorithms to implement in a uniform
 * manor ensure extensability and reusability. It also primarly handles the automatic selection of GPU and CPU
 * implementations based on conpiler and user availiblity.
 */
class AlgorithmBase {
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
};
} // namespace ORNL

#endif // ALGORITHM_H

#include "algorithms/algorithm_base.h"

namespace ORNL {
AlgorithmBase::AlgorithmBase() {}

void AlgorithmBase::execute() { this->executeCPU(); }
} // namespace ORNL

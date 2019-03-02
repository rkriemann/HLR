#include "tensor.hh"
#include "tools.hh"

namespace TLR
{

namespace MPI
{

typeid_t  type_dense, type_lr, type_ghost;
double    time_mpi = 0;
int       RANK     = 0;

}// namespace MPI

}// namespace TLR

// Local Variables:
// mode: c++
// End:

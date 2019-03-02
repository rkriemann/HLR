//
// Project     : HLib
// File        : tlr-mpi.cc
// Description : TLR LU using MPI RDMA
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "mpi/matrix.hh"
#include "mpi/arith-rdma.hh"

namespace ARITH = HLR::TLR::MPI::rdma;

#include "tlr-mpi-main.inc"

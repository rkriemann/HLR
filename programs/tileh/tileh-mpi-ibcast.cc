//
// Project     : HLR
// Module      : tileh-mpi-bcast.cc
// Description : Tile-H arithmetic with MPI broadcast
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "hlr/mpi/matrix.hh"
#include "hlr/mpi/arith-ibcast.hh"

namespace          impl      = hlr::mpi::ibcast::tileh;
const std::string  impl_name = "ibcast";

#include "tileh-mpi.hh"

//
// Project     : HLib
// File        : tileh-mpi-bcast.cc
// Description : Tile-H arithmetic with MPI broadcast
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/mpi/matrix.hh"
#include "hlr/mpi/arith-bcast.hh"

namespace          impl      = hlr::mpi::bcast::tileh;
const std::string  impl_name = "bcast";

#include "tileh-mpi.hh"

//
// Project     : HLR
// File        : tlr-seq.cc
// Description : sequential TLR-LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/seq/matrix.hh"
#include "hlr/seq/arith.hh"

namespace          impl      = hlr::seq;
const std::string  impl_name = "seq";

#include "tlr.hh"

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}

//
// Project     : HLR
// Module      : gaspi.hh
// Description : C++ GASPI/GPI wrapper
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "gaspi/gaspi.hh"

namespace HLR
{

namespace GASPI
{

environment::environment ()
{
    GASPI_CHECK_RESULT( gaspi_proc_init, ( GASPI_BLOCK ) );

    //
    // connect to all ranks for further communication (all-to-all)
    //

    HLR::log( 3, "setting up connections" );

    gaspi_rank_t  pid = -1, nprocs;

    GASPI_CHECK_RESULT( gaspi_proc_rank, ( & pid ) );
    GASPI_CHECK_RESULT( gaspi_proc_num,  ( & nprocs ) );
    
    for ( int  p = 0; p < nprocs; ++p )
    {
        GASPI_CHECK_RESULT( gaspi_connect, ( p, GASPI_BLOCK ) );
    }// for
}

environment::~environment ()
{
    HLR::log( 3, "disconnecting" );
    
    for ( int  p = 0; p < nprocs; ++p )
    {
        GASPI_CHECK_RESULT( gaspi_disconnect, ( p, GASPI_BLOCK ) );
    }// for
    
    GASPI_CHECK_RESULT( gaspi_proc_term, ( GASPI_BLOCK ) );
}

}// namespace GASPI

}// namespace HLR


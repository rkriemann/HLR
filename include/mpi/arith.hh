#ifndef __HLR_MPI_ARITH_HH
#define __HLR_MPI_ARITH_HH
//
// Project     : HLib
// File        : arith.hh
// Description : common functions for MPI version
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <matrix/TDenseMatrix.hh>
#include <matrix/TRkMatrix.hh>
#include <matrix/TGhostMatrix.hh>

#include <utils/tensor.hh>

namespace HLR
{

using namespace HLIB;

///////////////////////////////////////////////////////////////////////
//
// common arithmetic functions for tile low-rank format
//
///////////////////////////////////////////////////////////////////////

namespace TLR
{

namespace MPI
{

const typeid_t  TYPE_DENSE = RTTI::type_to_id( "TDenseMatrix" );
const typeid_t  TYPE_LR    = RTTI::type_to_id( "TRkMatrix" );
const typeid_t  TYPE_GHOST = RTTI::type_to_id( "TGhostMatrix" );

//
// create matrix defined by <type>
//
std::unique_ptr< TMatrix >
create_matrix ( const TMatrix *  A,
                const typeid_t   type,
                const int        proc )
{
    assert(( type == TYPE_DENSE ) || ( type == TYPE_LR ));
    
    std::unique_ptr< TMatrix >  T;

    if      ( type == TYPE_DENSE ) T = std::make_unique< TDenseMatrix >( A->row_is(), A->col_is(), A->is_complex() );
    else if ( type == TYPE_LR )    T = std::make_unique< TRkMatrix >( A->row_is(), A->col_is(), A->is_complex() );

    T->set_id( A->id() );
    T->set_procs( ps_single( proc ) );
    
    return T;
}

//
// create n×m block matrix type info
//
tensor2< typeid_t >
build_type_matrix ( const TBlockMatrix *  A )
{
    const auto           nbr = A->nblock_rows();
    const auto           nbc = A->nblock_cols();
    tensor2< typeid_t >  mat_types( nbr, nbc );
    mpi::communicator    world;
    const auto           nprocs = world.size();

    for ( uint  i = 0; i < nbr; ++i )
        for ( uint  j = 0; j < nbc; ++j )
            mat_types(i,j) = A->block( i, j )->type();
    
    for ( uint  p = 0; p < nprocs; ++p )
    {
        tensor2< typeid_t >  rem_types( mat_types );

        world.broadcast( & rem_types(0,0), nbr * nbc, p );
        
        for ( uint  i = 0; i < nbr; ++i )
            for ( uint  j = 0; j < nbc; ++j )
                if ( rem_types(i,j) != TYPE_GHOST )
                    mat_types(i,j) = rem_types(i,j);
    }// for

    // for ( uint  i = 0; i < nbr; ++i )
    // {
    //     for ( uint  j = 0; j < nbc; ++j )
    //         std::cout << RTTI::id_to_type( mat_types(i,j) ) << "  ";
    //     std::cout << std::endl;
    // }// for

    return mat_types;
}

}// namespace MPI

}// namespace TLR

///////////////////////////////////////////////////////////////////////
//
// common arithmetic functions for HODLR format
//
///////////////////////////////////////////////////////////////////////

namespace HODLR
{

namespace MPI
{

}// namespace MPI

}// namespace HODLR

///////////////////////////////////////////////////////////////////////
//
// common arithmetic functions for tile H format
//
///////////////////////////////////////////////////////////////////////

namespace TileH
{

namespace MPI
{

}// namespace MPI

}// namespace TileH

}// namespace HLR

#endif // __HLR_MPI_ARITH_HH

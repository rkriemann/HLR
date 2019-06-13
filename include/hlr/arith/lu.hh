#ifndef __HLR_ARITH_LU_HH
#define __HLR_ARITH_LU_HH
//
// Project     : HLib
// File        : lu.hh
// Description : LU factorization functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <algebra/mat_fac.hh>

#include "hlr/utils/checks.hh"
#include "hlr/matrix/level_matrix.hh"

namespace hlr { namespace arith {

//
// compute LU factorization of A within L
//
void
lu ( TMatrix &               A,
     matrix::level_matrix &  L,
     const TTruncAcc &       acc )
{
    DBG::printf( "lu( %d )", A.id() );

    ///////////////////////////////////////////////////////////////
    //
    // recurse to visit all diagonal blocks
    //
    ///////////////////////////////////////////////////////////////

    if ( is_blocked( A ) )
    {
        auto        B      = ptrcast( & A, TBlockMatrix );
        const uint  nbrows = B->nblock_rows();
        const uint  nbcols = B->nblock_cols();

        for ( uint  i = 0; i < std::min( nbrows, nbcols ); ++i )
        {
            auto  A_ii = B->block( i, i );
            
            assert( ! is_null( A_ii ) );
            
            lu( * A_ii, *( L.below() ), acc );
        }// for
    }// if

    ///////////////////////////////////////////////////////////////
    //
    // actual factorisation of A, solving in block row/column of A
    // and update of trailing sub matrix with respect to A (all on L)
    //
    ///////////////////////////////////////////////////////////////
    
    //
    // get block row/column of A in L
    //

    const auto  [ bi, bj ] = L.get_index( A );
    const auto  nbrows     = L.nblock_rows();
    const auto  nbcols     = L.nblock_cols();

    // check if found in L
    assert(( bi != L.nblock_rows() ) && ( bj != L.nblock_cols() ));
    
    // should be on diagonal
    assert( bi == bj );
    
    //
    // factorise diagonal
    //

    if ( is_leaf( A ) || is_small( A ) )
        HLIB::LU::factorise_rec( & A, acc, fac_options_t( block_wise, store_inverse, false ) );
    
    //
    // off-diagonal solves in current block row/column
    //
    
    for ( uint  j = bi+1; j < nbcols; ++j )
    {
        auto  L_ij = L.block( bi, j );
            
        if ( ! is_null( L_ij ) && is_leaf( L_ij ) )
        {
            DBG::printf( "solve_lower_left( %d, %d )", A.id(), L_ij->id() );

            solve_lower_left( apply_normal, & A, L_ij, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
        }// if
    }// for
        
    for ( uint  j = bi+1; j < nbrows; ++j )
    {
        auto  L_ji = L.block( j, bi );
            
        if ( ! is_null( L_ji ) && is_leaf( L_ji ) )
        {
            DBG::printf( "solve_upper_right( %d, %d )", A.id(), L_ji->id() );

            solve_upper_right( L_ji, & A, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
        }// if
    }// for

    //
    // update of trailing sub matrix
    //

    for ( uint  j = bi+1; j < nbrows; ++j )
    {
        auto  L_ji = L.block( j, bi );
            
        for ( uint  l = bi+1; l < nbcols; ++l )
        {
            auto  L_il = L.block( bi, l );
            auto  L_jl = L.block(  j, l );
            
            if ( ! is_null_any( L_ji, L_il, L_jl ) && is_leaf_any( L_ji, L_il, L_jl ) )
            {
                DBG::printf( "update( %d, %d, %d )", L_ji->id(), L_il->id(), L_jl->id() );
                
                multiply( real(-1),
                          apply_normal, L_ji,
                          apply_normal, L_il,
                          real(1), L_jl, acc );
            }// if
        }// for
    }// for
}

//
// factorize given level matrix
// - only calls actual LU algorithm for all diagonal blocks
//
void
lu ( matrix::level_matrix &  A,
     const TTruncAcc &       acc )
{
    const uint  nbrows = A.nblock_rows();
    const uint  nbcols = A.nblock_cols();

    for ( uint  i = 0; i < std::min( nbrows, nbcols ); ++i )
    {
        auto  A_ii = A.block( i, i );

        assert( ! is_null( A_ii ) );
        
        lu( * A_ii, A, acc );
    }// for
}

}}// namespace hlr::arith

#endif // __HLR_ARITH_LU_HH

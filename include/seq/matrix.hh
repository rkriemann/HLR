#ifndef __HLR_SEQ_MATRIX_HH
#define __HLR_SEQ_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <type_traits>

#include <matrix/TMatrix.hh>
#include <matrix/TBlockMatrix.hh>
#include <matrix/structure.hh>
#include <base/TTruncAcc.hh>

namespace HLR
{

using namespace HLIB;

namespace Matrix
{
    
namespace Seq
{

//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< HLIB::TMatrix >
build ( const HLIB::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const HLIB::TTruncAcc &      acc )
{
    static_assert( std::is_same< typename coeff_t::value_t,
                                 typename lrapx_t::value_t >::value,
                   "coefficient function and low-rank approximation must have equal value type" );
    
    assert( bct != nullptr );
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< TMatrix >  M;
    
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< TMatrix >( lrapx.build( bct, acc ) );
        }// if
        else
        {
            M = coeff.build( bct->is().row_is(), bct->is().col_is() );
        }// else
    }// if
    else
    {
        M = std::make_unique< TBlockMatrix >( bct );
        
        auto  B = ptrcast( M.get(), TBlockMatrix );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( bct->son( i, j ) != nullptr )
                {
                    auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
    }// else

    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

//
// return copy of matrix
//
std::unique_ptr< TMatrix >
copy ( const TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, TBlockMatrix );
        auto  N  = std::make_unique< TBlockMatrix >();
        auto  B  = ptrcast( N.get(), TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else
    {
        // assuming non-structured block
        return M.copy();
    }// else
}

}// namespace Seq

}// namespace Matrix

}// namespace HLR

#endif // __HLR_SEQ_MATRIX_HH

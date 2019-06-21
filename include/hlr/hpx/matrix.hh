#ifndef __HLR_HPX_MATRIX_HH
#define __HLR_HPX_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <type_traits>

#include <hpx/parallel/task_block.hpp>

#include <matrix/TMatrix.hh>
#include <matrix/TBlockMatrix.hh>
#include <base/TTruncAcc.hh>

#include "hlr/seq/matrix.hh"

namespace hlr
{
    
namespace hpx
{

namespace matrix
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

    auto        M     = std::unique_ptr< TMatrix >();
    const auto  rowis = bct->is().row_is();
    const auto  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return seq::matrix::build( bct, coeff, lrapx, acc );
        
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M.reset( lrapx.build( bct, acc ) );
        }// if
        else
        {
            M = coeff.build( rowis, colis );
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
        ::hpx::parallel::v2::define_task_block(
            [&,B,bct] ( auto &  tb )
            {
                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < B->nblock_rows(); ++j )
                    {
                        if ( bct->son( i, j ) != nullptr )
                        {
                            tb.run( [bct,i,j,B,&coeff,&lrapx,&acc]
                                    {
                                        auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc );

                                        B->set_block( i, j, B_ij.release() );
                                    } );
                        }// if
                    }// for
                }// for
            } );
    }// else

    // copy properties from the cluster
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

//
// return copy of matrix
// - copy operation is performed in parallel for sub blocks
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
        
        ::hpx::parallel::v2::define_task_block(
            [B,BM] ( auto &  tb )
            {
                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < B->nblock_rows(); ++j )
                    {
                        if ( BM->block( i, j ) != nullptr )
                        {
                            tb.run( [BM,B,i,j]
                                    {
                                        auto  B_ij = copy( * BM->block( i, j ) );
                            
                                        B_ij->set_parent( B );
                                        B->set_block( i, j, B_ij.release() );
                                    } );
                        }// if
                    }// for
                }// for
            } );

        return N;
    }// if
    else
    {
        // assuming non-structured block and hence no parallel copy needed
        return M.copy();
    }// else
}

//
// reallocate matrix blocks
// - frees old data
// - local operation thereby limiting extra memory usage
//
std::unique_ptr< TMatrix >
realloc ( TMatrix *  A )
{
    if ( is_null( A ) )
        return nullptr;
    
    if ( is_blocked( A ) )
    {
        auto  B  = ptrcast( A, TBlockMatrix );
        auto  C  = B->create();
        auto  BC = ptrcast( C.get(), TBlockMatrix );

        C->copy_struct_from( B );

        ::hpx::parallel::v2::define_task_block(
            [B,BC] ( auto &  tb )
            {
                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < B->nblock_cols(); ++j )
                    {
                        tb.run( [BC,B,i,j]
                                {
                                    auto  C_ij = realloc( B->block( i, j ) );
                        
                                    BC->set_block( i, j, C_ij.release() );
                                    B->set_block( i, j, nullptr );
                                } );
                    }// for
                }// for
            } );

        delete B;

        return C;
    }// if
    else
    {
        auto  C = A->copy();

        delete A;

        return C;
    }// else
}

}// namespace matrix

}// namespace hpx

}// namespace hlr

#endif // __HLR_HPX_MATRIX_HH

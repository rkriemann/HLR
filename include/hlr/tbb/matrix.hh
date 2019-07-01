#ifndef __HLR_TBB_MATRIX_HH
#define __HLR_TBB_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <type_traits>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <matrix/TMatrix.hh>
#include <matrix/TBlockMatrix.hh>
#include <matrix/structure.hh>
#include <base/TTruncAcc.hh>

#include "hlr/seq/matrix.hh"

namespace hlr
{

using namespace HLIB;
    
namespace tbb
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
std::unique_ptr< TMatrix >
build ( const TBlockCluster *  bct,
        const coeff_t &        coeff,
        const lrapx_t &        lrapx,
        const TTruncAcc &      acc )
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
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,bct] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( bct->son( i, j ) != nullptr )
                        {
                            auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc );
                            
                            B->set_block( i, j, B_ij.release() );
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
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [B,BM] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( BM->block( i, j ) != nullptr )
                        {
                            auto  B_ij = copy( * BM->block( i, j ) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
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
// copy data of A to matrix B
// - ASSUMPTION: identical matrix structure
//
void
copy_to ( const TMatrix &  A,
          TMatrix &        B )
{
    assert( A.type()     == B.type() );
    assert( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, TBlockMatrix );
        auto  BB = ptrcast(  &B, TBlockMatrix );

        assert( BA->nblock_rows() == BB->nblock_rows() );
        assert( BA->nblock_cols() == BB->nblock_cols() );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, BA->nblock_rows(),
                                            0, BA->nblock_cols() ),
            [BA,BB] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( BA->block( i, j ) != nullptr )
                        {
                            assert( ! is_null( BB->block( i, j ) ) );
                            
                            copy_to( * BA->block( i, j ), * BB->block( i, j ) );
                        }// if
                    }// for
                }// for
            } );
    }// if
    else
    {
        A.copy_to( & B );
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
        auto  C  = std::make_unique< TBlockMatrix >();
        auto  BC = ptrcast( C.get(), TBlockMatrix );

        C->copy_struct_from( B );

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [B,BC] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  C_ij = realloc( B->block( i, j ) );

                        BC->set_block( i, j, C_ij.release() );
                        B->set_block( i, j, nullptr );
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

}// namespace tbb

}// namespace hlr

#endif // __HLR_TBB_MATRIX_HH

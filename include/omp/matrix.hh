#ifndef __HLR_OMP_MATRIX_HH
#define __HLR_OMP_MATRIX_HH
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
#include <base/TTruncAcc.hh>

#include "seq/matrix.hh"

namespace HLR
{

namespace Matrix
{
    
namespace OMP
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
build_task ( const HLIB::TBlockCluster *  bct,
             const coeff_t &              coeff,
             const lrapx_t &              lrapx,
             const HLIB::TTruncAcc &      acc )
{
    static_assert( std::is_same< typename coeff_t::value_t,
                   typename lrapx_t::value_t >::value,
                   "coefficient function and low-rank approximation must have equal value type" );

    assert( bct != nullptr );

    using  value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto        M     = std::unique_ptr< TMatrix >();
    const auto  rowis = bct->is().row_is();
    const auto  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return Matrix::Seq::build( bct, coeff, lrapx, acc );
        
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
        #pragma omp taskgroup
        {
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( bct->son( i, j ) != nullptr )
                    {
                        #pragma omp task
                        {
                            auto  B_ij = build_task( bct->son( i, j ), coeff, lrapx, acc );
                    
                            B->set_block( i, j, B_ij.release() );
                        }// omp task
                    }// if
                }// for
            }// for
        }// omp taskgroup

        // wait for child tasks
        // #pragma omp taskwait
    }// else

    // copy properties from the cluster
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< HLIB::TMatrix >
build ( const HLIB::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const HLIB::TTruncAcc &      acc )
{
    std::unique_ptr< TMatrix >  res;

    // spawn parallel region for tasks
    #pragma omp parallel
    {
        #pragma omp task
        {
            res = build_task( bct, coeff, lrapx, acc );
        }// omp task
    }// omp parallel

    return res;
}

//
// return copy of matrix
// - copy operation is performed in parallel for sub blocks
//
std::unique_ptr< TMatrix >
copy_task ( const TMatrix &  M )
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
                    #pragma omp task
                    {
                        auto  B_ij = copy_task( * BM->block( i, j ) );
                            
                        B_ij->set_parent( B );
                        B->set_block( i, j, B_ij.release() );
                    }// omp task
                }// if
            }// for
        }// for

        #pragma omp taskwait
        
        return N;
    }// if
    else
    {
        // assuming non-structured block and hence no parallel copy needed
        return M.copy();
    }// else
}

std::unique_ptr< TMatrix >
copy ( const TMatrix &  M )
{
    std::unique_ptr< TMatrix >  res;

    // spawn parallel region for tasks
    #pragma omp parallel
    {
        #pragma omp task
        {
            res = copy_task( M );
        }// omp task
    }// omp parallel

    return res;
}

}// namespace OMP

}// namespace Matrix

}// namespace HLR

#endif // __HLR_OMP_MATRIX_HH

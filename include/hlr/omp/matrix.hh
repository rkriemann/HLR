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

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/base/TTruncAcc.hh>

#include "hlr/seq/matrix.hh"

namespace hlr
{

namespace omp
{

namespace matrix
{

//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
namespace detail
{

template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< HLIB::TMatrix >
build_task ( const HLIB::TBlockCluster *  bct,
             const coeff_t &              coeff,
             const lrapx_t &              lrapx,
             const HLIB::TTruncAcc &      acc,
             const size_t                 nseq )
{
    static_assert( std::is_same< typename coeff_t::value_t,
                   typename lrapx_t::value_t >::value,
                   "coefficient function and low-rank approximation must have equal value type" );

    assert( bct != nullptr );

    //
    // decide upon cluster type, how to construct matrix
    //

    auto        M     = std::unique_ptr< HLIB::TMatrix >();
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
    else if ( std::min( rowis.size(), colis.size() ) <= nseq )
    {
        M = hlr::seq::matrix::build( bct, coeff, lrapx, acc, nseq );
    }// if
    else
    {
        M = std::make_unique< HLIB::TBlockMatrix >( bct );
        
        auto  B = ptrcast( M.get(), HLIB::TBlockMatrix );

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
                            auto  B_ij = build_task( bct->son( i, j ), coeff, lrapx, acc, nseq );
                    
                            B->set_block( i, j, B_ij.release() );
                        }// omp task
                    }// if
                }// for
            }// for
        }// omp taskgroup

        // wait for child tasks
        // #pragma omp taskwait

        // make value type consistent in block matrix and sub blocks
        B->adjust_value_type();
    }// else

    // copy properties from the cluster
    M->set_cluster_force( bct );
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

}// namespace detail

template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< HLIB::TMatrix >
build ( const HLIB::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const HLIB::TTruncAcc &      acc,
        const size_t                 nseq = hpro::CFG::Arith::max_seq_size )
{
    std::unique_ptr< HLIB::TMatrix >  res;

    // spawn parallel region for tasks
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                res = detail::build_task( bct, coeff, lrapx, acc, nseq );
            }// omp task
        }// omp single
    }// omp parallel

    return res;
}

//
// return copy of matrix
// - copy operation is performed in parallel for sub blocks
//
namespace detail
{

std::unique_ptr< HLIB::TMatrix >
copy_task ( const HLIB::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, HLIB::TBlockMatrix );
        auto  N  = std::make_unique< HLIB::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), HLIB::TBlockMatrix );

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

}// namespace detail

std::unique_ptr< HLIB::TMatrix >
copy ( const HLIB::TMatrix &  M )
{
    std::unique_ptr< HLIB::TMatrix >  res;

    // spawn parallel region for tasks
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                res = detail::copy_task( M );
            }// omp task
        }// omp single
    }// omp parallel

    return res;
}

//
// copy data of A to matrix B
// - ASSUMPTION: identical matrix structure
//
namespace detail
{

void
copy_to_task ( const HLIB::TMatrix &  A,
               HLIB::TMatrix &        B )
{
    assert( A.type()     == B.type() );
    assert( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, HLIB::TBlockMatrix );
        auto  BB = ptrcast(  &B, HLIB::TBlockMatrix );

        assert( BA->nblock_rows() == BB->nblock_rows() );
        assert( BA->nblock_cols() == BB->nblock_cols() );
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( BA->block( i, j ) != nullptr )
                {
                    #pragma omp task
                    {
                        assert( ! is_null( BB->block( i, j ) ) );

                        copy_to_task( * BA->block( i, j ), * BB->block( i, j ) );
                    }// omp task
                }// if
            }// for
        }// for
    }// if
    else
    {
        A.copy_to( & B );
    }// else
}

}// namespace detail

void
copy_to ( const HLIB::TMatrix &  A,
          HLIB::TMatrix &        B )
{
    // spawn parallel region for tasks
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                detail::copy_to_task( A, B );
            }// omp task
        }// omp single
    }// omp parallel
}

//
// reallocate matrix blocks
// - frees old data
// - local operation thereby limiting extra memory usage
//
namespace detail
{

std::unique_ptr< HLIB::TMatrix >
realloc_task ( HLIB::TMatrix *  A )
{
    if ( is_null( A ) )
        return nullptr;
    
    if ( is_blocked( A ) )
    {
        auto  B  = ptrcast( A, HLIB::TBlockMatrix );
        auto  C  = std::make_unique< HLIB::TBlockMatrix >();
        auto  BC = ptrcast( C.get(), HLIB::TBlockMatrix );

        C->copy_struct_from( B );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                #pragma omp task
                {
                    auto  C_ij = realloc_task( B->block( i, j ) );
                    
                    BC->set_block( i, j, C_ij.release() );
                    B->set_block( i, j, nullptr );
                }
            }// for
        }// for

        #pragma omp taskwait
        
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

}// namespace detail

std::unique_ptr< HLIB::TMatrix >
realloc ( HLIB::TMatrix *  A )
{
    std::unique_ptr< HLIB::TMatrix >  res;

    // spawn parallel region for tasks
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                res = detail::realloc_task( A );
            }// omp task
        }// omp single
    }// omp parallel

    return res;
}

}// namespace matrix

}// namespace omp

}// namespace hlr

#endif // __HLR_OMP_MATRIX_HH

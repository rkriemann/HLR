#ifndef __HLR_OMP_MATRIX_COMPRESS_HH
#define __HLR_OMP_MATRIX_COMPRESS_HH
//
// Project     : HLR
// Module      : omp/compress
// Description : matrix compression functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/lrsmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/tensor/structured_tensor.hh>
#include <hlr/utils/tensor.hh>
#include <hlr/approx/aca.hh>
#include <hlr/approx/accuracy.hh>

#include <hlr/omp/detail/compress.hh>

namespace hlr { namespace omp {

namespace matrix
{

//
// build H-matrix from given dense matrix without reording rows/columns
// starting lowrank approximation at blocks of size ntile × ntile and
// then trying to agglomorate low-rank blocks up to the root
//
template < typename value_t,
           typename approx_t,
           typename zconfig_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
compress ( const indexset &                 rowis,
           const indexset &                 colis,
           const blas::matrix< value_t > &  D,
           const accuracy &                 acc,
           const approx_t &                 approx,
           const size_t                     ntile,
           const zconfig_t *                zconf = nullptr )
{
    using namespace hlr::matrix;

    auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();

    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    M = detail::compress( rowis, colis, D, acc, approx, ntile, zconf );

    HLR_ASSERT( ! is_null( M ) );

    //
    // handle ZFP compression for global lowrank/dense case
    //
    
    if ( ! is_null( zconf ) )
    {
        if ( is_compressible_lowrank( *M ) )
            ptrcast( M.get(), lrmatrix< value_t > )->compress( *zconf );
                
        if ( is_compressible_dense( *M ) )
            ptrcast( M.get(), dense_matrix< value_t > )->compress( *zconf );
    }// if

    return M;
}
    
//
// top-down compression approach: if low-rank approximation is possible
// within given accuracy and maximal rank, stop recursion
//
template < typename value_t,
           typename approx_t,
           typename zconfig_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
compress_topdown ( const indexset &                 rowis,
                   const indexset &                 colis,
                   const blas::matrix< value_t > &  D,
                   const accuracy &                 acc,
                   const approx_t &                 approx,
                   const size_t                     ntile,
                   const size_t                     max_rank,
                   const zconfig_t *                zconf = nullptr )
{
    using namespace hlr::matrix;

    auto  M = detail::compress_topdown( rowis, colis, D, acc, approx, ntile, max_rank, zconf );

    HLR_ASSERT( ! is_null( M ) );

    //
    // handle SZ/ZFP compression for global dense case
    //
    
    if ( ! is_null( zconf ) )
    {
        if ( is_compressible_dense( *M ) )
            ptrcast( M.get(), dense_matrix< value_t > )->compress( *zconf );
    }// if

    return M;
}

template < typename value_t,
           typename approx_t,
           typename zconfig_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
compress_topdown_orig ( const indexset &                 rowis,
                        const indexset &                 colis,
                        const blas::matrix< value_t > &  D,
                        const accuracy &                 acc,
                        const approx_t &                 approx,
                        const size_t                     ntile,
                        const size_t                     max_rank,
                        const zconfig_t *                zconf = nullptr )
{
    using namespace hlr::matrix;

    auto  M = detail::compress_topdown_orig( rowis, colis, D, acc, approx, ntile, max_rank, zconf );

    HLR_ASSERT( ! is_null( M ) );

    //
    // handle SZ/ZFP compression for global dense case
    //
    
    if ( ! is_null( zconf ) )
    {
        if ( is_compressible_dense( *M ) )
            ptrcast( M.get(), dense_matrix< value_t > )->compress( *zconf );
    }// if

    return M;
}

//
// multi-level compression with contributions from all levels
//
template < typename value_t,
           typename approx_t,
           typename zconfig_t >
void
compress_ml ( const indexset &           rowis,
              const indexset &           colis,
              blas::matrix< value_t > &  D,
              size_t &                   csize,
              const size_t               lvl_rank,
              const accuracy &           acc,
              const approx_t &           approx,
              const size_t               ntile,
              const zconfig_t *          zconf = nullptr )
{
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    detail::compress_ml< value_t, approx_t, zconfig_t >( rowis, colis, D, csize, lvl_rank, acc, approx, ntile, zconf );
}

//
// compress compressible sub-blocks within H-matrix
//
template < typename value_t >
void
compress ( Hpro::TMatrix< value_t > &  A,
           const accuracy &            acc )
{
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    detail::compress( A, acc );
}

//
// compress cluster basis data
//
template < typename value_t,
           typename cluster_basis_t >
void
compress ( cluster_basis_t &  cb,
           const accuracy &   acc )
{
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    detail::compress( cb, acc );
}

//
// decompress H-matrix
//
template < typename value_t >
void
decompress ( Hpro::TMatrix< value_t > &  A )
{
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    detail::decompress( A );
}

//
// decompress cluster basis data
//
template < typename value_t,
           typename cluster_basis_t >
void
decompress ( cluster_basis_t &  cb )
{
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    detail::decompress( cb );
}

}// namespace matrix

namespace tensor
{

using namespace hlr::tensor;

//
// compress/decompress compressible sub-blocks within H-tensor
//
template < typename value_t >
void
compress ( tensor::base_tensor3< value_t > &  A,
           const accuracy &                   acc )
{
    using namespace hlr::tensor;

    if ( is_structured( A ) )
    {
        auto  BA = ptrcast( &A, structured_tensor3< value_t > );
        
        #pragma omp taskloop collapse(3) default(shared)
        for ( uint  l = 0; l < BA->nblocks(0); ++l )
        {
            for ( uint  j = 0; j < BA->nblocks(1); ++j )
            {
                for ( uint  i = 0; i < BA->nblocks(2); ++i )
                {
                    if ( is_null( BA->block( i, j, l ) ) )
                        continue;
                    
                    compress( *BA->block( i, j, l ), acc );
                }// for
            }// for
        }// for
    }// if
    else if ( compress::is_compressible( A ) )
    {
        dynamic_cast< compress::compressible * >( &A )->compress( acc );
    }// if
}

template < typename value_t >
void
decompress ( tensor::base_tensor3< value_t > &  A )
{
    using namespace hlr::tensor;

    if ( is_structured( A ) )
    {
        auto  BA = ptrcast( &A, structured_tensor3< value_t > );
        
        #pragma omp taskloop collapse(3) default(shared)
        for ( uint  l = 0; l < BA->nblocks(0); ++l )
        {
            for ( uint  j = 0; j < BA->nblocks(1); ++j )
            {
                for ( uint  i = 0; i < BA->nblocks(2); ++i )
                {
                    if ( is_null( BA->block( i, j, l ) ) )
                        continue;
                    
                    decompress( *BA->block( i, j, l ) );
                }// for
            }// for
        }// for
    }// if
    else if ( compress::is_compressible( A ) )
    {
        dynamic_cast< compress::compressible * >( &A )->decompress();
    }// if
}

}// namespace tensor

}}// namespace hlr::omp

#endif // __HLR_OMP_MATRIX_COMPRESS_HH

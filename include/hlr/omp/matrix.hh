#ifndef __HLR_OMP_MATRIX_HH
#define __HLR_OMP_MATRIX_HH
//
// Project     : HLR
// Module      : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cassert>
#include <type_traits>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/base/TTruncAcc.hh>

#include "hlr/seq/matrix.hh"

#include "hlr/omp/detail/matrix.hh"

namespace hlr { namespace omp { namespace matrix {

using namespace hlr::matrix;

//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build ( const Hpro::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const Hpro::TTruncAcc &      acc,
        const size_t                 nseq = Hpro::CFG::Arith::max_seq_size )
{
    using  value_t = typename coeff_t::value_t;

    auto  res = std::unique_ptr< Hpro::TMatrix< value_t > >();

    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    res = detail::build_task( bct, coeff, lrapx, acc, nseq );

    return res;
}

//
// same as above but use compressable matrix types,
// e.g., dense_matrix and lrmatrix, and directly compress
// matrix data
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build_compressed ( const Hpro::TBlockCluster *  bct,
                   const coeff_t &              coeff,
                   const lrapx_t &              lrapx,
                   const Hpro::TTruncAcc &      acc,
                   const size_t                 nseq = Hpro::CFG::Arith::max_seq_size )
{
    using  value_t = typename coeff_t::value_t;

    auto  res = std::unique_ptr< Hpro::TMatrix< value_t > >();

    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    res = detail::build_compressed( bct, coeff, lrapx, acc, nseq );

    return res;
}

//
// use lrsvmatrix instead of lrmatrix
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build_mixedprec ( const Hpro::TBlockCluster *  bct,
                  const coeff_t &              coeff,
                  const lrapx_t &              lrapx,
                  const Hpro::TTruncAcc &      acc,
                  const size_t                 nseq = Hpro::CFG::Arith::max_seq_size )
{
    using  value_t = typename coeff_t::value_t;

    auto  res = std::unique_ptr< Hpro::TMatrix< value_t > >();

    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    res = detail::build_mixedprec( bct, coeff, lrapx, acc, nseq );

    return res;
}

//
// build representation of nearfield of dense matrix with
// matrix structure defined by <bct>, matrix coefficients
// defined by <coeff>
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build_nearfield ( const Hpro::TBlockCluster *  bct,
                  const coeff_t &              coeff,
                  const size_t                 nseq = Hpro::CFG::Arith::max_seq_size )
{
    using  value_t = typename coeff_t::value_t;

    auto  res = std::unique_ptr< Hpro::TMatrix< value_t > >();

    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    res = detail::build_nearfield( bct, coeff, nseq );

    return res;
}

//
// build representation of sparse matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <M>. If low rank blocks are not zero, they are
// truncated to accuracy <acc> using approximation method <apx>
//
template < typename value_t,
           typename approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build ( const Hpro::TBlockCluster &             bct,
        const Hpro::TSparseMatrix< value_t > &  S,
        const Hpro::TTruncAcc &                 acc,
        const approx_t &                        apx,
        const size_t                            nseq = Hpro::CFG::Arith::max_seq_size ) // ignored
{
    auto  res = std::unique_ptr< Hpro::TMatrix< value_t > >();

    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    res = detail::build( bct, S, acc, apx, nseq );

    return res;
}

template < typename value_t,
           typename approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_nd ( const Hpro::TBlockCluster &             bct,
           const Hpro::TSparseMatrix< value_t > &  S,
           const Hpro::TTruncAcc &                 acc,
           const approx_t &                        apx,
           const size_t                            nseq = Hpro::CFG::Arith::max_seq_size ) // ignored
{
    auto  res = std::unique_ptr< Hpro::TMatrix< value_t > >();

    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    res = detail::build_nd( bct, S, acc, apx, nseq );

    return res;
}

//
// same as above but use sparse matrices instead of dense matrices
// for non-admissible blocks
//
template < typename value_t,
           typename approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_sparse ( const Hpro::TBlockCluster &             bct,
               const Hpro::TSparseMatrix< value_t > &  S,
               const Hpro::TTruncAcc &                 acc,
               const approx_t &                        apx,
               const size_t                            nseq = Hpro::CFG::Arith::max_seq_size )
{
    auto  res = std::unique_ptr< Hpro::TMatrix< value_t > >();

    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    res = detail::build_sparse( bct, S, acc, apx, nseq );

    return res;
}

#if 0
//
// build representation of dense matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and low-rank blocks computed by <lrapx>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//
template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< value_t > > >
build_uniform_lvl ( const Hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const Hpro::TTruncAcc &      acc,
                    const size_t                 /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
{
    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

    auto  rowcb = std::unique_ptr< cluster_basis >();
    auto  colcb = std::unique_ptr< cluster_basis >();
    auto  M     = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    {
        std::tie( rowcb, colcb, M ) = detail::build_uniform_lvl( bct, coeff, lrapx, basisapx, acc );
    }// omp task

    return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}

//
// build representation of dense matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and low-rank blocks computed by <lrapx>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//
template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< value_t > > >
build_uniform_rec ( const Hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const Hpro::TTruncAcc &      acc,
                    const size_t                 /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< typename coeff_t::value_t, typename basisapx_t::value_t >,
                   "coefficient function and basis approximation must have equal value type" );
    
    HLR_ASSERT( ! is_null( bct ) );

    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

    auto  rowcb  = std::make_unique< cluster_basis >( bct->is().row_is() );
    auto  colcb  = std::make_unique< cluster_basis >( bct->is().col_is() );

    rowcb->set_nsons( bct->rowcl()->nsons() );
    colcb->set_nsons( bct->colcl()->nsons() );

    detail::init_cluster_bases( bct, *rowcb, *colcb );
    
    auto  basis_data = detail::rec_basis_data_t();
    auto  M          = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    {
        M = std::move( detail::build_uniform_rec( bct, coeff, lrapx, basisapx, acc, *rowcb, *colcb, basis_data ) );
    }// omp task
    
    return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}

//
// build uniform-H version from given H-matrix <A>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//
template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< value_t > > >
build_uniform_rec ( const Hpro::TMatrix< value_t > &    A,
                    const basisapx_t &       basisapx,
                    const Hpro::TTruncAcc &  acc,
                    const size_t             /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;

    auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
    auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

    if ( is_blocked( A ) )
    {
        rowcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
        colcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
    }// if

    detail::init_cluster_bases( A, *rowcb, *colcb );
    
    auto  basis_data = detail::rec_basis_data_t();
    auto  M          = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    {
        M = std::move( detail::build_uniform_rec( A, basisapx, acc, *rowcb, *colcb, basis_data ) );
    }// omp task

    return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}

#endif

//
// assign block cluster to matrix
//
template < typename value_t >
void
assign_cluster ( Hpro::TMatrix< value_t > &   M,
                 const Hpro::TBlockCluster &  bc )
{
    hlr::seq::matrix::assign_cluster( M, bc );
}

//
// return copy of matrix
// - copy operation is performed in parallel for sub blocks
//
namespace detail
{

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_task ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );

        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2)
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( ! is_null( BM->block( i, j ) ) )
                    {
                        auto  B_ij = copy_task( * BM->block( i, j ) );
                            
                        B_ij->set_parent( B );
                        B->set_block( i, j, B_ij.release() );
                    }// if
                }// for
            }// omp taskloop for
        }// omp taskgroup

        return N;
    }// if
    else
    {
        // assuming non-structured block and hence no parallel copy needed
        return M.copy();
    }// else
}

}// namespace detail

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy ( const Hpro::TMatrix< value_t > &  M )
{
    std::unique_ptr< Hpro::TMatrix< value_t > >  res;

    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    res = detail::copy_task( M );

    return res;
}

//
// copy data of A to matrix B
// - ASSUMPTION: identical matrix structure
//
namespace detail
{

template < typename value_t >
void
copy_to_task ( const Hpro::TMatrix< value_t > &  A,
               Hpro::TMatrix< value_t > &        B )
{
    HLR_ASSERT( A.type()     == B.type() );
    HLR_ASSERT( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  BB = ptrcast(  &B, Hpro::TBlockMatrix< value_t > );

        HLR_ASSERT( BA->nblock_rows() == BB->nblock_rows() );
        HLR_ASSERT( BA->nblock_cols() == BB->nblock_cols() );

        #pragma omp taskloop collapse(2)
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    HLR_ASSERT( ! is_null( BB->block( i, j ) ) );

                    copy_to_task( * BA->block( i, j ), * BB->block( i, j ) );
                }// if
            }// for
        }// omp taskloop for
    }// if
    else
    {
        A.copy_to( & B );
    }// else
}

}// namespace detail

template < typename value_t >
void
copy_to ( const Hpro::TMatrix< value_t > &  A,
          Hpro::TMatrix< value_t > &        B )
{
    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    detail::copy_to_task( A, B );
}

//
// reallocate matrix blocks
// - frees old data
// - local operation thereby limiting extra memory usage
//
namespace detail
{

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
realloc_task ( Hpro::TMatrix< value_t > *  A )
{
    if ( is_null( A ) )
        return nullptr;
    
    if ( is_blocked( A ) )
    {
        auto  B  = ptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto  C  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  BC = ptrcast( C.get(), Hpro::TBlockMatrix< value_t > );

        C->copy_struct_from( B );

        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2)
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    auto  C_ij = realloc_task( B->block( i, j ) );
                    
                    BC->set_block( i, j, C_ij.release() );
                    B->set_block( i, j, nullptr );
                }// for
            }// for
        }// omp taskgroup
        
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

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
realloc ( Hpro::TMatrix< value_t > *  A )
{
    std::unique_ptr< Hpro::TMatrix< value_t > >  res;

    // spawn parallel region for tasks
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    res = detail::realloc_task( A );

    return res;
}

//
// return copy of matrix with uniform low-rank matrices
// - TODO: add cluster basis as template argument to allow
//         different bases
//
#if 0
namespace detail
{

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_uniform_task ( const Hpro::TMatrix< value_t > &         M,
                    hlr::matrix::cluster_basis< value_t > &  rowcb,
                    hlr::matrix::cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );

        HLR_ASSERT( B->nblock_rows() == rowcb.nsons() );
        HLR_ASSERT( B->nblock_cols() == colcb.nsons() );

        #pragma omp taskloop collapse(2) default(shared) firstprivate(B,BM)
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = copy_uniform_task( * BM->block( i, j ), * rowcb.son(i), * colcb.son(j) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// omp taskloop for
        
        return N;
    }// if
    else if ( is_lowrank( M ) )
    {
        //
        // project into row/column cluster basis:
        //
        //   M = A·B^H = (V·V^H·A) (U·U^H·B)^H
        //             = U · (U^H·A)·(V^H·B)^H · V^H
        //             = U · S · V^H   with  S = (U^H·A)·(V^H·B)^H

        auto  R  = cptrcast( &M, Hpro::TRkMatrix< value_t > );

        auto  UA = rowcb.transform_forward( Hpro::blas_mat_A< value_t >( R ) );
        auto  VB = colcb.transform_forward( Hpro::blas_mat_B< value_t >( R ) );
        auto  S  = blas::prod( value_t(1), UA, blas::adjoint( VB ) );
        
        return std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( M.row_is(), M.col_is(),
                                                                             rowcb, colcb,
                                                                             std::move( S ) );
    }// if
    else
    {
        // assuming dense block (no low-rank)
        return M.copy();
    }// else
}

}// namespace detail

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_uniform ( const Hpro::TMatrix< value_t > &         M,
               hlr::matrix::cluster_basis< value_t > &  rowcb,
               hlr::matrix::cluster_basis< value_t > &  colcb )
{
    auto  T = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    {
        T = detail::copy_uniform_task( M, rowcb, colcb );
    }// omp task

    return T;
}

#endif

//
// return compressible version of M
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_compressible ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        #pragma omp taskloop collapse(2) default(shared) firstprivate(B,BM)
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = copy_compressible( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        auto  N = M.copy();
        
        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  N = M.copy();
        
        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( Hpro::is_dense( M ) )
    {
        auto  D  = cptrcast( &M, Hpro::TDenseMatrix< value_t > );
        auto  DD = blas::copy( blas::mat( D ) );
        auto  N  = std::make_unique< matrix::dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( DD ) );

        N->set_id( M.id() );
        
        return N;
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );

    return 0;
}

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_mixedprec ( const Hpro::TMatrix< value_t > &  M )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        #pragma omp taskloop collapse(2) default(shared) firstprivate(B,BM)
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = copy_mixedprec( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::lrmatrix< value_t > );
        auto  U = blas::copy( R->U() );
        auto  V = blas::copy( R->V() );
        auto  N = std::make_unique< matrix::lrsvmatrix< value_t > >( R->row_is(), R->col_is(), std::move( U ), std::move( V ) );
            
        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  D  = cptrcast( &M, matrix::dense_matrix< value_t > );
        auto  DD = blas::copy( D->mat() );
        auto  N  = std::make_unique< matrix::dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( DD ) );
        
        N->set_id( M.id() );
        
        return N;
    }// if
    else
        return M.copy();
}

//
// return copy of matrix
// - copy operation is performed in parallel for sub blocks
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_nearfield ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        #pragma omp taskloop collapse(2) default(shared) firstprivate(B,BM)
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = copy_nearfield( * BM->block( i, j ) );
                    
                    if ( ! is_null( B_ij.get() ) )
                    {
                        B_ij->set_parent( B );
                        B->set_block( i, j, B_ij.release() );
                    }// if
                }// if
            }// for
        }// for

        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_dense( M ) )
    {
        // assuming non-structured block
        auto  N = M.copy();

        N->set_id( M.id() );
        
        return N;
    }// else
    else
        return nullptr;
}

//
// return coarsend copy of matrix, i.e., try to convert dense blocks to 
// lowrank format and merge lowrank siblings; keep new lowrank blocks
// if more memory efficient
//
template < typename                    value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
coarsen ( const Hpro::TMatrix< value_t > &  M,
          const accuracy &                  acc,
          const approx_t &                  approx )
{
    if ( is_blocked( M ) )
    {
        auto  BM          = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N           = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B           = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );
        auto  all_lowrank = std::atomic< bool >( true );
        uint  k_sum       = 0;

        B->copy_struct_from( BM );

        #pragma omp taskloop collapse(2) default(shared) firstprivate(B,BM)
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = coarsen( * BM->block( i, j ), acc, approx );
                    
                    if ( matrix::is_lowrank( *B_ij ) )
                        k_sum += cptrcast( B_ij.get(), matrix::lrmatrix< value_t > )->rank();
                    else
                        all_lowrank = false;
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        if ( all_lowrank )
        {
            auto    U_sum = blas::matrix< value_t >( M.nrows(), k_sum );
            auto    V_sum = blas::matrix< value_t >( M.ncols(), k_sum );
            uint    pos   = 0;
            
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    auto  R_ij = cptrcast( B->block( i, j ), matrix::lrmatrix< value_t > );

                    if ( is_null( R_ij ) )
                        continue;

                    auto  RU   = R_ij->U();
                    auto  RV   = R_ij->V();
                    auto  U_i  = blas::matrix< value_t >( U_sum, R_ij->row_is() - M.row_ofs(), blas::range( pos, pos + R_ij->rank() - 1 ) );
                    auto  V_j  = blas::matrix< value_t >( V_sum, R_ij->col_is() - M.col_ofs(), blas::range( pos, pos + R_ij->rank() - 1 ) );

                    blas::copy( RU, U_i );
                    blas::copy( RV, V_j );
                    pos += R_ij->rank();
                }// for
            }// for

            auto  [ U, V ] = approx( U_sum, V_sum, acc );
            auto  R        = std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );

            if ( R->byte_size() <= B->byte_size() )
                return R;
        }// if
        
        return N;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        return M.copy();
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  R = convert_to_lowrank( M, acc, approx );

        if ( R->byte_size() <= M.byte_size() )
            return R;

        return M.copy();
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

}}}// namespace hlr::omp::matrix

#endif // __HLR_OMP_MATRIX_HH

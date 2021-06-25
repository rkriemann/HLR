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

#include "hlr/omp/detail/matrix.hh"

namespace hlr { namespace omp { namespace matrix {

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

    HLR_ASSERT( bct != nullptr );

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
            #pragma omp taskloop collapse(2) default(shared)
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( bct->son( i, j ) != nullptr )
                    {
                        auto  B_ij = build_task( bct->son( i, j ), coeff, lrapx, acc, nseq );
                        
                        B->set_block( i, j, B_ij.release() );
                    }// if
                }// for
            }// for
        }// omp taskgroup

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
    #pragma omp single
    #pragma omp task
    res = detail::build_task( bct, coeff, lrapx, acc, nseq );

    return res;
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
std::tuple< std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hpro::TMatrix > >
build_uniform_lvl ( const hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const hpro::TTruncAcc &      acc,
                    const size_t                 /* nseq */ = hpro::CFG::Arith::max_seq_size ) // ignored
{
    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;

    auto  rowcb = std::unique_ptr< cluster_basis >();
    auto  colcb = std::unique_ptr< cluster_basis >();
    auto  M     = std::unique_ptr< hpro::TMatrix >();
    
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
std::tuple< std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hpro::TMatrix > >
build_uniform_rec ( const hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const hpro::TTruncAcc &      acc,
                    const size_t                 /* nseq */ = hpro::CFG::Arith::max_seq_size ) // ignored
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< typename coeff_t::value_t, typename basisapx_t::value_t >,
                   "coefficient function and basis approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );

    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;

    auto  rowcb  = std::make_unique< cluster_basis >( bct->is().row_is() );
    auto  colcb  = std::make_unique< cluster_basis >( bct->is().col_is() );

    rowcb->set_nsons( bct->rowcl()->nsons() );
    colcb->set_nsons( bct->colcl()->nsons() );

    detail::init_cluster_bases( bct, *rowcb, *colcb );
    
    auto  basis_data = detail::rec_basis_data_t();
    auto  M          = std::unique_ptr< hpro::TMatrix >();
    
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
// assign block cluster to matrix
//
inline
void
assign_cluster ( hpro::TMatrix &              M,
                 const hpro::TBlockCluster &  bc )
{
    hlr::seq::matrix::assign_cluster( M, bc );
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

        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2)
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( BM->block( i, j ) != nullptr )
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

std::unique_ptr< HLIB::TMatrix >
copy ( const HLIB::TMatrix &  M )
{
    std::unique_ptr< HLIB::TMatrix >  res;

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

void
copy_to_task ( const HLIB::TMatrix &  A,
               HLIB::TMatrix &        B )
{
    HLR_ASSERT( A.type()     == B.type() );
    HLR_ASSERT( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, HLIB::TBlockMatrix );
        auto  BB = ptrcast(  &B, HLIB::TBlockMatrix );

        HLR_ASSERT( BA->nblock_rows() == BB->nblock_rows() );
        HLR_ASSERT( BA->nblock_cols() == BB->nblock_cols() );

        #pragma omp taskloop collapse(2)
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( BA->block( i, j ) != nullptr )
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

void
copy_to ( const HLIB::TMatrix &  A,
          HLIB::TMatrix &        B )
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

std::unique_ptr< HLIB::TMatrix >
realloc ( HLIB::TMatrix *  A )
{
    std::unique_ptr< HLIB::TMatrix >  res;

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
namespace detail
{

template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_uniform_task ( const hpro::TMatrix &                    M,
                    hlr::matrix::cluster_basis< value_t > &  rowcb,
                    hlr::matrix::cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );

        HLR_ASSERT( B->nblock_rows() == rowcb.nsons() );
        HLR_ASSERT( B->nblock_cols() == colcb.nsons() );

        #pragma omp taskloop collapse(2) default(shared) firstprivate(B,BM)
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy_uniform_task( * BM->block( i, j ), * rowcb.son(i), * colcb.son(j) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// omp taskloop for

        #pragma omp taskwait
        
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

        auto  R  = cptrcast( &M, hpro::TRkMatrix );

        auto  UA = rowcb.transform_forward( hpro::blas_mat_A< value_t >( R ) );
        auto  VB = colcb.transform_forward( hpro::blas_mat_B< value_t >( R ) );
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
std::unique_ptr< hpro::TMatrix >
copy_uniform ( const hpro::TMatrix &                    M,
               hlr::matrix::cluster_basis< value_t > &  rowcb,
               hlr::matrix::cluster_basis< value_t > &  colcb )
{
    auto  T = std::unique_ptr< hpro::TMatrix >();
    
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    {
        T = detail::copy_uniform_task( M, rowcb, colcb );
    }// omp task

    return T;
}

//
// convert given matrix into lowrank format
//
namespace detail
{

template < typename approx_t >
std::unique_ptr< hpro::TRkMatrix >
convert_to_lowrank ( const hpro::TMatrix &    M,
                     const hpro::TTruncAcc &  acc,
                     const approx_t &         approx )
{
    using  value_t = typename approx_t::value_t;
    
    if ( is_blocked( M ) )
    {
        //
        // convert each sub block into low-rank format and 
        // enlarge to size of M (pad with zeroes)
        //

        auto        B  = cptrcast( &M, hpro::TBlockMatrix );
        auto        Us = std::list< blas::matrix< value_t > >();
        auto        Vs = std::list< blas::matrix< value_t > >();
        std::mutex  mtx;

        #pragma omp taskloop collapse(2) default(shared) firstprivate(B)
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( is_null( B_ij ) )
                    continue;

                auto  R_ij = convert_to_lowrank( *B_ij, acc, approx );
                auto  U    = blas::matrix< value_t >( M.nrows(), R_ij->rank() );
                auto  V    = blas::matrix< value_t >( M.ncols(), R_ij->rank() );
                auto  U_i  = blas::matrix< value_t >( U, R_ij->row_is() - M.row_ofs(), blas::range::all );
                auto  V_j  = blas::matrix< value_t >( V, R_ij->col_is() - M.col_ofs(), blas::range::all );

                blas::copy( hpro::blas_mat_A< value_t >( R_ij ), U_i );
                blas::copy( hpro::blas_mat_B< value_t >( R_ij ), V_j );

                std::scoped_lock  lock( mtx );
                            
                Us.push_back( std::move( U ) );
                Vs.push_back( std::move( V ) );
            }// for
        }// omp taskloop for

        #pragma omp taskwait
        
        auto  [ U, V ] = approx( Us, Vs, acc );

        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( is_dense( M ) )
    {
        auto  D        = cptrcast( &M, hpro::TDenseMatrix );
        auto  T        = std::move( blas::copy( hpro::blas_mat< value_t >( D ) ) );
        auto  [ U, V ] = approx( T, acc );

        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R        = cptrcast( &M, hpro::TRkMatrix );
        auto  [ U, V ] = approx( hpro::blas_mat_A< value_t >( R ),
                                 hpro::blas_mat_B< value_t >( R ),
                                 acc );
        
        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

}// namespace detail

template < typename approx_t >
std::unique_ptr< hpro::TRkMatrix >
convert_to_lowrank ( const hpro::TMatrix &    M,
                     const hpro::TTruncAcc &  acc,
                     const approx_t &         approx )
{
    auto  T = std::unique_ptr< hpro::TRkMatrix >();
    
    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    {
        T = detail::convert_to_lowrank( M, acc, approx );
    }// omp task

    return T;
}

}}}// namespace hlr::omp::matrix

#endif // __HLR_OMP_MATRIX_HH

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

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/base/TTruncAcc.hh>

#include "hlr/arith/blas.hh"
#include "hlr/arith/norm.hh"
#include "hlr/approx/svd.hh" // DEBUG
#include "hlr/matrix/cluster_basis.hh"
#include "hlr/matrix/uniform_lrmatrix.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/matrix/convert.hh"
#include "hlr/matrix/restrict.hh"
#include "hlr/utils/checks.hh"

#include "hlr/seq/detail/matrix.hh"

namespace hlr { namespace seq { namespace matrix {

namespace hpro = HLIB;

using namespace hlr::matrix;

//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< hpro::TMatrix >
build ( const hpro::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const hpro::TTruncAcc &      acc,
        const size_t                 nseq = hpro::CFG::Arith::max_seq_size ) // ignored
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< hpro::TMatrix >  M;
    
    if ( bct->is_leaf() )
    {
        auto  rowis = bct->is().row_is();
        auto  colis = bct->is().col_is();
        
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< hpro::TMatrix >( lrapx.build( bct, acc( rowis, colis ) ) );
        }// if
        else
        {
            M = coeff.build( rowis, colis );
        }// else
    }// if
    else
    {
        M = std::make_unique< hpro::TBlockMatrix >( bct );
        
        auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

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
                    auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc, nseq );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        // make value type consistent in block matrix and sub blocks
        B->adjust_value_type();
    }// else

    // M->set_cluster_force( bct );
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

//
// build representation of nearfield of dense matrix with
// matrix structure defined by <bct>,  matrix coefficients
// defined by <coeff>
//
template < typename coeff_t >
std::unique_ptr< hpro::TMatrix >
build_nearfield ( const hpro::TBlockCluster *  bct,
                  const coeff_t &              coeff,
                  const size_t                 nseq = hpro::CFG::Arith::max_seq_size ) // ignored
{
    HLR_ASSERT( bct != nullptr );
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< hpro::TMatrix >  M;
    
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
            return nullptr;
        else
        {
            M = coeff.build( bct->is().row_is(), bct->is().col_is() );
        }// else
    }// if
    else
    {
        M = std::make_unique< hpro::TBlockMatrix >( bct );
        
        auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

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
                    auto  B_ij = build_nearfield( bct->son( i, j ), coeff, nseq );

                    if ( ! is_null( B_ij.get() ) )
                         B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        // make value type consistent in block matrix and sub blocks
        B->adjust_value_type();
    }// else

    // M->set_cluster_force( bct );
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

//
// build representation of sparse matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <M>. If low rank blocks are not zero, they are
// truncated to accuracy <acc> using approximation method <apx>
//
template < typename approx_t >
std::unique_ptr< hpro::TMatrix >
build ( const hpro::TBlockCluster *  bct,
        const hpro::TSparseMatrix &  S,
        const hpro::TTruncAcc &      acc,
        const approx_t &             apx,
        const size_t                 nseq = hpro::CFG::Arith::max_seq_size ) // ignored
{
    using  value_t = typename approx_t::value_t;
    
    // static_assert( std::is_same< typename coeff_t::value_t,
    //                              typename lrapx_t::value_t >::value,
    //                "coefficient function and low-rank approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< hpro::TMatrix >  M;
    
    if ( bct->is_leaf() )
    {
        //
        // restrict to local cluster and convert to desired format
        //

        auto  S_bct = hlr::matrix::restrict( S, bct->is() );
        
        if ( bct->is_adm() )
        {
            M = matrix::convert_to_lowrank( *S_bct, acc, apx );
        }// if
        else
        {
            M = matrix::convert_to_dense< value_t >( *S_bct );
        }// else
    }// if
    else
    {
        M = std::make_unique< hpro::TBlockMatrix >( bct );
        
        auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

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
                    auto  B_ij = build( bct->son( i, j ), S, acc, apx, nseq );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        // make value type consistent in block matrix and sub blocks
        B->adjust_value_type();
    }// else

    // M->set_cluster_force( bct );
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
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
    return detail::build_uniform_lvl( bct, coeff, lrapx, basisapx, acc );
}

template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hpro::TMatrix > >
build_uniform_lvl ( const hpro::TMatrix &    A,
                    const basisapx_t &       basisapx,
                    const hpro::TTruncAcc &  acc,
                    const size_t             /* nseq */ = hpro::CFG::Arith::max_seq_size ) // ignored
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;

    auto  rowcb = std::make_unique< cluster_basis >( A.row_is() );
    auto  colcb = std::make_unique< cluster_basis >( A.col_is() );

    if ( is_blocked( A ) )
    {
        rowcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix )->nblock_rows() );
        colcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix )->nblock_cols() );
    }// if

    detail::init_cluster_bases( A, *rowcb, *colcb );

    auto  M = detail::build_uniform_lvl( A, basisapx, acc, *rowcb, *colcb );

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
    
    auto  rowmap = detail::is_matrix_map_t();
    auto  colmap = detail::is_matrix_map_t();
    
    auto  M      = detail::build_uniform_rec( bct, coeff, lrapx, basisapx, acc, *rowcb, *colcb, rowmap, colmap );

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
            std::unique_ptr< hpro::TMatrix > >
build_uniform_rec ( const hpro::TMatrix &        A,
                    const basisapx_t &           basisapx,
                    const hpro::TTruncAcc &      acc,
                    const size_t                 /* nseq */ = hpro::CFG::Arith::max_seq_size ) // ignored
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;

    auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
    auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

    if ( is_blocked( A ) )
    {
        rowcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix )->nblock_rows() );
        colcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix )->nblock_cols() );
    }// if
    
    auto  rowmap = detail::is_matrix_map_t();
    auto  colmap = detail::is_matrix_map_t();
    auto  M      = detail::build_uniform_rec( A, basisapx, acc, *rowcb, *colcb, rowmap, colmap );

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
    M.set_cluster_force( & bc );
    
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, hpro::TBlockMatrix );

        HLR_ASSERT( ( B->nblock_rows() == bc.nrows() ) &&
                    ( B->nblock_cols() == bc.ncols() ) );
                    
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( B->block( i, j ) == nullptr )
                    continue;

                if ( bc.son( i, j ) == nullptr )
                    HLR_ERROR( "null cluster for non-null sub-block" );
                
                assign_cluster( * B->block( i, j ), * bc.son( i, j ) );
            }// for
        }// for
    }// if
}

//
// return copy of matrix
//
inline
std::unique_ptr< hpro::TMatrix >
copy ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

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

//
// return copy of matrix
//
inline
std::unique_ptr< hpro::TMatrix >
copy_nearfield ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
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
        
        return N;
    }// if
    else if ( is_dense( M ) )
    {
        // assuming non-structured block
        return M.copy();
    }// else
    else
        return nullptr;
}

//
// return copy of matrix with TRkMatrix replaced by tiled_lrmatrix
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_tiled ( const hpro::TMatrix &  M,
             const size_t           ntile )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = copy_tiled< value_t >( * BM->block( i, j ), ntile );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_lowrank( M ) )
    {
        //
        // copy low-rank data into tiled form
        //

        auto  RM = cptrcast( & M, hpro::TRkMatrix );
        auto  R  = std::make_unique< tiled_lrmatrix< value_t > >( RM->row_is(),
                                                                  RM->col_is(),
                                                                  ntile,
                                                                  blas::mat_U< value_t >( RM ),
                                                                  blas::mat_V< value_t >( RM ) );

        R->set_id( RM->id() );

        return R;
    }// if
    else
    {
        // assuming non-structured block
        return M.copy();
    }// else
}

//
// return copy of matrix with tiled_lrmatrix replaced by TRkMatrix
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_nontiled ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = copy_nontiled< value_t >( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( IS_TYPE( & M, tiled_lrmatrix ) )
    {
        //
        // copy low-rank data into tiled form
        //

        HLR_ASSERT( M.is_real() );
        
        auto  RM = cptrcast( & M, tiled_lrmatrix< real > );
        auto  R  = std::make_unique< hpro::TRkMatrix >( RM->row_is(), RM->col_is() );
        auto  U  = to_dense( RM->U() );
        auto  V  = to_dense( RM->V() );

        R->set_lrmat( U, V );
        R->set_id( RM->id() );

        return R;
    }// if
    else
    {
        // assuming non-structured block
        return M.copy();
    }// else
}

//
// return copy of (block-wise) diagonal part of matrix
//
inline
std::unique_ptr< hpro::TMatrix >
copy_diag ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < std::min( B->nblock_rows(), B->nblock_cols() ); ++i )
        {
            if ( ! is_null( BM->block( i, i ) ) )
            {
                auto  B_ii = copy_diag( * BM->block( i, i ) );
                    
                B_ii->set_parent( B );
                B->set_block( i, i, B_ii.release() );
            }// if
        }// for
        
        return N;
    }// if
    else
    {
        return M.copy();
    }// else
}

//
// return copy of (block-wise) lower-left part of matrix
//
inline
std::unique_ptr< hpro::TMatrix >
copy_ll ( const hpro::TMatrix &    M,
          const hpro::diag_type_t  diag = hpro::general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j <= i; ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = ( i == j ? copy_ll( * BM->block( i, j ), diag ) : copy( * BM->block( i, j ) ) );
                    
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
        auto  T = M.copy();

        if ( diag == hpro::unit_diag )
        {
            if ( T->row_is() == T->col_is() )
            {
                HLR_ASSERT( is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), hpro::TDenseMatrix );

                if ( D->is_complex() )
                    D->blas_cmat() = blas::identity< hpro::complex >( D->nrows() );
                else
                    D->blas_rmat() = blas::identity< hpro::real >( D->nrows() );
            }// if
        }// if

        return T;
    }// else
}

//
// return copy of (block-wise) upper-right part of matrix
//
inline
std::unique_ptr< hpro::TMatrix >
copy_ur ( const hpro::TMatrix &    M,
          const hpro::diag_type_t  diag = hpro::general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = i; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = ( i == j ? copy_ur( * BM->block( i, j ), diag ) : copy( * BM->block( i, j ) ) );
                    
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
        auto  T = M.copy();

        if ( diag == hpro::unit_diag )
        {
            if ( T->row_is() == T->col_is() )
            {
                HLR_ASSERT( is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), hpro::TDenseMatrix );

                if ( D->is_complex() )
                    D->blas_cmat() = blas::identity< hpro::complex >( D->nrows() );
                else
                    D->blas_rmat() = blas::identity< hpro::real >( D->nrows() );
            }// if
        }// if

        return T;
    }// else
}

//
// copy data of A to matrix B
// - ASSUMPTION: identical matrix structure
//
inline
void
copy_to ( const hpro::TMatrix &  A,
          hpro::TMatrix &        B )
{
    HLR_ASSERT( A.type()     == B.type() );
    HLR_ASSERT( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = ptrcast(  &B, hpro::TBlockMatrix );

        HLR_ASSERT( BA->nblock_rows() == BB->nblock_rows() );
        HLR_ASSERT( BA->nblock_cols() == BB->nblock_cols() );
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    HLR_ASSERT( ! is_null( BB->block( i, j ) ) );

                    copy_to( * BA->block( i, j ), * BB->block( i, j ) );
                }// if
                else
                    HLR_ASSERT( is_null( BB->block( i, j ) ) );
            }// for
        }// for
    }// if
    else
    {
        A.copy_to( & B );
    }// else
}

//
// copy lower-left data of A to matrix B
// - ASSUMPTION: identical matrix structure in lower-left part
//
inline
void
copy_to_ll ( const hpro::TMatrix &  A,
             hpro::TMatrix &        B )
{
    HLR_ASSERT( A.type()     == B.type() );
    HLR_ASSERT( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = ptrcast(  &B, hpro::TBlockMatrix );

        HLR_ASSERT( BA->nblock_rows() == BB->nblock_rows() );
        HLR_ASSERT( BA->nblock_cols() == BB->nblock_cols() );
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j <= i; ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    HLR_ASSERT( ! is_null( BB->block( i, j ) ) );

                    if ( i == j )
                        copy_to_ll( * BA->block( i, j ), * BB->block( i, j ) );
                    else
                        copy_to( * BA->block( i, j ), * BB->block( i, j ) );
                }// if
                else
                    HLR_ASSERT( is_null( BB->block( i, j ) ) );
            }// for
        }// for
    }// if
    else
    {
        A.copy_to( & B );
    }// else
}

//
// copy upper-right data of A to matrix B
// - ASSUMPTION: identical matrix structure in upper-right part
//
inline
void
copy_to_ur ( const hpro::TMatrix &  A,
             hpro::TMatrix &        B )
{
    HLR_ASSERT( A.type()     == B.type() );
    HLR_ASSERT( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = ptrcast(  &B, hpro::TBlockMatrix );

        HLR_ASSERT( BA->nblock_rows() == BB->nblock_rows() );
        HLR_ASSERT( BA->nblock_cols() == BB->nblock_cols() );
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = i; j < BA->nblock_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    HLR_ASSERT( ! is_null( BB->block( i, j ) ) );

                    if ( i == j )
                        copy_to_ur( * BA->block( i, j ), * BB->block( i, j ) );
                    else
                        copy_to( * BA->block( i, j ), * BB->block( i, j ) );
                }// if
                else
                    HLR_ASSERT( is_null( BB->block( i, j ) ) );
            }// for
        }// for
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
inline
std::unique_ptr< hpro::TMatrix >
realloc ( hpro::TMatrix *  A )
{
    if ( is_null( A ) )
        return nullptr;
    
    if ( is_blocked( A ) )
    {
        auto  B  = ptrcast( A, hpro::TBlockMatrix );
        auto  C  = std::make_unique< hpro::TBlockMatrix >();
        auto  BC = ptrcast( C.get(), hpro::TBlockMatrix );

        C->copy_struct_from( B );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  C_ij = realloc( B->block( i, j ) );

                BC->set_block( i, j, C_ij.release() );
                B->set_block( i, j, nullptr );
            }// for
        }// for

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

//
// nullify data in matrix, e.g., M := 0
//
inline
void
clear ( hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( & M, hpro::TBlockMatrix );

        for ( uint  i = 0; i < BM->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BM->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    clear( * BM->block( i, j ) );
                }// if
            }// for
        }// for
    }// if
    else if ( is_lowrank( & M ) )
    {
        auto  R = ptrcast( & M, hpro::TRkMatrix );

        R->set_rank( 0 );
    }// if
    else if ( is_dense( & M ) )
    {
        auto  D = ptrcast( & M, hpro::TDenseMatrix );

        if ( D->is_complex() )
            blas::fill( hpro::blas_mat< hpro::complex >( D ), hpro::complex(0) );
        else
            blas::fill( hpro::blas_mat< hpro::real >( D ), hpro::real(0) );
    }// if
    else
        HLR_ASSERT( false );
}

//
// return copy of matrix with uniform low-rank matrices
// - TODO: add cluster basis as template argument to allow
//         different bases
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_uniform ( const hpro::TMatrix &       M,
               cluster_basis< value_t > &  rowcb,
               cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );

        HLR_ASSERT( B->nblock_rows() == rowcb.nsons() );
        HLR_ASSERT( B->nblock_cols() == colcb.nsons() );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = copy_uniform( * BM->block( i, j ), * rowcb.son(i), * colcb.son(j) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_lowrank( M ) )
    {
        //
        // project into row/column cluster basis:
        //
        //   M = A·B^H = (U·U^H·A) (V·V^H·B)^H
        //             = U · (U^H·A)·(V^H·B)^H · V^H
        //             = U · S · V^H   with  S = (U^H·A)·(V^H·B)^H
        //
        
        auto  R  = cptrcast( &M, hpro::TRkMatrix );

        auto  UA = rowcb.transform_forward( blas::mat_U< value_t >( R ) );
        auto  VB = colcb.transform_forward( blas::mat_V< value_t >( R ) );
        auto  S  = blas::prod( UA, blas::adjoint( VB ) );
        auto  UR = std::make_unique< uniform_lrmatrix< value_t > >( M.row_is(), M.col_is(),
                                                                    rowcb, colcb,
                                                                    std::move( S ) );

        UR->set_id( R->id() );

        return UR;
    }// if
    else
    {
        // assuming dense block (no low-rank)
        return M.copy();
    }// else
}

//
// return copy of matrix with uniform low-rank matrices converted
// to standard lowrank matrices
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_nonuniform ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = copy_nonuniform< value_t >( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_uniform_lowrank( M ) )
    {
        auto  R  = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  U  = blas::prod( R->row_cb().basis(), R->coeff() );
        auto  V  = blas::copy( R->col_cb().basis() );
        auto  SR = std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );

        SR->set_id( R->id() );

        return SR;
    }// if
    else
    {
        // assuming dense block (no low-rank)
        return M.copy();
    }// else
}

//
// import functions from matrix module
//
using hlr::matrix::convert_to_lowrank;
using hlr::matrix::convert_to_dense;

}}}// namespace hlr::seq::matrix

#endif // __HLR_SEQ_MATRIX_HH

#ifndef __HLR_SEQ_MATRIX_HH
#define __HLR_SEQ_MATRIX_HH
//
// Project     : HLib
// Module      : seq/matrix
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
#include "hlr/matrix/lrmatrix.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/matrix/convert.hh"
#include "hlr/matrix/restrict.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tensor.hh"

#include "hlr/seq/detail/matrix.hh"

namespace hlr { namespace seq { namespace matrix {

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
        const size_t                 nseq = Hpro::CFG::Arith::max_seq_size ) // ignored
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );

    using value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< Hpro::TMatrix< value_t > >  M;
    
    if ( bct->is_leaf() )
    {
        auto  rowis = bct->is().row_is();
        auto  colis = bct->is().col_is();
        
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( bct, acc( rowis, colis ) ) );

            // if ( is_lowrank( *M ) )
            // {
            //     auto  R  = ptrcast( M.get(), Hpro::TRkMatrix< value_t > );
            //     auto  zR = std::make_unique< hlr::matrix::lrmatrix< value_t > >( rowis, colis,
            //                                                                      std::move( blas::mat_U( R ) ),
            //                                                                      std::move( blas::mat_V( R ) ) );
            //     M = std::move( zR );
            // }// if
        }// if
        else
        {
            M = coeff.build( rowis, colis );

            // if ( is_dense( *M ) )
            // {
            //     auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
            //     auto  zD = std::make_unique< hlr::matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) );

            //     M = std::move( zD );
            // }// if
        }// else
    }// if
    else
    {
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >( bct );
        
        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

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
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build_nearfield ( const Hpro::TBlockCluster *  bct,
                  const coeff_t &              coeff,
                  const size_t                 nseq = Hpro::CFG::Arith::max_seq_size ) // ignored
{
    using  value_t = typename coeff_t::value_t;
    
    HLR_ASSERT( bct != nullptr );
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< Hpro::TMatrix< value_t > >  M;
    
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
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >( bct );
        
        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

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
template < typename value_t,
           typename approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build ( const Hpro::TBlockCluster *  bct,
        const Hpro::TSparseMatrix< value_t > &  S,
        const Hpro::TTruncAcc &      acc,
        const approx_t &             apx,
        const size_t                 nseq = Hpro::CFG::Arith::max_seq_size ) // ignored
{
    // static_assert( std::is_same< typename coeff_t::value_t,
    //                              typename lrapx_t::value_t >::value,
    //                "coefficient function and low-rank approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< Hpro::TMatrix< value_t > >  M;
    
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
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >( bct );
        
        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

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
            std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > > >
build_uniform_lvl ( const Hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const Hpro::TTruncAcc &      acc,
                    const size_t                 /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
{
    return detail::build_uniform_lvl( bct, coeff, lrapx, basisapx, acc );
}

template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_uniform_lvl ( const Hpro::TMatrix< typename basisapx_t::value_t > &    A,
                    const basisapx_t &       basisapx,
                    const Hpro::TTruncAcc &  acc,
                    const size_t             /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;

    auto  rowcb = std::make_unique< cluster_basis >( A.row_is() );
    auto  colcb = std::make_unique< cluster_basis >( A.col_is() );

    if ( is_blocked( A ) )
    {
        rowcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
        colcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
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
            std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > > >
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
    
    HLR_ASSERT( bct != nullptr );

    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;

    auto  rowcb  = std::make_unique< cluster_basis >( bct->is().row_is() );
    auto  colcb  = std::make_unique< cluster_basis >( bct->is().col_is() );

    rowcb->set_nsons( bct->rowcl()->nsons() );
    colcb->set_nsons( bct->colcl()->nsons() );
    
    auto  rowmap = detail::is_matrix_map_t< value_t >();
    auto  colmap = detail::is_matrix_map_t< value_t >();
    
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
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_uniform_rec ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
                    const basisapx_t &           basisapx,
                    const Hpro::TTruncAcc &      acc,
                    const size_t                 /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
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
    
    auto  rowmap = detail::is_matrix_map_t< value_t >();
    auto  colmap = detail::is_matrix_map_t< value_t >();
    auto  M      = detail::build_uniform_rec( A, basisapx, acc, *rowcb, *colcb, rowmap, colmap );

    return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}
    
//
// assign block cluster to matrix
//
template < typename value_t >
void
assign_cluster ( Hpro::TMatrix< value_t > &   M,
                 const Hpro::TBlockCluster &  bc )
{
    M.set_cluster_force( & bc );
    
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, Hpro::TBlockMatrix< value_t > );

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
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = hlr::seq::matrix::copy( * BM->block( i, j ) );
                    
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
// return truncated copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy ( const Hpro::TMatrix< value_t > &    M,
       const Hpro::TTruncAcc &  acc )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy( * BM->block( i, j ), acc );
                    
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
        auto  Mc = M.copy();

        Mc->truncate( acc( M.row_is(), M.col_is() ) );

        return Mc;
    }// else
}

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
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy_compressible( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( hlr::matrix::is_compressible_lowrank( M ) )
    {
        return M.copy();
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, Hpro::TRkMatrix< value_t > );
        auto  U = blas::copy( blas::mat_U( R ) );
        auto  V = blas::copy( blas::mat_V( R ) );

        if ( false )
        {
            auto  RU = blas::matrix< value_t >( U.ncols(), U.ncols() );
            auto  RV = blas::matrix< value_t >( V.ncols(), V.ncols() );

            blas::qr( U, RU );
            blas::qr( V, RV );

            auto  S = blas::prod( RU, blas::adjoint( RV ) );

            return  std::make_unique< matrix::lrsmatrix< value_t > >( R->row_is(), R->col_is(), std::move( U ), std::move( S ), std::move( V ) );
        }// if
        else
        {
            return  std::make_unique< matrix::lrmatrix< value_t > >( R->row_is(), R->col_is(), std::move( U ), std::move( V ) );
        }// else
    }// if
    else if ( matrix::is_compressible_dense( M ) )
    {
        return M.copy();
    }// if
    else if ( is_dense( M ) )
    {
        auto  D  = cptrcast( &M, Hpro::TDenseMatrix< value_t > );
        auto  DD = blas::copy( blas::mat( D ) );

        return  std::make_unique< matrix::dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( DD ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );

    return 0;
}

//
// return copy of matrix
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
// return copy of matrix with TRkMatrix< value_t > replaced by tiled_lrmatrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_tiled ( const Hpro::TMatrix< value_t > &  M,
             const size_t           ntile )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

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

        auto  RM = cptrcast( & M, Hpro::TRkMatrix< value_t > );
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
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_nontiled ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

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
        
        auto  RM = cptrcast( & M, tiled_lrmatrix< value_t > );
        auto  R  = std::make_unique< Hpro::TRkMatrix< value_t > >( RM->row_is(), RM->col_is() );
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
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_diag ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

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
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_ll ( const Hpro::TMatrix< value_t > &    M,
          const Hpro::diag_type_t  diag = Hpro::general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j <= i; ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = ( i == j ? copy_ll( * BM->block( i, j ), diag ) : hlr::seq::matrix::copy( * BM->block( i, j ) ) );
                    
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

        if ( diag == Hpro::unit_diag )
        {
            if ( T->row_is() == T->col_is() )
            {
                HLR_ASSERT( is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), Hpro::TDenseMatrix< value_t > );

                D->blas_mat() = blas::identity< value_t >( D->nrows() );
            }// if
        }// if

        return T;
    }// else
}

//
// return copy of (block-wise) upper-right part of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_ur ( const Hpro::TMatrix< value_t > &    M,
          const Hpro::diag_type_t  diag = Hpro::general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = i; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = ( i == j ? copy_ur( * BM->block( i, j ), diag ) : hlr::seq::matrix::copy( * BM->block( i, j ) ) );
                    
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

        if ( diag == Hpro::unit_diag )
        {
            if ( T->row_is() == T->col_is() )
            {
                HLR_ASSERT( is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), Hpro::TDenseMatrix< value_t > );

                D->blas_mat() = blas::identity< value_t >( D->nrows() );
            }// if
        }// if

        return T;
    }// else
}

//
// copy data of A to matrix B
// - ASSUMPTION: identical matrix structure
//
template < typename value_t >
void
copy_to ( const Hpro::TMatrix< value_t > &  A,
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
template < typename value_t >
void
copy_to_ll ( const Hpro::TMatrix< value_t > &  A,
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
template < typename value_t >
void
copy_to_ur ( const Hpro::TMatrix< value_t > &  A,
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
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
realloc ( Hpro::TMatrix< value_t > *  A )
{
    if ( is_null( A ) )
        return nullptr;
    
    if ( is_blocked( A ) )
    {
        auto  B  = ptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto  C  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  BC = ptrcast( C.get(), Hpro::TBlockMatrix< value_t > );

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
template < typename value_t >
void
clear ( Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( & M, Hpro::TBlockMatrix< value_t > );

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
        auto  R = ptrcast( & M, Hpro::TRkMatrix< value_t > );

        R->set_rank( 0 );
    }// if
    else if ( is_dense( & M ) )
    {
        auto  D = ptrcast( & M, Hpro::TDenseMatrix< value_t > );

        blas::fill( Hpro::blas_mat< value_t >( D ), value_t(0) );
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
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_uniform ( const Hpro::TMatrix< value_t > &       M,
               cluster_basis< value_t > &  rowcb,
               cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

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
        
        auto  R  = cptrcast( &M, Hpro::TRkMatrix< value_t > );

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
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_nonuniform ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

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
        auto  SR = std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );

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

// //
// // return copy of matrix with uniform low-rank matrices converted
// // to standard lowrank matrices
// //
// inline
// std::unique_ptr< Hpro::TMatrix< value_t > >
// copy_nongeneric ( const Hpro::TMatrix< value_t > &  M )
// {
//     if ( is_blocked( M ) )
//     {
//         auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
//         auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
//         auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

//         B->copy_struct_from( BM );

//         for ( uint  i = 0; i < B->nblock_rows(); ++i )
//         {
//             for ( uint  j = 0; j < B->nblock_cols(); ++j )
//             {
//                 if ( ! is_null( BM->block( i, j ) ) )
//                 {
//                     auto  B_ij = copy_nongeneric( * BM->block( i, j ) );
                    
//                     B_ij->set_parent( B );
//                     B->set_block( i, j, B_ij.release() );
//                 }// if
//             }// for
//         }// for
        
//         return N;
//     }// if
//     else if ( is_generic_lowrank( M ) )
//     {
//         auto  R  = cptrcast( &M, lrmatrix );

//         if ( R->is_compressed() )
//         {
//             HLR_ERROR( "TODO" );
//         }// if

//         auto  SR = std::unique_ptr< Hpro::TRkMatrix< value_t > >();
        
//         if ( R->value_type() == blas::value_type::rfp32 )
//         {
//             auto  U = blas::copy< Hpro::real, float >( R->U< float >() );
//             auto  V = blas::copy< Hpro::real, float >( R->V< float >() );

//             SR = std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
//         }// if
//         else if ( R->value_type() == blas::value_type::rfp64 )
//         {
//             auto  U = blas::copy< Hpro::real, double >( R->U< double >() );
//             auto  V = blas::copy< Hpro::real, double >( R->V< double >() );

//             SR = std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
//         }// if
//         else if ( R->value_type() == blas::value_type::cfp32 )
//         {
//             auto  U = blas::copy< Hpro::complex, std::complex< float > >( R->U< std::complex< float > >() );
//             auto  V = blas::copy< Hpro::complex, std::complex< float > >( R->V< std::complex< float > >() );

//             SR = std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
//         }// if
//         else if ( R->value_type() == blas::value_type::cfp64 )
//         {
//             auto  U = blas::copy< Hpro::complex, std::complex< double > >( R->U< std::complex< double > >() );
//             auto  V = blas::copy< Hpro::complex, std::complex< double > >( R->V< std::complex< double > >() );

//             SR = std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
//         }// if
        
//         SR->set_id( R->id() );

//         return SR;
//     }// if
//     else if ( is_generic_dense( M ) )
//     {
//         auto  D  = cptrcast( &M, dense_matrix );

//         if ( D->is_compressed() )
//         {
//             HLR_ERROR( "TODO" );
//         }// if

//         auto  SD = std::unique_ptr< Hpro::TDenseMatrix< value_t > >();

//         if ( D->value_type() == blas::value_type::rfp32 )
//         {
//             SD = std::make_unique< Hpro::TDenseMatrix< value_t > >( M.row_is(), M.col_is(), std::move( blas::copy< Hpro::real, float >( D->M< float >() ) ) );
//         }// if
//         else if ( D->value_type() == blas::value_type::rfp64 )
//         {
//             SD = std::make_unique< Hpro::TDenseMatrix< value_t > >( M.row_is(), M.col_is(), std::move( blas::copy< Hpro::real, double >( D->M< double >() ) ) );
//         }// if
//         else if ( D->value_type() == blas::value_type::cfp32 )
//         {
//             SD = std::make_unique< Hpro::TDenseMatrix< value_t > >( M.row_is(), M.col_is(), std::move( blas::copy< Hpro::complex, std::complex< float > >( D->M< std::complex< float > >() ) ) );
//         }// if
//         else if ( D->value_type() == blas::value_type::cfp64 )
//         {
//             SD = std::make_unique< Hpro::TDenseMatrix< value_t > >( M.row_is(), M.col_is(), std::move( blas::copy< Hpro::complex, std::complex< double > >( D->M< std::complex< double > >() ) ) );
//         }// if

//         SD->set_id( D->id() );

//         return SD;
//     }// if
//     else
//     {
//         return M.copy();
//     }// else
// }

}}}// namespace hlr::seq::matrix

#endif // __HLR_SEQ_MATRIX_HH

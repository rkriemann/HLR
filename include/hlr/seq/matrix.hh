#ifndef __HLR_SEQ_MATRIX_HH
#define __HLR_SEQ_MATRIX_HH
//
// Project     : HLR
// Module      : seq/matrix
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cassert>
#include <type_traits>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>

#include <hlr/arith/blas.hh>
#include <hlr/approx/accuracy.hh>
#include <hlr/approx/traits.hh>
#include <hlr/bem/traits.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/uniform_lr2matrix.hh>
#include <hlr/matrix/h2_lrmatrix.hh>
#include <hlr/matrix/lrsvmatrix.hh>
#include <hlr/matrix/tiled_lrmatrix.hh>
#include <hlr/matrix/sparse_matrix.hh>
#include <hlr/matrix/convert.hh>
#include <hlr/matrix/restrict.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/tensor.hh>

#include <hlr/seq/detail/uniform_matrix.hh>
#include <hlr/seq/detail/h2_matrix.hh>

namespace hlr { namespace seq { namespace matrix {

using namespace hlr::matrix;

//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
template < coefficient_function_type  coeff_t,
           lowrank_approx_type        lrapx_t >
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build ( const Hpro::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const accuracy &             acc,
        const bool                   compress = false,
        const size_t                 nseq     = 0 ) // ignored
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );

    using value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    if ( bct->is_leaf() )
    {
        auto  rowis = bct->is().row_is();
        auto  colis = bct->is().col_is();
        auto  lacc  = acc( rowis, colis );
        
        if ( bct->is_adm() )
        {
            M = lrapx.build( bct, acc( rowis, colis ) );

            if ( Hpro::is_lowrank( *M ) )
            {
                auto  R = ptrcast( M.get(), Hpro::TRkMatrix< value_t > );
                auto  zR = std::make_unique< hlr::matrix::lrmatrix< value_t > >( rowis, colis,
                                                                                 std::move( blas::mat_U( R ) ),
                                                                                 std::move( blas::mat_V( R ) ) );

                if ( compress )
                    zR->compress( lacc );
                
                M = std::move( zR );
            }// if
            else if ( matrix::is_lowrank( *M ) )
            {
                auto  R = ptrcast( M.get(), matrix::lrmatrix< value_t > );
                
                if ( compress )
                    R->compress( lacc );
            }// if
            else if ( Hpro::is_dense( *M ) )
            {
                auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                auto  zD = std::make_unique< matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) );

                if ( compress )
                    zD->compress( lacc );
                
                M = std::move( zD );
            }// if
            else if ( matrix::is_dense( *M ) )
            {
                auto  D = ptrcast( M.get(), matrix::dense_matrix< value_t > );

                if ( compress )
                    D->compress( lacc );
            }// if
            else
                HLR_ERROR( "unsupported matrix type: " + M->typestr() );
        }// if
        else
        {
            M = coeff.build( rowis, colis );

            if ( Hpro::is_dense( *M ) )
            {
                auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                auto  zD = std::make_unique< matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) );

                if ( compress )
                    zD->compress( lacc );
                
                M = std::move( zD );
            }// if
            else if ( matrix::is_dense( *M ) )
            {
                auto  D = ptrcast( M.get(), matrix::dense_matrix< value_t > );

                if ( compress )
                    D->compress( lacc );
            }// if
            else
                HLR_ERROR( "unsupported matrix type: " + M->typestr() );
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
                    auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc, compress, nseq );

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
// same as above but use lrsvmatrix for lowrank blocks
//
template < coefficient_function_type  coeff_t,
           lowrank_approx_type        lrapx_t >
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build_sv ( const Hpro::TBlockCluster *  bct,
           const coeff_t &              coeff,
           const lrapx_t &              lrapx,
           const accuracy &             acc,
           const bool                   compress,
           const size_t                 nseq = 0 ) // ignored
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );

    using value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    if ( bct->is_leaf() )
    {
        auto  rowis = bct->is().row_is();
        auto  colis = bct->is().col_is();
        auto  lacc  = acc( rowis, colis );
        
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( bct, lacc ) );

            if ( Hpro::is_lowrank( *M ) )
            {
                auto  R  = ptrcast( M.get(), Hpro::TRkMatrix< value_t > );
                auto  zR = std::make_unique< hlr::matrix::lrsvmatrix< value_t > >( rowis, colis,
                                                                                   std::move( blas::mat_U( R ) ),
                                                                                   std::move( blas::mat_V( R ) ) );

                if ( compress )
                    zR->compress( lacc );
                
                M = std::move( zR );
            }// if
            else if ( matrix::is_lowrank( *M ) )
            {
                auto  R  = ptrcast( M.get(), matrix::lrmatrix< value_t > );
                auto  zR = std::make_unique< hlr::matrix::lrsvmatrix< value_t > >( rowis, colis, R->U(), R->V() );
                
                if ( compress )
                    zR->compress( lacc );
                
                M = std::move( zR );
            }// if
            else if ( matrix::is_lowrank_sv( *M ) )
            {
                auto  R = ptrcast( M.get(), matrix::lrsvmatrix< value_t > );

                if ( compress )
                    R->compress( lacc );
            }// if
            else if ( Hpro::is_dense( *M ) )
            {
                auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                auto  zD = std::make_unique< matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) );

                if ( compress )
                    zD->compress( lacc );
                
                M = std::move( zD );
            }// if
            else if ( matrix::is_dense( *M ) )
            {
                auto  D = ptrcast( M.get(), matrix::dense_matrix< value_t > );

                if ( compress )
                    D->compress( lacc );
            }// if
            else
                HLR_ERROR( "unsupported matrix type: " + M->typestr() );
        }// if
        else
        {
            M = coeff.build( rowis, colis );

            if ( Hpro::is_dense( *M ) )
            {
                auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                auto  zD = std::make_unique< matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) );

                if ( compress )
                    zD->compress( lacc );
                
                M = std::move( zD );
            }// if
            else if ( matrix::is_dense( *M ) )
            {
                auto  D = ptrcast( M.get(), matrix::dense_matrix< value_t > );

                if ( compress )
                    D->compress( lacc );
            }// if
            else
                HLR_ERROR( "unsupported matrix type: " + M->typestr() );
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
                    auto  B_ij = build_sv( bct->son( i, j ), coeff, lrapx, acc, compress, nseq );

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
// build matrix blocks but just report memory sizes
//
template < coefficient_function_type  coeff_t,
           lowrank_approx_type        lrapx_t >
size_t
mem_sv ( const Hpro::TBlockCluster *  bct,
         const coeff_t &              coeff,
         const lrapx_t &              lrapx,
         const accuracy &             acc,
         const size_t                 nseq = 0 ) // ignored
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );

    using value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    size_t  mem = 0;
    
    if ( bct->is_leaf() )
    {
        auto  M     = std::unique_ptr< Hpro::TMatrix< value_t > >();
        auto  rowis = bct->is().row_is();
        auto  colis = bct->is().col_is();
        auto  lacc  = acc( rowis, colis );
        
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( bct, lacc ) );

            if ( Hpro::is_lowrank( *M ) )
            {
                auto  R  = ptrcast( M.get(), Hpro::TRkMatrix< value_t > );
                auto  zR = std::make_unique< hlr::matrix::lrsvmatrix< value_t > >( rowis, colis,
                                                                                   std::move( blas::mat_U( R ) ),
                                                                                   std::move( blas::mat_V( R ) ) );

                zR->compress( lacc );
                M = std::move( zR );
            }// if
            else if ( matrix::is_lowrank( *M ) )
            {
                auto  R  = ptrcast( M.get(), matrix::lrmatrix< value_t > );
                auto  zR = std::make_unique< hlr::matrix::lrsvmatrix< value_t > >( rowis, colis, R->U(), R->V() );
                
                zR->compress( lacc );
                M = std::move( zR );
            }// if
            else if ( matrix::is_lowrank_sv( *M ) )
            {
                auto  R = ptrcast( M.get(), matrix::lrsvmatrix< value_t > );

                R->compress( lacc );
            }// if
            else if ( Hpro::is_dense( *M ) )
            {
                auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                auto  zD = std::make_unique< matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) );

                zD->compress( lacc );
                M = std::move( zD );
            }// if
            else if ( matrix::is_dense( *M ) )
            {
                auto  D = ptrcast( M.get(), matrix::dense_matrix< value_t > );

                D->compress( lacc );
            }// if
            else
                HLR_ERROR( "unsupported matrix type: " + M->typestr() );
        }// if
        else
        {
            M = coeff.build( rowis, colis );

            if ( Hpro::is_dense( *M ) )
            {
                auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                auto  zD = std::make_unique< matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) );

                zD->compress( lacc );
                M = std::move( zD );
            }// if
            else if ( matrix::is_dense( *M ) )
            {
                auto  D = ptrcast( M.get(), matrix::dense_matrix< value_t > );

                D->compress( lacc );
            }// if
            else
                HLR_ERROR( "unsupported matrix type: " + M->typestr() );
        }// else

        mem += M->data_byte_size();
    }// if
    else
    {
        //
        // recurse
        //
        
        for ( uint  i = 0; i < bct->nrows(); ++i )
        {
            for ( uint  j = 0; j < bct->ncols(); ++j )
            {
                if ( bct->son( i, j ) != nullptr )
                {
                    mem += mem_sv( bct->son( i, j ), coeff, lrapx, acc, nseq );
                }// if
            }// for
        }// for
    }// else
    
    return mem;
}

//
// build representation of nearfield of dense matrix with
// matrix structure defined by <bct>,  matrix coefficients
// defined by <coeff>
//
template < coefficient_function_type coeff_t >
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build_nearfield ( const Hpro::TBlockCluster *  bct,
                  const coeff_t &              coeff,
                  const size_t                 nseq = 0 ) // ignored
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
            const auto  row_is = bct->is().row_is();
            const auto  col_is = bct->is().col_is();
            
            M = coeff.build( row_is, col_is );

            if ( Hpro::is_dense( *M ) )
            {
                auto  D = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );

                M = std::move( std::make_unique< hlr::matrix::dense_matrix< value_t > >( row_is, col_is, std::move( blas::mat( D ) ) ) );
            }// if
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
           approx::approximation_type  approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build ( const Hpro::TBlockCluster &             bct,
        const Hpro::TSparseMatrix< value_t > &  S,
        const accuracy &                        acc,
        const approx_t &                        apx,
        const size_t                            nseq = 0 ) // ignored
{
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< Hpro::TMatrix< value_t > >  M;
    
    if ( bct.is_leaf() )
    {
        //
        // restrict to local cluster and convert to desired format
        //

        auto  S_bct = hlr::matrix::restrict( S, bct.is() );
        
        if ( bct.is_adm() )
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
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >( &bct );
        
        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct.nrows() ) ||
            ( B->nblock_cols() != bct.ncols() ))
            B->set_block_struct( bct.nrows(), bct.ncols() );

        // recurse
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( bct.son( i, j ) ) )
                {
                    auto  B_ij = build( *bct.son( i, j ), S, acc, apx, nseq );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
    }// else

    // M->set_cluster_force( bct );
    M->set_id( bct.id() );
    M->set_procs( bct.procs() );

    return M;
}

template < typename value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_nd ( const Hpro::TBlockCluster &             bct,
           const Hpro::TSparseMatrix< value_t > &  S,
           const accuracy &                        acc,
           const approx_t &                        apx,
           const size_t                            nseq = 0 ) // ignored
{
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< Hpro::TMatrix< value_t > >  M;
    
    if ( bct.is_leaf() )
    {
        return build( bct, S, acc, apx, nseq );
    }// if
    else
    {
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >( &bct );
        
        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct.nrows() ) ||
            ( B->nblock_cols() != bct.ncols() ))
            B->set_block_struct( bct.nrows(), bct.ncols() );

        //
        // recurse for diagonal
        //
        
        const auto  nbr = B->nblock_rows();
        const auto  nbc = B->nblock_cols();
        
        for ( uint  i = 0; i < std::min( nbr, nbc )-1; ++i )
        {
            if ( ! is_null( bct.son( i, i ) ) )
            {
                auto  B_ii = build_nd( *bct.son( i, i ), S, acc, apx, nseq );
                
                B->set_block( i, i, B_ii.release() );
            }// if
        }// for

        //
        // standard construction for interface-domain couplings
        //

        for ( uint  i = 0; i < nbr-1; ++i )
        {
            if ( ! is_null( bct.son( i, nbc-1 ) ) )
            {
                auto  B_ij = build( *bct.son( i, nbc-1 ), S, acc, apx, nseq );
                
                B->set_block( i, nbc-1, B_ij.release() );
            }// if
        }// for
        
        for ( uint  j = 0; j < nbc-1; ++j )
        {
            if ( ! is_null( bct.son( nbr-1, j ) ) )
            {
                auto  B_ij = build( *bct.son( nbr-1, j ), S, acc, apx, nseq );
                
                B->set_block( nbr-1, j, B_ij.release() );
            }// if
        }// for

        //
        // finally, the interface-interface block
        //

        if ( ! is_null( bct.son( nbr-1, nbc-1 ) ) )
        {
            auto  B_ij = build( *bct.son( nbr-1, nbc-1 ), S, acc, apx, nseq );
                
            B->set_block( nbr-1, nbc-1, B_ij.release() );
        }// if
    }// else

    // M->set_cluster_force( bct );
    M->set_id( bct.id() );
    M->set_procs( bct.procs() );

    return M;
}

//
// same as above but use sparse matrices instead of dense matrices
// for non-admissible blocks
//
template < typename value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_sparse ( const Hpro::TBlockCluster &             bct,
               const Hpro::TSparseMatrix< value_t > &  S,
               const accuracy &                        acc,
               const approx_t &                        apx,
               const size_t                            nseq = 0 ) // ignored
{
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< Hpro::TMatrix< value_t > >  M;
    
    if ( bct.is_leaf() )
    {
        //
        // restrict to local cluster and convert to desired format
        //

        auto  S_bct = hlr::matrix::restrict( S, bct.is() );
        
        if ( bct.is_adm() )
        {
            // M = matrix::convert_to_dense< value_t >( *S_bct );
            M = matrix::convert_to_lowrank( *S_bct, acc, apx );
        }// if
        else if ( is_sparse( *S_bct ) )
        {
            // M = std::make_unique< matrix::sparse_matrix< value_t > >( * ptrcast( S_bct.get(), Hpro::TSparseMatrix< value_t > ) );
            M = std::move( S_bct );
        }// else
        else
        {
            M = matrix::convert_to_dense< value_t >( *S_bct );
        }// else
    }// if
    else
    {
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >( &bct );
        
        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct.nrows() ) ||
            ( B->nblock_cols() != bct.ncols() ))
            B->set_block_struct( bct.nrows(), bct.ncols() );

        // recurse
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( bct.son( i, j ) ) )
                {
                    auto  B_ij = build_sparse( *bct.son( i, j ), S, acc, apx, nseq );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
    }// else

    // M->set_cluster_force( bct );
    M->set_id( bct.id() );
    M->set_procs( bct.procs() );

    return M;
}

//
// build representation of dense matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and low-rank blocks computed by <lrapx>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//
template < coefficient_function_type coeff_t,
           lowrank_approx_type lrapx_t,
           approx::approximation_type basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
            std::unique_ptr< Hpro::TMatrix< Hpro::value_type_t< coeff_t > > > >
build_uniform ( const Hpro::TBlockCluster *  bc,
                const coeff_t &              coeff,
                const lrapx_t &              lrapx,
                const basisapx_t &           basisapx,
                const accuracy &             acc,
                const bool                   compress,
                const size_t                 /* nseq */ = 0 ) // ignored
{
    using value_t = typename coeff_t::value_t;

    auto  row_cls    = std::list< const Hpro::TCluster * >();
    auto  col_cls    = std::list< const Hpro::TCluster * >();
    auto  row_map    = std::vector< std::list< const Hpro::TBlockCluster * > >( bc->rowcl()->id() + 1 );
    auto  col_map    = std::vector< std::list< const Hpro::TBlockCluster * > >( bc->colcl()->id() + 1 );
    auto  row_cbs    = std::vector< shared_cluster_basis< value_t > * >( bc->rowcl()->id() + 1 );
    auto  col_cbs    = std::vector< shared_cluster_basis< value_t > * >( bc->colcl()->id() + 1 );
    auto  mat_map_H  = std::vector< Hpro::TMatrix< value_t > * >( bc->id() + 1, nullptr );
    auto  mat_map_U  = std::vector< Hpro::TMatrix< value_t > * >( bc->id() + 1, nullptr );
    auto  row_coup   = std::vector< blas::matrix< value_t > >( bc->id() + 1 );
    auto  col_coup   = std::vector< blas::matrix< value_t > >( bc->id() + 1 );
    auto  rowcb      = std::make_unique< shared_cluster_basis< value_t > >( * bc->rowcl() );
    auto  colcb      = std::make_unique< shared_cluster_basis< value_t > >( * bc->colcl() );

    detail::collect_clusters( bc->rowcl(), row_cls );
    detail::collect_clusters( bc->colcl(), col_cls );
    detail::build_block_map< value_t >( bc, row_map, col_map );

    //
    // intermix row/column clusters to free lowrank blocks as soon as possible
    //

    auto  iter_rows = row_cls.begin();
    auto  iter_cols = col_cls.begin();

    while (( iter_rows != row_cls.end() ) &&
           ( iter_cols != col_cls.end() ))
    {
        if ( iter_rows != row_cls.end() )
        {
            auto  rowcl = *iter_rows;
            auto  rowcb = std::make_unique< hlr::matrix::shared_cluster_basis< value_t > >( *rowcl );
            
            rowcb->set_nsons( rowcl->nsons() );
            rowcb->set_id( rowcl->id() );
            row_cbs[ rowcb->id() ] = rowcb.get();
            
            detail::build_uniform( rowcl, rowcb.release(),
                                   coeff, lrapx, basisapx, acc, compress,
                                   row_map, mat_map_H, mat_map_U, row_coup, col_coup,
                                   apply_normal );
            
            ++iter_rows;
        }// if

        if ( iter_cols != col_cls.end() )
        {
            auto  colcl = *iter_cols;
            auto  colcb = std::make_unique< hlr::matrix::shared_cluster_basis< value_t > >( *colcl );
            
            colcb->set_nsons( colcl->nsons() );
            colcb->set_id( colcl->id() );
            col_cbs[ colcb->id() ] = colcb.get();
            
            detail::build_uniform( colcl, colcb.release(),
                                   coeff, lrapx, basisapx, acc, compress,
                                   col_map, mat_map_H, mat_map_U, col_coup, row_coup,
                                   apply_adjoint );
            
            ++iter_cols;
        }// if
    }// while

    // check if all low rank blocks are gone
    for ( auto  M : mat_map_H )
    {
        HLR_ASSERT( ! hlr::matrix::is_lowrank( M ) );
    }// for
    
    auto  rowcb_root = std::unique_ptr< hlr::matrix::shared_cluster_basis< value_t > >( row_cbs[ bc->rowcl()->id() ] );
    auto  colcb_root = std::unique_ptr< hlr::matrix::shared_cluster_basis< value_t > >( col_cbs[ bc->colcl()->id() ] );
    auto  M_root     = std::unique_ptr< Hpro::TMatrix< value_t > >( mat_map_U[ bc->id() ] );

    HLR_ASSERT( ! is_null( M_root ) && ! is_null( rowcb_root ) && ! is_null( colcb_root ) );

    detail::fix_hierarchy( bc->rowcl(), rowcb_root.get(), row_cbs );
    detail::fix_hierarchy( bc->colcl(), colcb_root.get(), col_cbs );
    detail::fix_hierarchy( bc, M_root.get(), mat_map_U );
    
    return { std::move( rowcb_root ), std::move( colcb_root ), std::move( M_root ) };
}

template < coefficient_function_type coeff_t,
           lowrank_approx_type lrapx_t,
           approx::approximation_type basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
            std::unique_ptr< Hpro::TMatrix< Hpro::value_type_t< coeff_t > > > >
build_uniform_sep ( const Hpro::TBlockCluster *  bc,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const accuracy &             acc,
                    const bool                   compress,
                    const size_t                 /* nseq */ = 0 ) // ignored
{
    using value_t = typename coeff_t::value_t;

    auto  row_cls    = std::list< const Hpro::TCluster * >();
    auto  col_cls    = std::list< const Hpro::TCluster * >();
    auto  row_map    = std::vector< std::list< const Hpro::TBlockCluster * > >( bc->rowcl()->id() + 1 );
    auto  col_map    = std::vector< std::list< const Hpro::TBlockCluster * > >( bc->colcl()->id() + 1 );
    auto  row_cbs    = std::vector< shared_cluster_basis< value_t > * >( bc->rowcl()->id() + 1 );
    auto  col_cbs    = std::vector< shared_cluster_basis< value_t > * >( bc->colcl()->id() + 1 );
    auto  mat_map_H  = std::vector< Hpro::TMatrix< value_t > * >( bc->id() + 1, nullptr );
    auto  mat_map_U  = std::vector< Hpro::TMatrix< value_t > * >( bc->id() + 1, nullptr );
    auto  rowcb      = std::make_unique< shared_cluster_basis< value_t > >( * bc->rowcl() );
    auto  colcb      = std::make_unique< shared_cluster_basis< value_t > >( * bc->colcl() );

    detail::collect_clusters( bc->rowcl(), row_cls );
    detail::collect_clusters( bc->colcl(), col_cls );
    detail::build_block_map< value_t >( bc, row_map, col_map );

    //
    // intermix row/column clusters to free lowrank blocks as soon as possible
    //

    auto  iter_rows = row_cls.begin();
    auto  iter_cols = col_cls.begin();

    while (( iter_rows != row_cls.end() ) &&
           ( iter_cols != col_cls.end() ))
    {
        if ( iter_rows != row_cls.end() )
        {
            auto  rowcl = *iter_rows;
            auto  rowcb = std::make_unique< hlr::matrix::shared_cluster_basis< value_t > >( *rowcl );
            
            rowcb->set_nsons( rowcl->nsons() );
            rowcb->set_id( rowcl->id() );
            row_cbs[ rowcb->id() ] = rowcb.get();
            
            detail::build_uniform_sep( rowcl, rowcb.release(),
                                       coeff, lrapx, basisapx, acc, compress,
                                       row_map, mat_map_H, mat_map_U,
                                       apply_normal );
            
            ++iter_rows;
        }// if

        if ( iter_cols != col_cls.end() )
        {
            auto  colcl = *iter_cols;
            auto  colcb = std::make_unique< hlr::matrix::shared_cluster_basis< value_t > >( *colcl );
            
            colcb->set_nsons( colcl->nsons() );
            colcb->set_id( colcl->id() );
            col_cbs[ colcb->id() ] = colcb.get();
            
            detail::build_uniform_sep( colcl, colcb.release(),
                                       coeff, lrapx, basisapx, acc, compress,
                                       col_map, mat_map_H, mat_map_U,
                                       apply_adjoint );
            
            ++iter_cols;
        }// if
    }// while

    // check if all low rank blocks are gone
    for ( auto  M : mat_map_H )
    {
        HLR_ASSERT( ! hlr::matrix::is_lowrank( M ) );
    }// for
    
    auto  rowcb_root = std::unique_ptr< hlr::matrix::shared_cluster_basis< value_t > >( row_cbs[ bc->rowcl()->id() ] );
    auto  colcb_root = std::unique_ptr< hlr::matrix::shared_cluster_basis< value_t > >( col_cbs[ bc->colcl()->id() ] );
    auto  M_root     = std::unique_ptr< Hpro::TMatrix< value_t > >( mat_map_U[ bc->id() ] );

    HLR_ASSERT( ! is_null( M_root ) && ! is_null( rowcb_root ) && ! is_null( colcb_root ) );

    detail::fix_hierarchy( bc->rowcl(), rowcb_root.get(), row_cbs );
    detail::fix_hierarchy( bc->colcl(), colcb_root.get(), col_cbs );
    detail::fix_hierarchy( bc, M_root.get(), mat_map_U );
    
    return { std::move( rowcb_root ), std::move( colcb_root ), std::move( M_root ) };
}

template < coefficient_function_type coeff_t,
           lowrank_approx_type lrapx_t,
           approx::approximation_type basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
            std::unique_ptr< Hpro::TMatrix< Hpro::value_type_t< coeff_t > > > >
build_uniform_lvl ( const Hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const accuracy &             acc,
                    const bool                   compress,
                    const size_t                 /* nseq */ = 0 ) // ignored
{
    auto  [ rowcb, colcb, A ] = detail::build_uniform_lvl( bct, coeff, lrapx, basisapx, acc, compress );

    { int  id = 0;  detail::set_ids( *rowcb, id ); }
    { int  id = 0;  detail::set_ids( *colcb, id ); }

    return { std::move( rowcb ), std::move( colcb ), std::move( A ) };
}

template < coefficient_function_type coeff_t,
           lowrank_approx_type lrapx_t,
           approx::approximation_type basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
            std::unique_ptr< Hpro::TMatrix< Hpro::value_type_t< coeff_t > > > >
build_uniform_lvl_sep ( const Hpro::TBlockCluster *  bct,
                        const coeff_t &              coeff,
                        const lrapx_t &              lrapx,
                        const basisapx_t &           basisapx,
                        const accuracy &             acc,
                        const bool                   compress,
                        const size_t                 /* nseq */ = 0 ) // ignored
{
    auto  [ rowcb, colcb, A ] = detail::build_uniform_lvl_sep( bct, coeff, lrapx, basisapx, acc, compress );

    { int  id = 0;  detail::set_ids( *rowcb, id ); }
    { int  id = 0;  detail::set_ids( *colcb, id ); }

    return { std::move( rowcb ), std::move( colcb ), std::move( A ) };
}

// template < coefficient_function_type coeff_t,
//            lowrank_approx_type lrapx_t,
//            approx::approximation_type basisapx_t >
// std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
//             std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
//             std::unique_ptr< Hpro::TMatrix< Hpro::value_type_t< coeff_t > > > >
// build_uniform_cl ( const Hpro::TBlockCluster *  bt,
//                    const coeff_t &              coeff,
//                    const lrapx_t &              lrapx,
//                    const basisapx_t &           basisapx,
//                    const accuracy &             acc,
//                    const bool                   compress,
//                    const size_t                 /* nseq */ = 0 ) // ignored
// {
//     auto  rowct      = bt->rowcl();
//     auto  nrowcl     = rowct->id() + 1;
//     auto  row_blocks = std::vector< std::list< const Hpro::TBlockCluster * > >( nrowcl );

//     detail::build_block_lists( bt, row_blocks, false );

//     auto  colct      = bt->colcl();
//     auto  ncolcl     = colct->id() + 1;
//     auto  col_blocks = std::vector< std::list< const Hpro::TBlockCluster * > >( ncolcl );

//     detail::build_block_lists( bt, col_blocks, true );

//     auto  [ rowcb, colcb, A ] = detail::build_uniform_cl( bt, coeff, lrapx, basisapx, acc, compress,
//                                                           row_blocks, col_blocks );
// }

template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_uniform_lvl ( const Hpro::TMatrix< typename basisapx_t::value_t > &    A,
                    const basisapx_t &       basisapx,
                    const accuracy &         acc,
                    const size_t             /* nseq */ = 0 ) // ignored
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

    auto  rowcb = std::make_unique< cluster_basis >( A.row_is() );
    auto  colcb = std::make_unique< cluster_basis >( A.col_is() );

    if ( is_blocked( A ) )
    {
        rowcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
        colcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
    }// if

    int  row_id = 0;
    int  col_id = 0;
    
    detail::init_cluster_bases( A, *rowcb, *colcb, row_id, col_id );

    auto  M = detail::build_uniform_lvl( A, basisapx, acc, *rowcb, *colcb );

    return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}

//
// build representation of dense matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and low-rank blocks computed by <lrapx>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//
template < coefficient_function_type coeff_t,
           lowrank_approx_type lrapx_t,
           approx::approximation_type basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
            std::unique_ptr< Hpro::TMatrix< Hpro::value_type_t< coeff_t > > > >
build_uniform_rec ( const Hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const accuracy &             acc,
                    const size_t                 /* nseq */ = 0 ) // ignored
{
    static_assert( std::is_same_v< Hpro::value_type_t< coeff_t >, Hpro::value_type_t< lrapx_t > >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< Hpro::value_type_t< coeff_t >, Hpro::value_type_t< basisapx_t > >,
                   "coefficient function and basis approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );

    using value_t       = Hpro::value_type_t< coeff_t >;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

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
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_uniform_rec ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
                    const basisapx_t &           basisapx,
                    const accuracy &             acc,
                    const bool                   compress,
                    const size_t                 /* nseq */ = 0 ) // ignored
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

    //
    // mapping of index sets to lowrank matrices 
    //

    auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
    auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

    if ( is_blocked( A ) )
    {
        rowcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_rows() );
        colcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_cols() );
    }// if

    auto  row_map = detail::lr_coupling_map_t< value_t >();
    auto  col_map = detail::lr_coupling_map_t< value_t >();
    
    detail::build_mat_map( A, *rowcb, *colcb, row_map, col_map );
    
    //
    // build cluster bases
    //
    
    detail::build_cluster_basis( *rowcb, basisapx, acc, row_map, false );
    detail::build_cluster_basis( *colcb, basisapx, acc, col_map, true );

    //
    // construct uniform lowrank matrices with given cluster bases
    //
    
    auto  M = detail::build_uniform( A, *rowcb, *colcb );
    
    return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };

    // using value_t       = typename basisapx_t::value_t;
    // using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

    // auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
    // auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

    // if ( is_blocked( A ) )
    // {
    //     rowcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
    //     colcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
    // }// if
    
    // auto  rowmap = detail::is_matrix_map_t< value_t >();
    // auto  colmap = detail::is_matrix_map_t< value_t >();
    // auto  M      = detail::build_uniform_rec( A, basisapx, acc, *rowcb, *colcb, rowmap, colmap );

    // return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}

template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_uniform_rec_sep ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
                        const basisapx_t &           basisapx,
                        const accuracy &             acc,
                        const size_t                 /* nseq */ = 0 ) // ignored
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

    //
    // mapping of index sets to lowrank matrices 
    //

    auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
    auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

    if ( is_blocked( A ) )
    {
        rowcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_rows() );
        colcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_cols() );
    }// if

    auto  row_map = detail::lr_coupling_map_t< value_t >();
    auto  col_map = detail::lr_coupling_map_t< value_t >();
    
    detail::build_mat_map( A, *rowcb, *colcb, row_map, col_map );
    
    //
    // build cluster bases
    //
    
    detail::build_cluster_basis( *rowcb, basisapx, acc, row_map, false );
    detail::build_cluster_basis( *colcb, basisapx, acc, col_map, true );

    //
    // construct uniform lowrank matrices with given cluster bases
    //
    
    auto  M = detail::build_uniform_sep( A, *rowcb, *colcb );
    
    return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_uniform_sep ( const Hpro::TMatrix< value_t > &                A,
                    hlr::matrix::shared_cluster_basis< value_t > &  rowcb,
                    hlr::matrix::shared_cluster_basis< value_t > &  colcb )
{
    return detail::build_uniform_sep( A, rowcb, colcb );
}

namespace tlr
{

//
// special versions for BLR format
//
template < coefficient_function_type coeff_t,
           lowrank_approx_type       lrapx_t,
           approx::approximation_type        basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_uniform ( const Hpro::TBlockCluster *  bc,
                const coeff_t &              coeff,
                const lrapx_t &              lrapx,
                const basisapx_t &           basisapx,
                const accuracy &             acc )
{
    return detail::build_blr2( bc, coeff, lrapx, basisapx, acc );
}

template < coefficient_function_type coeff_t,
           lowrank_approx_type       lrapx_t,
           approx::approximation_type        basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_uniform_sep ( const Hpro::TBlockCluster *  bc,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const accuracy &             acc,
                    const bool                   compress )
{
    return detail::build_blr2_sep( bc, coeff, lrapx, basisapx, acc, compress );
}

template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_uniform ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
                const basisapx_t &                                     basisapx,
                const accuracy &                                       acc )
{
    return detail::build_blr2( A, basisapx, acc );
}

}// namespace blr

template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_h2_rec ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
               const basisapx_t &                                     basisapx,
               const accuracy &                                       acc,
               const bool                                             compress,
               const size_t                                           /* nseq */ = 0 ) // ignored
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::nested_cluster_basis< value_t >;

    //
    // mapping of index sets to lowrank matrices 
    //

    auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
    auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

    if ( is_blocked( A ) )
    {
        rowcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_rows() );
        colcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_cols() );
    }// if

    auto  row_map      = detail::lr_mat_map_t< value_t >();
    auto  row_coupling = detail::coupling_map_t< value_t >();
    auto  col_map      = detail::lr_mat_map_t< value_t >();
    auto  col_coupling = detail::coupling_map_t< value_t >();
    
    detail::build_mat_map( A, *rowcb, *colcb, row_map, row_coupling, col_map, col_coupling );
    
    //
    // build cluster bases
    //

    auto  empty_list = detail::lr_mat_list_t< value_t >();
    
    detail::build_nested_cluster_basis( *rowcb, basisapx, acc, row_map, row_coupling, empty_list, false, compress );
    detail::build_nested_cluster_basis( *colcb, basisapx, acc, col_map, col_coupling, empty_list, true,  compress );

    { int  id = 0;  detail::set_ids( *rowcb, id ); }
    { int  id = 0;  detail::set_ids( *colcb, id ); }

    //
    // construct uniform lowrank matrices with given cluster bases
    //
    
    auto  M = detail::build_h2( A, *rowcb, *colcb, acc, compress );
    
    return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}
    
template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_h2_rec_sep ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
                   const basisapx_t &                                     basisapx,
                   const accuracy &                                       acc,
                   const bool                                             compress,
                   const size_t                                           /* nseq */ = 0 ) // ignored
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::nested_cluster_basis< value_t >;

    //
    // mapping of index sets to lowrank matrices 
    //

    auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
    auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

    if ( is_blocked( A ) )
    {
        rowcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_rows() );
        colcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_cols() );
    }// if

    auto  row_map      = detail::lr_mat_map_t< value_t >();
    auto  row_coupling = detail::coupling_map_t< value_t >();
    auto  col_map      = detail::lr_mat_map_t< value_t >();
    auto  col_coupling = detail::coupling_map_t< value_t >();
    
    detail::build_mat_map( A, *rowcb, *colcb, row_map, row_coupling, col_map, col_coupling );
    
    //
    // build cluster bases
    //

    auto  empty_list = detail::lr_mat_list_t< value_t >();
    
    detail::build_nested_cluster_basis( *rowcb, basisapx, acc, row_map, row_coupling, empty_list, false, compress );
    detail::build_nested_cluster_basis( *colcb, basisapx, acc, col_map, col_coupling, empty_list, true,  compress );

    { int  id = 0;  detail::set_ids( *rowcb, id ); }
    { int  id = 0;  detail::set_ids( *colcb, id ); }

    //
    // construct uniform lowrank matrices with given cluster bases
    //
    
    auto  M = detail::build_h2_sep( A, *rowcb, *colcb, acc, compress );
    
    return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}
    
template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_h2 ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
           const hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > &  srowcb,
           const hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > &  scolcb,
           const basisapx_t &                                                         basisapx,
           const accuracy &                                                           acc,
           const bool                                                                 compress,
           const size_t                                                               /* nseq */ = 0 ) // ignored
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::nested_cluster_basis< value_t >;

    //
    // build cluster bases
    //

    auto  nrowcb  = std::make_unique< cluster_basis >( A.row_is() );
    auto  ncolcb  = std::make_unique< cluster_basis >( A.col_is() );
    auto  Xp_row  = blas::matrix< value_t >( A.nrows(), 0 );
    auto  Xp_col  = blas::matrix< value_t >( A.ncols(), 0 );
    
    detail::build_nested_cluster_basis( *nrowcb, srowcb, Xp_row, basisapx, acc, compress );
    detail::build_nested_cluster_basis( *ncolcb, scolcb, Xp_col, basisapx, acc, compress );

    { int  id = 0;  detail::set_ids( *nrowcb, id ); }
    { int  id = 0;  detail::set_ids( *ncolcb, id ); }

    //
    // construct uniform lowrank matrices with given cluster bases
    //
    
    auto  M = detail::build_h2( A, *nrowcb, *ncolcb, acc, compress );
    
    return  { std::move( nrowcb ), std::move( ncolcb ), std::move( M ) };
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
// return identity matrix with same structure as M
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
identity ( const Hpro::TMatrix< value_t > &  M )
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
                    auto  B_ij = hlr::seq::matrix::identity( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  N = M.copy();

        N->scale( 0 );
        
        if ( M.row_is() == M.col_is() )
        {
            //
            // fill identity to diagonal
            //
            
            auto  D  = ptrcast( N.get(), matrix::dense_matrix< value_t > );
            auto  DD = D->mat();
            
            HLR_ASSERT( ! D->is_compressed() );
            
            for ( uint  i = 0; i < DD.nrows(); ++i )
                DD(i,i) = 1;
        }// if
        
        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        HLR_ASSERT( M.row_is() != M.col_is() );
        
        return M.copy_struct();
    }// if
    else
        HLR_ERROR( "todo" );
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

        N->set_id( M.id() );
        
        return N;
    }// if
    else
    {
        // assuming non-structured block
        auto  N = M.copy();

        N->set_id( M.id() );
        
        return N;
    }// else
}

//
// return structural copy of matrix (no data)
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_struct ( const Hpro::TMatrix< value_t > &  M )
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
                    auto  B_ij = hlr::seq::matrix::copy_struct( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        N->set_id( M.id() );
        
        return N;
    }// if
    else
    {
        // assuming non-structured block
        auto  N = M.copy_struct();

        N->set_id( M.id() );
        
        return N;
    }// else
}

//
// return truncated copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy ( const Hpro::TMatrix< value_t > &  M,
       const accuracy &                  acc )
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

        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        auto  N = M.copy();
        
        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( Hpro::is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, Hpro::TRkMatrix< value_t > );
        auto  U = blas::copy( blas::mat_U( R ) );
        auto  V = blas::copy( blas::mat_V( R ) );
        auto  N = std::make_unique< matrix::lrmatrix< value_t > >( R->row_is(), R->col_is(), std::move( U ), std::move( V ) );
            
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
copy_mixedprec ( const hpro::TMatrix< value_t > &  M )
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
    else if ( Hpro::is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, Hpro::TRkMatrix< value_t > );
        auto  U = blas::copy( blas::mat_U( R ) );
        auto  V = blas::copy( blas::mat_V( R ) );
        auto  N = std::make_unique< matrix::lrsvmatrix< value_t > >( R->row_is(), R->col_is(), std::move( U ), std::move( V ) );
            
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
        return M.copy();
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
// return copy of matrix with lrmatrix< value_t > replaced by tiled_lrmatrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_tiled ( const Hpro::TMatrix< value_t > &  M,
             const size_t                      ntile )
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
    else if ( matrix::is_lowrank( M ) )
    {
        //
        // copy low-rank data into tiled form
        //

        auto  RM = cptrcast( & M, matrix::lrmatrix< value_t > );
        auto  R  = std::make_unique< tiled_lrmatrix< value_t > >( RM->row_is(), RM->col_is(), ntile, RM->U(), RM->V() );

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
// return copy of matrix with tiled_lrmatrix replaced by lrmatrix
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
        auto  U  = to_dense( RM->U() );
        auto  V  = to_dense( RM->V() );
        auto  R  = std::make_unique< matrix::lrmatrix< value_t > >( RM->row_is(), RM->col_is(), std::move( U ), std::move( V ) );

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
                auto  B_ii = hlr::seq::matrix::copy_diag( * BM->block( i, i ) );
                    
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
// return diagonal of M
//
namespace detail
{

template < typename value_t >
void
diagonal ( const Hpro::TMatrix< value_t > &  M,
           blas::vector< value_t > &         d )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        for ( uint  i = 0; i < std::min( B->nblock_rows(), B->nblock_cols() ); ++i )
        {
            if ( ! is_null( B->block( i, i ) ) )
                diagonal( * B->block( i, i ), d );
        }// for
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  D  = cptrcast( &M, matrix::dense_matrix< value_t > );
        auto  DD = D->mat();

        for ( uint  i = 0; i < std::min( DD.nrows(), DD.ncols() ); ++i )
            d( M.row_ofs() + i ) = DD(i,i);
    }// if
    else
        HLR_ERROR( "todo" );
}

}// namespace detail

template < typename value_t >
blas::vector< value_t >
diagonal ( const Hpro::TMatrix< value_t > &  M )
{
    auto  d = blas::vector< value_t >( std::min( M.nrows(), M.ncols() ) );

    detail::diagonal( M, d );

    return d;
}

//
// return copy of (block-wise) lower-left part of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_ll ( const Hpro::TMatrix< value_t > &  M,
          const Hpro::diag_type_t           diag = Hpro::general_diag )
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
                HLR_ASSERT( matrix::is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), matrix::dense_matrix< value_t > );

                D->set_matrix( blas::identity< value_t >( D->nrows() ) );
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
                HLR_ASSERT( matrix::is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), matrix::dense_matrix< value_t > );

                D->set_matrix( blas::identity< value_t >( D->nrows() ) );
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
    else if ( matrix::is_lowrank( & M ) )
    {
        auto  R = ptrcast( & M, matrix::lrmatrix< value_t > );

        R->set_lrmat( std::move( blas::matrix< value_t >() ),
                      std::move( blas::matrix< value_t >() ) );
    }// if
    else if ( matrix::is_dense( & M ) )
    {
        auto  D = ptrcast( & M, matrix::lrmatrix< value_t > );

        blas::fill( D->mat_direct(), value_t(0) );
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
copy_uniform ( const Hpro::TMatrix< value_t > &   M,
               shared_cluster_basis< value_t > &  rowcb,
               shared_cluster_basis< value_t > &  colcb )
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
    else if ( matrix::is_lowrank( M ) )
    {
        //
        // project into row/column cluster basis:
        //
        //   M = A·B^H = (U·U^H·A) (V·V^H·B)^H
        //             = U · (U^H·A)·(V^H·B)^H · V^H
        //             = U · S · V^H   with  S = (U^H·A)·(V^H·B)^H
        //
        
        auto  R  = cptrcast( &M, matrix::lrmatrix< value_t > );

        auto  UA = rowcb.transform_forward( R->U() );
        auto  VB = colcb.transform_forward( R->V() );
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
        auto  SR = std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );

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
        bool  all_lowrank = true;
        uint  k_sum       = 0;

        B->copy_struct_from( BM );

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

//
// same as above but use lrsvmatrix for lowrank blocks
//
template < typename                    value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
coarsen_sv ( const Hpro::TMatrix< value_t > &  M,
             const accuracy &                  acc,
             const approx_t &                  approx )
{
    if ( is_blocked( M ) )
    {
        auto  BM          = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N           = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B           = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );
        bool  all_lowrank = true;
        uint  k_sum       = 0;

        B->copy_struct_from( BM );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = coarsen( * BM->block( i, j ), acc, approx );
                    
                    if ( matrix::is_lowrank_sv( *B_ij ) )
                        k_sum += cptrcast( B_ij.get(), matrix::lrsvmatrix< value_t > )->rank();
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

            auto  [ U, S, V ] = approx.approx_sv( U_sum, V_sum, acc );
            auto  R           = std::make_unique< matrix::lrsvmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( S ), std::move( V ) );

            if ( R->byte_size() <= B->byte_size() )
                return R;
        }// if
        
        return N;
    }// if
    else if ( matrix::is_lowrank_sv( M ) )
    {
        return M.copy();
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  R = convert_to_lowrank_sv( M, acc, approx );

        if ( R->byte_size() <= M.byte_size() )
            return R;

        return M.copy();
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

}}}// namespace hlr::seq::matrix

#endif // __HLR_SEQ_MATRIX_HH

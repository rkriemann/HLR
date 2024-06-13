#ifndef __HLR_OMP_DETAIL_MATRIX_HH
#define __HLR_OMP_DETAIL_MATRIX_HH
//
// Project     : HLR
// Module      : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>

#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/lrsmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>

namespace hlr { namespace omp { namespace matrix { namespace detail {

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
build_task ( const Hpro::TBlockCluster *  bct,
             const coeff_t &              coeff,
             const lrapx_t &              lrapx,
             const Hpro::TTruncAcc &      acc,
             const size_t                 nseq )
{
    static_assert( std::is_same< typename coeff_t::value_t,
                   typename lrapx_t::value_t >::value,
                   "coefficient function and low-rank approximation must have equal value type" );

    HLR_ASSERT( bct != nullptr );

    using  value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto        M     = std::unique_ptr< Hpro::TMatrix< value_t > >();
    const auto  rowis = bct->is().row_is();
    const auto  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return seq::matrix::build( bct, coeff, lrapx, acc );
        
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            // auto  T = lrapx.build( bct, Hpro::absolute_prec( acc.abs_eps() * std::sqrt( double(rowis.size() * colis.size()) ) ) );
            M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( bct, acc( rowis, colis ) ) );

            if ( Hpro::is_lowrank( *M ) )
            {
                auto  R = ptrcast( M.get(), Hpro::TRkMatrix< value_t > );

                M = std::move( std::make_unique< hlr::matrix::lrmatrix< value_t > >( rowis, colis,
                                                                                     std::move( blas::mat_U( R ) ),
                                                                                     std::move( blas::mat_V( R ) ) ) );
            }// if
        }// if
        else
        {
            M = coeff.build( rowis, colis );

            if ( Hpro::is_dense( *M ) )
            {
                auto  D = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );

                M = std::move( std::make_unique< hlr::matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) ) );
            }// if
        }// else
    }// if
    else if ( std::min( rowis.size(), colis.size() ) <= nseq )
    {
        M = hlr::seq::matrix::build( bct, coeff, lrapx, acc, nseq );
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
    }// else

    // copy properties from the cluster
    // M->set_cluster_force( bct );
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
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
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );

    HLR_ASSERT( bct != nullptr );

    using  value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto        M     = std::unique_ptr< Hpro::TMatrix< value_t > >();
    const auto  rowis = bct->is().row_is();
    const auto  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return seq::matrix::build( bct, coeff, lrapx, acc );
        
    if ( bct->is_leaf() )
    {
        auto  lacc = acc( rowis, colis );
        
        if ( bct->is_adm() )
        {
            // auto  T = lrapx.build( bct, Hpro::absolute_prec( acc.abs_eps() * std::sqrt( double(rowis.size() * colis.size()) ) ) );
            auto  T = lrapx.build( bct, lacc );

            if ( Hpro::is_lowrank( *T ) )
            {
                auto  R  = ptrcast( T.get(), Hpro::TRkMatrix< value_t > );
                auto  zR = std::make_unique< hlr::matrix::lrmatrix< value_t > >( rowis, colis,
                                                                                 std::move( blas::mat_U( R ) ),
                                                                                 std::move( blas::mat_V( R ) ) );
                
                zR->compress( lacc );
                M = std::move( zR );
            }// if
            else
            {
                M = std::move( T );
            }// else
        }// if
        else
        {
            M = coeff.build( rowis, colis );

            if ( Hpro::is_dense( *M ) )
            {
                auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                auto  zD = std::make_unique< hlr::matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) );

                zD->compress( lacc );
                M = std::move( zD );
            }// if
        }// else
    }// if
    else if ( std::min( rowis.size(), colis.size() ) <= nseq )
    {
        M = hlr::seq::matrix::build_compressed( bct, coeff, lrapx, acc, nseq );
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
        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2) default(shared)
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( bct->son( i, j ) != nullptr )
                    {
                        auto  B_ij = build_compressed( bct->son( i, j ), coeff, lrapx, acc, nseq );
                        
                        B->set_block( i, j, B_ij.release() );
                    }// if
                }// for
            }// for
        }// omp taskgroup
    }// else

    // copy properties from the cluster
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
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
                  const size_t                 nseq )
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );

    HLR_ASSERT( bct != nullptr );

    using  value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto        M     = std::unique_ptr< Hpro::TMatrix< value_t > >();
    const auto  rowis = bct->is().row_is();
    const auto  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return seq::matrix::build( bct, coeff, lrapx, acc );
        
    if ( bct->is_leaf() )
    {
        auto  lacc = acc( rowis, colis );
        
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
            else if ( hlr::matrix::is_lowrank( *M ) )
            {
                auto  R  = ptrcast( M.get(), hlr::matrix::lrmatrix< value_t > );
                auto  zR = std::make_unique< hlr::matrix::lrsvmatrix< value_t > >( rowis, colis, R->U(), R->V() );
                
                zR->compress( lacc );
                M = std::move( zR );
            }// if
            else if ( hlr::matrix::is_lowrank_sv( *M ) )
            {
                auto  R = ptrcast( M.get(), hlr::matrix::lrsvmatrix< value_t > );

                R->compress( lacc );
            }// if
            else if ( Hpro::is_dense( *M ) )
            {
                auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                auto  zD = std::make_unique< hlr::matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) );

                zD->compress( lacc );
                M = std::move( zD );
            }// if
            else if ( hlr::matrix::is_dense( *M ) )
            {
                auto  D = ptrcast( M.get(), hlr::matrix::dense_matrix< value_t > );

                D->compress( lacc );
            }// if
            else
            {
                HLR_LOG( 0, M->typestr() );
            }// else
        }// if
        else
        {
            M = coeff.build( rowis, colis );

            if ( Hpro::is_dense( *M ) )
            {
                auto  D  = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                auto  zD = std::make_unique< hlr::matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) );

                zD->compress( lacc );
                M = std::move( zD );
            }// if
            else if ( hlr::matrix::is_dense( *M ) )
            {
                auto  D = ptrcast( M.get(), hlr::matrix::dense_matrix< value_t > );

                D->compress( lacc );
            }// if
            else
            {
                HLR_LOG( 0, M->typestr() );
            }// else
        }// else
    }// if
    else if ( std::min( rowis.size(), colis.size() ) <= nseq )
    {
        M = hlr::seq::matrix::build_mixedprec( bct, coeff, lrapx, acc, nseq );
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
        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2) default(shared)
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( bct->son( i, j ) != nullptr )
                    {
                        auto  B_ij = build_mixedprec( bct->son( i, j ), coeff, lrapx, acc, nseq );
                        
                        B->set_block( i, j, B_ij.release() );
                    }// if
                }// for
            }// for
        }// omp taskgroup
    }// else

    // copy properties from the cluster
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

//
// build representation of nearfield of dense matrix with
// matrix structure defined by <bct>, matrix coefficients
// defined by <coeff>
//
template < typename coeff_t >
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build_nearfield ( const Hpro::TBlockCluster *  bct,
                  const coeff_t &              coeff,
                  const size_t                 nseq )
{
    using  value_t = typename coeff_t::value_t;
    
    HLR_ASSERT( bct != nullptr );

    //
    // decide upon cluster type, how to construct matrix
    //

    auto        M     = std::unique_ptr< Hpro::TMatrix< value_t > >();
    const auto  rowis = bct->is().row_is();
    const auto  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return seq::matrix::build_nearfield( bct, coeff );
        
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
            return nullptr;
        else
        {
            M = coeff.build( rowis, colis );

            if ( Hpro::is_dense( *M ) )
            {
                auto  D = ptrcast( M.get(), Hpro::TDenseMatrix< value_t > );

                M = std::move( std::make_unique< hlr::matrix::dense_matrix< value_t > >( rowis, colis, std::move( blas::mat( D ) ) ) );
            }// if
        }// else
    }// if
    else if ( std::min( rowis.size(), colis.size() ) <= nseq )
    {
        M = hlr::seq::matrix::build_nearfield( bct, coeff, nseq );
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
        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2) default(shared)
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( bct->son( i, j ) != nullptr )
                    {
                        auto  B_ij = build_nearfield( bct->son( i, j ), coeff, nseq );
                        
                        B->set_block( i, j, B_ij.release() );
                    }// if
                }// for
            }// for
        }// omp taskgroup
    }// else

    // copy properties from the cluster
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
build ( const Hpro::TBlockCluster &             bct,
        const Hpro::TSparseMatrix< value_t > &  S,
        const Hpro::TTruncAcc &                 acc,
        const approx_t &                        apx,
        const size_t                            nseq ) // ignored
{
    // std::cout << "build    : " << bct.to_string() << std::endl;
    
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
            M = hlr::matrix::convert_to_lowrank( *S_bct, acc, apx );
        }// if
        else
        {
            M = hlr::matrix::convert_to_dense< value_t >( *S_bct );
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
        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2) default(shared)
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( bct.son( i, j ) != nullptr )
                    {
                        auto  B_ij = build( bct.son( i, j ), S, acc, apx, nseq );
                        
                        B->set_block( i, j, B_ij.release() );
                    }// if
                }// for
            }// for
        }// omp taskgroup
    }// else

    // M->set_cluster_force( bct );
    M->set_id( bct.id() );
    M->set_procs( bct.procs() );

    return M;
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
    // std::cout << "build_nd : " << bct.to_string() << std::endl;
    
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

        const auto  nbr = B->nblock_rows();
        const auto  nbc = B->nblock_cols();

        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2) default(shared)
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( ! is_null( bct.son( i, j ) ) )
                    {
                        if ( i == j )
                        {
                            if ( i < std::min( nbr, nbc ) - 1 )
                            {
                                //
                                // recurse for diagonal
                                //
                                auto  B_ij = build_nd( *bct.son( i, j ), S, acc, apx, nseq );
                            
                                B->set_block( i, j, B_ij.release() );
                            }// if
                            else
                            {
                                //
                                // standard construction for interface-interface couplings
                                //
                                auto  B_ij = build( *bct.son( i, j ), S, acc, apx, nseq );
                            
                                B->set_block( i, j, B_ij.release() );
                            }// else
                        }// if
                        else
                        {
                            if (( i == nbr-1 ) || ( j == nbc-1 ))
                            {
                                //
                                // standard construction for domain-interface couplings
                                //
                                auto  B_ij = build( *bct.son( i, j ), S, acc, apx, nseq );
                            
                                B->set_block( i, j, B_ij.release() );
                            }// if
                        }// else
                    }// if
                }// for
            }// for
        }// omp taskgroup
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
           typename approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_sparse ( const Hpro::TBlockCluster &             bct,
               const Hpro::TSparseMatrix< value_t > &  S,
               const Hpro::TTruncAcc &                 acc,
               const approx_t &                        apx,
               const size_t                            nseq = Hpro::CFG::Arith::max_seq_size ) // ignored
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
            M = hlr::matrix::convert_to_lowrank( *S_bct, acc, apx );
        }// if
        else if ( is_sparse( *S_bct ) )
        {
            // M = std::make_unique< hlr::matrix::sparse_matrix< value_t > >( * ptrcast( S_bct.get(), Hpro::TSparseMatrix< value_t > ) );
            M = std::move( S_bct );
        }// else
        else
        {
            M = hlr::matrix::convert_to_dense< value_t >( *S_bct );
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
        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2) default(shared)
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
        }// omp taskgroup
    }// else

    // M->set_cluster_force( bct );
    M->set_id( bct.id() );
    M->set_procs( bct.procs() );

    return M;
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
            std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > > >
build_uniform_lvl ( const Hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const Hpro::TTruncAcc &      acc )
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< typename coeff_t::value_t, typename basisapx_t::value_t >,
                   "coefficient function and basis approximation must have equal value type" );
    
    assert( bct != nullptr );

    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;
    using basis_map_t   = std::unordered_map< indexset, cluster_basis *, indexset_hash >;
    using lrmat_map_t   = std::unordered_map< indexset, std::list< hlr::matrix::lrmatrix< value_t > * >, indexset_hash >;
    using bmat_map_t    = std::unordered_map< Hpro::idx_t, Hpro::TBlockMatrix< value_t > * >;

    //
    // go BFS-style through block cluster tree and construct leaves per level
    // then convert lowrank to uniform lowrank while constructing bases
    //

    // TODO: handle case of global lowrank matrix
    HLR_ASSERT( ! bct->is_adm() );
    
    auto  rowcb_root = std::unique_ptr< cluster_basis >();
    auto  colcb_root = std::unique_ptr< cluster_basis >();

    auto  rowcb_map  = basis_map_t();
    auto  colcb_map  = basis_map_t();

    auto  M_root     = std::unique_ptr< Hpro::TMatrix< value_t > >();

    auto  nodes      = std::deque< const Hpro::TBlockCluster * >{ bct };
    auto  bmat_map   = bmat_map_t();

    auto  bmtx       = std::mutex(); // for bmat_map
    auto  cmtx       = std::mutex(); // for children list
    auto  lmtx       = std::mutex(); // for row/col map lists
    auto  cbmtx      = std::mutex(); // for rowcb/colcb map lists

    //
    // local function to set up hierarchy (parent <-> M)
    //
    auto  insert_hier = [&] ( const Hpro::TBlockCluster *                    node,
                              std::unique_ptr< Hpro::TMatrix< value_t > > &  M )
    {
        if ( is_null( node->parent() ) )
        {
            M_root = std::move( M );
        }// if
        else
        {
            auto  parent   = node->parent();
            auto  M_parent = bmat_map_t::mapped_type( nullptr );

            {
                auto  lock = std::scoped_lock( bmtx );
                        
                M_parent = bmat_map.at( parent->id() );
            }

            for ( uint  i = 0; i < parent->nrows(); ++i ) 
            {
                for ( uint  j = 0; j < parent->ncols(); ++j )
                {
                    if ( parent->son( i, j ) == node )
                    {
                        M_parent->set_block( i, j, M.release() );
                        return;
                    }// if
                }// for
            }// for
        }// if
    };

    //
    // local function to create cluster basis objects (with hierarchy)
    //
    auto  create_cb = [&] ( const Hpro::TBlockCluster *  node )
    {
        //
        // build row/column cluster basis objects and set up
        // cluster bases hierarchy
        //

        auto              rowcl = node->rowcl();
        auto              colcl = node->colcl();
        cluster_basis *   rowcb = nullptr;
        cluster_basis *   colcb = nullptr;
        std::scoped_lock  lock( cbmtx );
                    
        if ( rowcb_map.find( *rowcl ) == rowcb_map.end() )
        {
            rowcb = new cluster_basis( *rowcl );
            rowcb->set_nsons( rowcl->nsons() );

            rowcb_map.emplace( *rowcl, rowcb );
        }// if
        else
            rowcb = rowcb_map.at( *rowcl );
                    
        if ( colcb_map.find( *colcl ) == colcb_map.end() )
        {
            colcb = new cluster_basis( *colcl );
            colcb->set_nsons( colcl->nsons() );
            colcb_map.emplace( *colcl, colcb );
        }// if
        else
            colcb = colcb_map.at( *colcl );

        if ( is_null( node->parent() ) )
        {
            rowcb_root.reset( rowcb_map[ *rowcl ] );
            colcb_root.reset( colcb_map[ *colcl ] );
        }// if
        else
        {
            auto  parent     = node->parent();
            auto  row_parent = parent->rowcl();
            auto  col_parent = parent->colcl();

            for ( uint  i = 0; i < row_parent->nsons(); ++i )
            {
                if ( row_parent->son( i ) == rowcl )
                {
                    rowcb_map.at( *row_parent )->set_son( i, rowcb );
                    break;
                }// if
            }// for

            for ( uint  i = 0; i < col_parent->nsons(); ++i )
            {
                if ( col_parent->son( i ) == colcl )
                {
                    colcb_map.at( *col_parent )->set_son( i, colcb );
                    break;
                }// if
            }// for
        }// else
    };

    //
    // level-wise iteration for matrix construction
    //
    
    while ( ! nodes.empty() )
    {
        auto  children = decltype( nodes )();
        auto  rowmap   = lrmat_map_t();
        auto  colmap   = lrmat_map_t();
        auto  lrmat    = std::deque< Hpro::TMatrix< value_t > * >();
        
        #pragma omp taskloop default(shared)
        for ( auto  node : nodes )
        {
            auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();

            if ( node->is_leaf() )
            {
                if ( node->is_adm() )
                {
                    M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( node, acc ) );
                        
                    if ( is_lowrank( *M ) )
                    {
                        auto  R    = ptrcast( M.get(), hlr::matrix::lrmatrix< value_t > );
                        auto  lock = std::scoped_lock( lmtx );
                            
                        lrmat.push_back( R );
                        rowmap[ M->row_is() ].push_back( R );
                        colmap[ M->col_is() ].push_back( R );
                    }// if
                }// if
                else 
                    M = coeff.build( node->is().row_is(), node->is().col_is() );
            }// if
            else
            {
                // collect children
                {
                    auto  lock = std::scoped_lock( cmtx );
                            
                    for ( uint  i = 0; i < node->nrows(); ++i )
                        for ( uint  j = 0; j < node->ncols(); ++j )
                            if ( node->son( i, j ) != nullptr )
                                children.push_back( node->son( i, j ) );
                }

                M = std::make_unique< Hpro::TBlockMatrix< value_t > >( node );
        
                auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

                // make sure, block structure is correct
                if (( B->nblock_rows() != node->nrows() ) ||
                    ( B->nblock_cols() != node->ncols() ))
                    B->set_block_struct( node->nrows(), node->ncols() );

                // make value type consistent in block matrix and sub blocks
                B->adjust_value_type();

                // remember all block matrices for setting up hierarchy
                {
                    auto  lock = std::scoped_lock( bmtx );
                        
                    bmat_map[ node->id() ] = B;
                }
            }// else

            M->set_id( node->id() );
            M->set_procs( node->procs() );

            insert_hier( node, M );
            create_cb( node );
        }// omp taskloop for
        
        nodes = std::move( children );
        
        #pragma omp taskgroup
        {
            #pragma omp task default(shared)
            {
                //
                // construct row bases for all block rows constructed on this level
                //

                auto  rowiss = std::deque< indexset >();

                for ( auto  [ is, matrices ] : rowmap )
                    rowiss.push_back( is );

                #pragma omp taskloop default(shared)
                for ( auto  is : rowiss )
                {
                    auto  matrices = rowmap.at( is );
                    
                    if ( matrices.size() == 0 )
                        continue;

                    //
                    // compute column basis for
                    //
                    //   ( U₀·V₀'  U₁·V₁'  U₂·V₂'  … ) =
                    //
                    //                  ⎛ V₀'        ⎞
                    //   ( U₀ U₁ U₂ … ) ⎜    V₁'     ⎟ =
                    //                  ⎜       V₂'  ⎟
                    //                  ⎝          … ⎠
                    //
                    //                  ⎛ Q₀·R₀             ⎞'
                    //   ( U₀ U₁ U₂ … ) ⎜      Q₁·R₁        ⎟ =
                    //                  ⎜           Q₂·R₂   ⎟
                    //                  ⎝                 … ⎠
                    //
                    //                  ⎛⎛Q₀     ⎞ ⎛R₀     ⎞⎞'
                    //   ( U₀ U₁ U₂ … ) ⎜⎜  Q₁   ⎟·⎜  R₁   ⎟⎟ =
                    //                  ⎜⎜    Q₂ ⎟ ⎜    R₂ ⎟⎟
                    //                  ⎝⎝      …⎠ ⎝      …⎠⎠
                    //
                    // Since diag(Q_i) is orthogonal, it can be omitted for row bases
                    // computation, leaving
                    //
                    //                  ⎛R₀     ⎞'                 
                    //   ( U₀ U₁ U₂ … ) ⎜  R₁   ⎟ = ( U₀·R₀' U₁·R₁' U₂·R₂' … )
                    //                  ⎜    R₂ ⎟                  
                    //                  ⎝      …⎠                  
                    //
                    // of which a column basis is computed.
                    //

                    //
                    // form U = ( U₀·R₀' U₁·R₁' U₂·R₁' … )
                    //
            
                    size_t  nrows_U = is.size();
                    size_t  ncols_U = 0;

                    for ( auto &  R : matrices )
                        ncols_U += R->rank();

                    auto    U   = blas::matrix< value_t >( nrows_U, ncols_U );
                    size_t  pos = 0;

                    for ( auto &  R : matrices )
                    {
                        // R = U·V' = W·T·X'
                        auto  U_i = blas::mat_U< value_t >( R );
                        auto  V_i = blas::copy( blas::mat_V< value_t >( R ) );
                        auto  R_i = blas::matrix< value_t >();
                        auto  k   = R->rank();
                
                        blas::qr( V_i, R_i );

                        auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );
                        auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                        blas::copy( UR_i, U_sub );
                
                        pos += k;
                    }// for

                    //
                    // QR of S and computation of row basis
                    //

                    auto  Un = basisapx.column_basis( U, acc );
            
                    // finally assign to cluster basis object
                    // (no change to "rowcb_map", therefore no lock)
                    rowcb_map.at( is )->set_basis( std::move( Un ) );
                }// omp taskloop for
            }// omp task

            #pragma omp task default(shared)
            {
                //
                // construct column bases for all block columns constructed on this level
                //

                auto  coliss = std::deque< indexset >();
            
                for ( auto  [ is, matrices ] : colmap )
                    coliss.push_back( is );

                #pragma omp taskloop default(shared)
                for ( auto  is : coliss )
                {
                    auto  matrices = colmap.at( is );

                    if ( matrices.size() == 0 )
                        continue;

                    //
                    // compute column basis for
                    //
                    //   ⎛U₀·V₀'⎞ 
                    //   ⎜U₁·V₁'⎟
                    //   ⎜U₂·V₂'⎟
                    //   ⎝  …   ⎠
                    //
                    // or row basis of
                    //
                    //   ⎛U₀·V₀'⎞' 
                    //   ⎜U₁·V₁'⎟ = ( V₀·U₀'  V₁·U₁'  V₂·U₂'  … ) =
                    //   ⎜U₂·V₂'⎟
                    //   ⎝  …   ⎠
                    //
                    //                  ⎛ U₀      ⎞'
                    //   ( V₀ V₁ V₂ … ) ⎜   U₁    ⎟ =
                    //                  ⎜     U₂  ⎟
                    //                  ⎝       … ⎠
                    //
                    //                  ⎛ Q₀·R₀               ⎞'
                    //   ( V₀ V₁ V₂ … ) ⎜       Q₁·R₁         ⎟ =
                    //                  ⎜             Q₂·R₂   ⎟
                    //                  ⎝                   … ⎠
                    //
                    //                  ⎛⎛Q₀     ⎞ ⎛R₀     ⎞⎞'
                    //   ( V₀ V₁ V₂ … ) ⎜⎜  Q₁   ⎟·⎜  R₁   ⎟⎟ =
                    //                  ⎜⎜    Q₂ ⎟ ⎜    R₂ ⎟⎟
                    //                  ⎝⎝      …⎠ ⎝      …⎠⎠
                    //
                    // Since diag(Q_i) is orthogonal, it can be omitted for column bases
                    // computation, leaving
                    //
                    //                  ⎛R₀     ⎞'                
                    //   ( V₀ V₁ V₂ … ) ⎜  R₁   ⎟ = ( V₀·R₀' V₁·R₁' V₂·R₂' … )
                    //                  ⎜    R₂ ⎟                
                    //                  ⎝      …⎠
                    //
                    // of which a column basis is computed.
                    //

                    //
                    // form matrix V = ( V₀·R₀' V₁·R₁' V₂·R₂' … )
                    //

                    size_t  nrows_V = is.size();
                    size_t  ncols_V = 0;

                    for ( auto &  R : matrices )
                        ncols_V += R->rank();

                    auto    V   = blas::matrix< value_t >( nrows_V, ncols_V );
                    size_t  pos = 0;

                    for ( auto &  R : matrices )
                    {
                        // R' = (U·V')' = V·U' = X·T'·W'
                        auto  V_i = blas::copy( blas::mat_V< value_t >( R ) );
                        auto  U_i = blas::copy( blas::mat_U< value_t >( R ) );
                        auto  R_i = blas::matrix< value_t >();
                        auto  k   = R->rank();
                
                        blas::qr( U_i, R_i );

                        auto  VR_i  = blas::prod( V_i, blas::adjoint( R_i ) );
                        auto  V_sub = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                        blas::copy( VR_i, V_sub );
                
                        pos += k;
                    }// for

                    auto  Vn = basisapx.column_basis( V, acc );

                    // finally assign to cluster basis object
                    // (no change to "colcb_map", therefore no lock)
                    colcb_map.at( is )->set_basis( std::move( Vn ) );
                }// omp taskloop for
            }// omp task
        }// omp taskgroup

        //
        // now convert all blocks on this level
        //

        #pragma omp taskloop default(shared)
        for ( auto  M : lrmat )
        {
            auto  R     = ptrcast( M, hlr::matrix::lrmatrix< value_t > );
            auto  rowcb = rowcb_map.at( R->row_is() );
            auto  colcb = colcb_map.at( R->col_is() );
            auto  Un    = rowcb->basis();
            auto  Vn    = colcb->basis();

            //
            // R = U·V' ≈ Un (Un' U V' Vn) Vn'
            //          = Un S Vn'  with  S = Un' U V' Vn
            //

            auto  UnU = blas::prod( blas::adjoint( Un ), blas::mat_U< value_t >( R ) );
            auto  VnV = blas::prod( blas::adjoint( Vn ), blas::mat_V< value_t >( R ) );
            auto  S   = blas::prod( UnU, blas::adjoint( VnV ) );

            auto  RU  = std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( R->row_is(),
                                                                                      R->col_is(),
                                                                                      *rowcb,
                                                                                      *colcb,
                                                                                      std::move( S ) );
            
            // replace standard lowrank block by uniform lowrank block
            R->parent()->replace_block( R, RU.release() );
            delete R;
        }// omp taskloop for
    }// while
    
    return { std::move( rowcb_root ),
             std::move( colcb_root ),
             std::move( M_root ) };
}

//
// recursively build uniform H-matrix while also constructing row/column cluster basis
// by updating bases after constructing low-rank blocks
//
template < typename value_t >
using  matrix_list_t = std::vector< Hpro::TMatrix< value_t > * >;
template < typename value_t >
using  matrix_map_t  = std::unordered_map< indexset, matrix_list_t< value_t >, indexset_hash >;
using  mutex_map_t   = std::unordered_map< indexset, std::mutex, indexset_hash >;

struct rec_basis_data_t
{
    // maps indexsets to set of uniform matrices sharing corresponding cluster basis
    // and their mutexes
    matrix_map_t   rowmap, colmap;
    std::mutex     rowmapmtx, colmapmtx;

    //
    // extend row basis <cb> by block W·T·X' (X is not needed for computation)
    //
    // - identical to implementation in "arith/detail/uniform.hh" but thread-safe,
    //   hence, for details look into original code
    //
    template < typename value_t,
               typename basis_approx_t >
    blas::matrix< value_t >
    compute_extended_basis ( const cluster_basis< value_t > &  cb,
                             const blas::matrix< value_t > &   W,
                             const blas::matrix< value_t > &   T,
                             const Hpro::TTruncAcc &           acc,
                             const basis_approx_t &            basisapx,
                             matrix_map_t &                    matmap,
                             std::mutex &                      matmapmtx,
                             const matop_t                     op )
    {
        using  real_t = Hpro::real_type_t< value_t >;

        // zero basis implies empty matrix list
        if ( cb.basis().ncols() == 0 )
            return std::move( blas::copy( W ) );
            
        //
        // copy uniform matrices for basis to local list for minimal blocking
        //
        
        auto  uni_mats = matrix_list_t();

        {
            auto  lock = std::scoped_lock( matmapmtx );

            HLR_ASSERT( matmap.find( cb.is() ) != matmap.end() );
            
            for ( auto  M_i : matmap.at( cb.is() ) )
                uni_mats.push_back( M_i );
        }

        //
        // collect scaled coupling matrices and filter out zero couplings
        //

        auto    couplings = std::list< blas::matrix< value_t > >();
        size_t  nrows_S   = T.ncols();
        auto    cmtx      = std::mutex();

        #pragma omp taskloop default(shared)
        for ( auto  M_i : uni_mats )
        {
            const auto  R_i = cptrcast( M_i, uniform_lrmatrix< value_t > );
            auto        S_i = blas::matrix< value_t >();
                        
            {
                auto  lock = std::scoped_lock( M_i->mutex() );

                S_i = std::move( blas::copy( blas::mat_view( op, R_i->coeff() ) ) );
            }
                        
            HLR_ASSERT( S_i.ncols() == cb.basis().ncols() );
            
            const auto  norm = norm::spectral( S_i );
                        
            if ( norm != real_t(0) )
            {
                blas::scale( value_t(1) / norm, S_i );

                {
                    auto  lock = std::scoped_lock( cmtx );
                    
                    nrows_S += S_i.nrows();
                    couplings.push_back( std::move( S_i ) );
                }
            }// if
        }// omp taskloop for

        //
        // assemble all scaled coupling matrices into joined matrix
        //

        auto    U   = cb.basis();
        auto    Ue  = blas::join_row< value_t >( { U, W } );
        auto    S   = blas::matrix< value_t >( nrows_S, Ue.ncols() );
        size_t  pos = 0;
            
        for ( auto  S_i : couplings )
        {
            HLR_ASSERT( pos + S_i.nrows() <= S.nrows() );
            HLR_ASSERT( S_i.ncols() == U.ncols() );
            
            auto  S_sub = blas::matrix< value_t >( S,
                                                   blas::range( pos, pos + S_i.nrows()-1 ),
                                                   blas::range( 0, U.ncols() - 1 ) );
                        
            blas::copy( S_i, S_sub );
            pos += S_i.nrows();
        }// for

        //
        // add part from W·T·X'
        //
        
        auto  S_i  = blas::copy( blas::mat_view( op, T ) );
        auto  norm = norm::spectral( T );
            
        if ( norm != real_t(0) )
            blas::scale( value_t(1) / norm, S_i );
            
        HLR_ASSERT( pos + S_i.nrows() <= S.nrows() );
        HLR_ASSERT( S_i.ncols() == Ue.ncols() - U.ncols() );
        
        auto  S_sub = blas::matrix< value_t >( S,
                                               blas::range( pos, pos + S_i.nrows()-1 ),
                                               blas::range( U.ncols(), Ue.ncols() - 1 ) );
            
        blas::copy( S_i, S_sub );
        
        //
        // form product Ue·S and compute column basis
        //
            
        auto  R = blas::matrix< value_t >();
        
        blas::qr( S, R, false );

        auto  UeR = blas::prod( Ue, blas::adjoint( R ) );
        auto  Un  = basisapx.column_basis( UeR, acc );

        return  Un;
    }

    template < typename value_t,
               typename basis_approx_t >
    blas::matrix< value_t >
    compute_extended_row_basis ( const cluster_basis< value_t > &  cb,
                                 const blas::matrix< value_t > &   W,
                                 const blas::matrix< value_t > &   T,
                                 const Hpro::TTruncAcc &           acc,
                                 const basis_approx_t &            basisapx )
    {
        return compute_extended_basis( cb, W, T, acc, basisapx, rowmap, rowmapmtx, apply_adjoint );
    }

    template < typename value_t,
               typename basis_approx_t >
    blas::matrix< value_t >
    compute_extended_col_basis ( const cluster_basis< value_t > &  cb,
                                 const blas::matrix< value_t > &   X,
                                 const blas::matrix< value_t > &   T,
                                 const Hpro::TTruncAcc &           acc,
                                 const basis_approx_t &            basisapx )
    {
        return compute_extended_basis( cb, X, T, acc, basisapx, colmap, colmapmtx, apply_normal );
    }

    //
    // update coupling matrices for all blocks sharing basis <cb> to new basis <Un>
    //
    template < typename value_t >
    void
    update_coupling ( const cluster_basis< value_t > &  cb,
                      const blas::matrix< value_t > &   Un,
                      matrix_map_t &                    matmap,
                      std::mutex &                      matmapmtx,
                      const bool                        cols )
    {
        if ( cb.basis().ncols() == 0 )
            return;
            
        auto  uni_mats = matrix_list_t();

        {
            auto  lock = std::scoped_lock( matmapmtx );
                    
            HLR_ASSERT( matmap.find( cb.is() ) != matmap.end() );
            
            for ( auto  M_i : matmap.at( cb.is() ) )
                uni_mats.push_back( M_i );
        }
        
        auto  U  = cb.basis();
        auto  TU = blas::prod( blas::adjoint( Un ), U );

        #pragma omp taskloop default(shared)
        for ( auto  M_i : uni_mats )
        {
            auto  lock = std::scoped_lock( M_i->mutex() );
            auto  R_i  = ptrcast( M_i, uniform_lrmatrix< value_t > );
            auto  S_i  = ( cols
                           ? blas::prod( R_i->coeff(), blas::adjoint( TU ) )
                           : blas::prod( TU, R_i->coeff() ) );

            R_i->set_coeff_unsafe( std::move( S_i ) );
        }// omp taskloop for
    }

    template < typename value_t >
    void
    update_row_coupling ( const cluster_basis< value_t > &  cb,
                          const blas::matrix< value_t > &   Un )
    {
        update_coupling( cb, Un, rowmap, rowmapmtx, false );
    }

    template < typename value_t >
    void
    update_col_coupling ( const cluster_basis< value_t > &  cb,
                          const blas::matrix< value_t > &   Vn )
    {
        update_coupling( cb, Vn, colmap, colmapmtx, true );
    }
};

template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_uniform_rec ( const Hpro::TBlockCluster *                   bct,
                    const coeff_t &                               coeff,
                    const lrapx_t &                               lrapx,
                    const basisapx_t &                            basisapx,
                    const Hpro::TTruncAcc &                       acc,
                    cluster_basis< typename coeff_t::value_t > &  rowcb,
                    cluster_basis< typename coeff_t::value_t > &  colcb,
                    rec_basis_data_t &                            basis_data )
{
    using value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( bct, acc ) );

            if ( hlr::matrix::is_lowrank( *M ) )
            {
                // std::cout << bct->id() << '{' << std::endl;
                
                //
                // compute LRS representation W·T·X' = U·V' = M
                //

                auto  R  = ptrcast( M.get(), hlr::matrix::lrmatrix< value_t > );
                auto  W  = std::move( blas::mat_U< value_t >( R ) ); // reuse storage from R
                auto  X  = std::move( blas::mat_V< value_t >( R ) );
                auto  Rw = blas::matrix< value_t >();
                auto  Rx = blas::matrix< value_t >();

                #pragma omp taskgroup
                {
                    #pragma omp task default(shared)
                    blas::qr( W, Rw );
                    
                    #pragma omp task default(shared)
                    blas::qr( X, Rx );
                }// omp taskgroup

                HLR_ASSERT( Rw.ncols() != 0 );
                HLR_ASSERT( Rx.ncols() != 0 );
                
                auto  T       = blas::prod( Rw, blas::adjoint( Rx ) );
                auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );

                // std::cout << bct->id() << '<' << std::endl;

                #pragma omp taskgroup
                {
                    #pragma omp task default(shared)
                    {
                        auto  Un = basis_data.compute_extended_row_basis( rowcb, W, T, acc, basisapx );
                        
                        basis_data.update_row_coupling( rowcb, Un );
                        rowcb.set_basis( std::move( Un ) );
                    }// omp task
                
                    #pragma omp task default(shared)
                    {
                        auto  Vn = basis_data.compute_extended_col_basis( colcb, X, T, acc, basisapx );
                        
                        basis_data.update_col_coupling( colcb, Vn );
                        colcb.set_basis( std::move( Vn ) );
                    }// omp task
                }// omp taskgroup

                //
                // transform T into new bases
                //

                auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), W );
                auto  TV = blas::prod( blas::adjoint( colcb.basis() ), X );
                auto  TS = blas::prod( TU, T );
                auto  S  = blas::prod( TS, blas::adjoint( TV ) );

                auto  RU = std::make_unique< uniform_lrmatrix< value_t > >( M->row_is(), M->col_is(), rowcb, colcb, std::move( S ) );

                {
                    auto  lock_is = std::scoped_lock( basis_data.rowmapmtx,
                                                      basis_data.colmapmtx );

                    basis_data.rowmap[ rowcb.is() ].push_back( RU.get() );
                    basis_data.colmap[ colcb.is() ].push_back( RU.get() );
                }

                // std::cout << bct->id() << ':' << rowcb.basis().ncols() << ',' << RU->coeff().nrows() << ',' << RU->coeff().ncols() << ',' << colcb.basis().ncols() << ">}" << std::endl;
                /// std::cout << bct->id() << '>' << '}' << std::endl;
                M = std::move( RU );
            }// if
        }// if
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
        B->set_block_struct( bct->nrows(), bct->ncols() );

        #pragma omp taskgroup
        {
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                auto  rowcb_i = rowcb.son( i );
                
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    auto  colcb_j = colcb.son( j );
                
                    if ( ! is_null( bct->son( i, j ) ) )
                    {
                        #pragma omp task default(shared)
                        {
                            HLR_ASSERT( ! is_null_all( rowcb_i, colcb_j ) );
                        
                            auto  B_ij = build_uniform_rec( bct->son( i, j ), coeff, lrapx, basisapx, acc, *rowcb_i, *colcb_j, basis_data );
                        
                            B->set_block( i, j, B_ij.release() );
                        }// omp task
                    }// if
                }// for
            }// for
        }// omp taskgroup

        // make value type consistent in block matrix and sub blocks
        B->adjust_value_type();
    }// else

    M->set_id( bct->id() );
    M->set_procs( bct->procs() );
    
    return M;
}

template < typename basisapx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_uniform_rec ( const Hpro::TMatrix< value_t > &                            A,
                    const basisapx_t &                               basisapx,
                    const Hpro::TTruncAcc &                          acc,
                    cluster_basis< typename basisapx_t::value_t > &  rowcb,
                    cluster_basis< typename basisapx_t::value_t > &  colcb,
                    rec_basis_data_t &                               basis_data )
{
    using value_t = typename basisapx_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    if ( hlr::matrix::is_lowrank( A ) )
    {
        //
        // compute LRS representation W·T·X' = U·V' = M
        //

        auto  R  = cptrcast( &A, hlr::matrix::lrmatrix< value_t > );
        auto  W  = blas::copy( blas::mat_U< value_t >( R ) );
        auto  X  = blas::copy( blas::mat_V< value_t >( R ) );
        auto  Rw = blas::matrix< value_t >();
        auto  Rx = blas::matrix< value_t >();

        #pragma omp taskgroup
        {
            #pragma omp task default(shared)
            blas::qr( W, Rw );
                    
            #pragma omp task default(shared)
            blas::qr( X, Rx );
        }// omp taskgroup

        HLR_ASSERT( Rw.ncols() != 0 );
        HLR_ASSERT( Rx.ncols() != 0 );
                
        auto  T       = blas::prod( Rw, blas::adjoint( Rx ) );
        auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );

        #pragma omp taskgroup
        {
            #pragma omp task default(shared)
            {
                auto  Un = basis_data.compute_extended_row_basis( rowcb, W, T, acc, basisapx );
                        
                basis_data.update_row_coupling( rowcb, Un );
                rowcb.set_basis( std::move( Un ) );
            }// omp task
                
            #pragma omp task default(shared)
            {
                auto  Vn = basis_data.compute_extended_col_basis( colcb, X, T, acc, basisapx );
                        
                basis_data.update_col_coupling( colcb, Vn );
                colcb.set_basis( std::move( Vn ) );
            }// omp task
        }// omp taskgroup

        //
        // transform T into new bases
        //

        auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), W );
        auto  TV = blas::prod( blas::adjoint( colcb.basis() ), X );
        auto  TS = blas::prod( TU, T );
        auto  S  = blas::prod( TS, blas::adjoint( TV ) );

        M = std::make_unique< uniform_lrmatrix< value_t > >( A.row_is(), A.col_is(), rowcb, colcb, std::move( S ) );

        {
            auto  lock_map = std::scoped_lock( basis_data.rowmapmtx, basis_data.colmapmtx );

            basis_data.rowmap[ rowcb.is() ].push_back( M.get() );
            basis_data.colmap[ colcb.is() ].push_back( M.get() );
        }
    }// if
    else if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >();

        auto  BM = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

        BM->copy_struct_from( BA );

        #pragma omp taskloop collapse(2) default(shared) firstprivate(BA,BM)
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                auto  rowcb_i = rowcb.son( i );
                auto  colcb_j = colcb.son( j );
                
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    HLR_ASSERT( ! is_null_any( rowcb_i, colcb_j ) );
                        
                    auto  B_ij = build_uniform_rec( *BA->block( i, j ), basisapx, acc, *rowcb_i, *colcb_j, basis_data );
                    
                    BM->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// omp taskloop for

        // make value type consistent in block matrix and sub blocks
        BM->adjust_value_type();
    }// if
    else
    {
        M = A.copy();
    }// else

    M->set_id( A.id() );
    M->set_procs( A.procs() );
    
    return M;
}

template < typename value_t >
void
init_cluster_bases ( const Hpro::TBlockCluster *  bct,
                     cluster_basis< value_t > &   rowcb,
                     cluster_basis< value_t > &   colcb )
{
    //
    // decide upon cluster type, how to construct matrix
    //

    auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    if ( ! bct->is_leaf() )
    {
        //
        // build cluster bases for next level
        //
        
        {
            auto  lock = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
            
            for ( uint  i = 0; i < bct->nrows(); ++i )
            {
                auto  rowcb_i = rowcb.son( i );
            
                for ( uint  j = 0; j < bct->ncols(); ++j )
                {
                    auto  colcb_j = colcb.son( j );
                
                    if ( ! is_null( bct->son( i, j ) ) )
                    {
                        if ( is_null( rowcb_i ) )
                        {
                            rowcb_i = new cluster_basis< value_t >( bct->son( i, j )->is().row_is() );
                            rowcb_i->set_nsons( bct->son( i, j )->rowcl()->nsons() );
                            rowcb.set_son( i, rowcb_i );
                        }// if
                    
                        if ( is_null( colcb_j ) )
                        {
                            colcb_j = new cluster_basis< value_t >( bct->son( i, j )->is().col_is() );
                            colcb_j->set_nsons( bct->son( i, j )->colcl()->nsons() );
                            colcb.set_son( j, colcb_j );
                        }// if
                    }// if
                }// for
            }// for
        }

        //
        // recurse
        //
        
        for ( uint  i = 0; i < bct->nrows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );
            
            for ( uint  j = 0; j < bct->ncols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
                
                if ( ! is_null( bct->son( i, j ) ) )
                    init_cluster_bases( bct->son( i, j ), *rowcb_i, *colcb_j );
            }// for
        }// for
    }// if
}

template < typename value_t >
void
init_cluster_bases ( const Hpro::TMatrix< value_t > &       M,
                     cluster_basis< value_t > &  rowcb,
                     cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        {
            auto  lock = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
            
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                auto  rowcb_i = rowcb.son( i );
            
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    auto  colcb_j = colcb.son( j );
                    auto  M_ij    = B->block( i, j );
                
                    if ( ! is_null( M_ij ) )
                    {
                        if ( is_null( rowcb_i ) )
                        {
                            rowcb_i = new cluster_basis< value_t >( M_ij->row_is() );
                            rowcb.set_son( i, rowcb_i );
                        }// if
            
                        if ( is_blocked( M_ij ) && ( rowcb_i->nsons() == 0 ))
                            rowcb_i->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
                        
                        if ( is_null( colcb_j ) )
                        {
                            colcb_j = new cluster_basis< value_t >( M_ij->col_is() );
                            colcb.set_son( j, colcb_j );
                        }// if
            
                        if ( is_blocked( M_ij ) && ( colcb_j->nsons() == 0 ))
                            colcb_j->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
                    }// if
                }// for
            }// for
        }

        //
        // recurse
        //
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );
            
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
                
                if ( ! is_null( B->block( i, j ) ) )
                    init_cluster_bases( *B->block( i, j ), *rowcb_i, *colcb_j );
            }// for
        }// for
    }// if
}

#endif

}}}}// namespace hlr::omp::matrix::detail

#endif // __HLR_OMP_DETAIL_MATRIX_HH

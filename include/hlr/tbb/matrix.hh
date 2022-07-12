#ifndef __HLR_TBB_MATRIX_HH
#define __HLR_TBB_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <type_traits>
#include <deque>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/base/TTruncAcc.hh>

#include "hlr/seq/matrix.hh"
#include "hlr/matrix/restrict.hh"
#include "hlr/matrix/convert.hh"
#include "hlr/matrix/lrmatrix.hh"

#include "hlr/tbb/detail/matrix.hh"

namespace hlr { namespace tbb { namespace matrix {

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
std::unique_ptr< hpro::TMatrix< typename coeff_t::value_t > >
build ( const hpro::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const hpro::TTruncAcc &      acc,
        const size_t                 nseq = hpro::CFG::Arith::max_seq_size )
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );

    HLR_ASSERT( bct != nullptr );

    using  value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto        M     = std::unique_ptr< hpro::TMatrix< value_t > >();
    const auto  rowis = bct->is().row_is();
    const auto  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return seq::matrix::build( bct, coeff, lrapx, acc );
        
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            // auto  T = lrapx.build( bct, hpro::absolute_prec( acc.abs_eps() * std::sqrt( double(rowis.size() * colis.size()) ) ) );
            auto  T = lrapx.build( bct, acc( rowis, colis ) );

            // if ( is_lowrank( *T ) )
            // {
            //     auto  RT = ptrcast( T.get(), hpro::TRkMatrix< value_t > );
            //     auto  R  = std::make_unique< matrix::lrmatrix >( T->row_is(), T->col_is() );

            //     if ( T->is_complex() )
            //         R->set_lrmat( std::move( blas::mat_U< hpro::complex >( *RT ) ),
            //                       std::move( blas::mat_V< hpro::complex >( *RT ) ) );
            //     else
            //         R->set_lrmat( std::move( blas::mat_U< hpro::real >( *RT ) ),
            //                       std::move( blas::mat_V< hpro::real >( *RT ) ) );

            //     M = std::move( R );
            // }// if
            // else
            {
                M = std::move( T );
            }// else
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
        M = std::make_unique< hpro::TBlockMatrix< value_t > >( bct );
        
        auto  B = ptrcast( M.get(), hpro::TBlockMatrix< value_t > );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,bct,nseq] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( bct->son( i, j ) ) )
                        {
                            auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc, nseq );
                            
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
    }// else

    // copy properties from the cluster
    // M->set_cluster_force( bct );
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
std::unique_ptr< hpro::TMatrix< typename coeff_t::value_t > >
build_nearfield ( const hpro::TBlockCluster *  bct,
                  const coeff_t &              coeff,
                  const size_t                 nseq = hpro::CFG::Arith::max_seq_size )
{
    using  value_t = typename coeff_t::value_t;
    
    HLR_ASSERT( bct != nullptr );

    //
    // decide upon cluster type, how to construct matrix
    //

    auto        M     = std::unique_ptr< hpro::TMatrix< value_t > >();
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
        }// else
    }// if
    else if ( std::min( rowis.size(), colis.size() ) <= nseq )
    {
        M = hlr::seq::matrix::build_nearfield( bct, coeff, nseq );
    }// if
    else
    {
        M = std::make_unique< hpro::TBlockMatrix< value_t > >( bct );
        
        auto  B = ptrcast( M.get(), hpro::TBlockMatrix< value_t > );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,bct,nseq] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( bct->son( i, j ) ) )
                        {
                            auto  B_ij = build_nearfield( bct->son( i, j ), coeff, nseq );

                            if ( ! is_null( B_ij.get() ) )
                                 B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
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
std::unique_ptr< hpro::TMatrix< value_t > >
build ( const hpro::TBlockCluster *  bct,
        const hpro::TSparseMatrix< value_t > &  S,
        const hpro::TTruncAcc &      acc,
        const approx_t &             apx,
        const size_t                 nseq = hpro::CFG::Arith::max_seq_size ) // ignored
{
    // static_assert( std::is_same< typename coeff_t::value_t,
    //                              typename lrapx_t::value_t >::value,
    //                "coefficient function and low-rank approximation must have equal value type" );
    
    HLR_ASSERT( bct != nullptr );
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< hpro::TMatrix< value_t > >  M;
    
    if ( bct->is_leaf() )
    {
        //
        // restrict to local cluster and convert to desired format
        //

        auto  S_bct = hlr::matrix::restrict( S, bct->is() );
        
        if ( bct->is_adm() )
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
        M = std::make_unique< hpro::TBlockMatrix< value_t > >( bct );
        
        auto  B = ptrcast( M.get(), hpro::TBlockMatrix< value_t > );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,bct,nseq] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( bct->son( i, j ) ) )
                        {
                            auto  B_ij = build( bct->son( i, j ), S, acc, apx, nseq );
                            
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
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
            std::unique_ptr< hpro::TMatrix< typename coeff_t::value_t > > >
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
            std::unique_ptr< hpro::TMatrix< typename basisapx_t::value_t > > >
build_uniform_lvl ( const hpro::TMatrix< typename basisapx_t::value_t > &    A,
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
        rowcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_rows() );
        colcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_cols() );
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
            std::unique_ptr< hpro::TMatrix< typename coeff_t::value_t > > >
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
    
    auto  constr = detail::rec_uniform_construction( basisapx );
    auto  M      = constr.build( bct, coeff, lrapx, acc, *rowcb, *colcb );

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
            std::unique_ptr< hpro::TMatrix< typename basisapx_t::value_t > > >
build_uniform_rec ( const hpro::TMatrix< typename basisapx_t::value_t > &    A,
                    const basisapx_t &       basisapx,
                    const hpro::TTruncAcc &  acc,
                    const size_t             /* nseq */ = hpro::CFG::Arith::max_seq_size ) // ignored
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;

    auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
    auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

    if ( is_blocked( A ) )
    {
        rowcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_rows() );
        colcb->set_nsons( cptrcast( &A, hpro::TBlockMatrix< value_t > )->nblock_cols() );
    }// if

    detail::init_cluster_bases( A, *rowcb, *colcb );
    
    auto  constr = detail::rec_uniform_construction( basisapx );
    auto  M      = constr.build( A, acc, *rowcb, *colcb );
    
    return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}

//
// assign block cluster to matrix
//
template < typename value_t >
void
assign_cluster ( hpro::TMatrix< value_t > &   M,
                 const hpro::TBlockCluster &  bc )
{
    hlr::seq::matrix::assign_cluster( M, bc );
}

//
// return copy of matrix
// - copy operation is performed in parallel for sub blocks
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix< value_t > >
copy ( const hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix< value_t > );

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
                        if ( ! is_null( BM->block( i, j ) ) )
                        {
                            auto  B_ij = hlr::tbb::matrix::copy( * BM->block( i, j ) );
                            
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
// return truncated copy of matrix
// - copy operation is performed in parallel for sub blocks
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix< value_t > >
copy ( const hpro::TMatrix< value_t > &  M,
       const hpro::TTruncAcc &           acc )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [B,BM,&acc] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( BM->block( i, j ) ) )
                        {
                            auto  B_ij = copy( * BM->block( i, j ), acc );
                            
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
        auto  Mc = M.copy();

        Mc->truncate( acc( M.row_is(), M.col_is() ) );

        return Mc;
    }// else
}

//
// return copy of matrix
// - copy operation is performed in parallel for sub blocks
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix< value_t > >
copy_nearfield ( const hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix< value_t > );

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
            } );

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
// - copy operation is performed in parallel for sub blocks
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix< value_t > >
copy_tiled ( const hpro::TMatrix< value_t > &  M,
             const size_t                      ntile )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [B,BM,ntile] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( BM->block( i, j ) ) )
                        {
                            auto  B_ij = copy_tiled< value_t >( * BM->block( i, j ), ntile );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
        
        return N;
    }// if
    else if ( is_lowrank( M ) )
    {
        //
        // copy low-rank data into tiled form
        //

        auto  RM = cptrcast( & M, hpro::TRkMatrix< value_t > );
        auto  R  = std::make_unique< hlr::matrix::tiled_lrmatrix< value_t > >( RM->row_is(),
                                                                               RM->col_is(),
                                                                               ntile,
                                                                               hpro::blas_mat_A< value_t >( RM ),
                                                                               hpro::blas_mat_B< value_t >( RM ) );

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
// - copy operation is performed in parallel for sub blocks
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix< value_t > >
copy_nontiled ( const hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix< value_t > );

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
                        if ( ! is_null( BM->block( i, j ) ) )
                        {
                            auto  B_ij = copy_nontiled< value_t >( * BM->block( i, j ) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
        
        return N;
    }// if
    else if ( IS_TYPE( & M, tiled_lrmatrix ) )
    {
        //
        // copy low-rank data into tiled form
        //

        auto  RM = cptrcast( & M, hlr::matrix::tiled_lrmatrix< value_t > );
        auto  R  = std::make_unique< hpro::TRkMatrix< value_t > >( RM->row_is(), RM->col_is() );
        auto  U  = hlr::matrix::to_dense( RM->U() );
        auto  V  = hlr::matrix::to_dense( RM->V() );

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
// return copy of (block-wise) lower-left part of matrix
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix< value_t > >
copy_ll ( const hpro::TMatrix< value_t > &  M,
          const hpro::diag_type_t           diag = hpro::general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        ::tbb::parallel_for< uint >(
            0, B->nblock_rows(),
            [B,BM,diag] ( const uint  i )
            {
                ::tbb::parallel_for< uint >(
                    0, i+1,
                    [B,BM,diag,i] ( const uint  j )
                    {
                        if ( ! is_null( BM->block( i, j ) ) )
                        {
                            auto  B_ij = ( i == j ? copy_ll( * BM->block( i, j ), diag ) : hlr::tbb::matrix::copy( * BM->block( i, j ) ) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    } );
            } );
        
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

                auto  D = ptrcast( T.get(), hpro::TDenseMatrix< value_t > );

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
std::unique_ptr< hpro::TMatrix< value_t > >
copy_ur ( const hpro::TMatrix< value_t > &    M,
          const hpro::diag_type_t  diag = hpro::general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        ::tbb::parallel_for< uint >(
            0, B->nblock_cols(),
            [B,BM,diag] ( const uint  j )
            {
                ::tbb::parallel_for< uint >(
                    0, j+1,
                    [B,BM,diag,j] ( const uint  i )
                    {
                        if ( ! is_null( BM->block( i, j ) ) )
                        {
                            auto  B_ij = ( i == j ? copy_ur( * BM->block( i, j ), diag ) : hlr::tbb::matrix::copy( * BM->block( i, j ) ) );
                    
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    } );
            } );
        
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

                auto  D = ptrcast( T.get(), hpro::TDenseMatrix< value_t > );

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
copy_to ( const hpro::TMatrix< value_t > &  A,
          hpro::TMatrix< value_t > &        B )
{
    HLR_ASSERT( A.type()     == B.type() );
    HLR_ASSERT( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix< value_t > );
        auto  BB = ptrcast(  &B, hpro::TBlockMatrix< value_t > );

        HLR_ASSERT( BA->nblock_rows() == BB->nblock_rows() );
        HLR_ASSERT( BA->nblock_cols() == BB->nblock_cols() );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, BA->nblock_rows(),
                                            0, BA->nblock_cols() ),
            [BA,BB] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( BA->block( i, j ) ) )
                        {
                            HLR_ASSERT( ! is_null( BB->block( i, j ) ) );
                            
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
template < typename value_t >
std::unique_ptr< hpro::TMatrix< value_t > >
realloc ( hpro::TMatrix< value_t > *  A )
{
    if ( is_null( A ) )
        return nullptr;
    
    if ( is_blocked( A ) )
    {
        auto  B  = ptrcast( A, hpro::TBlockMatrix< value_t > );
        auto  C  = std::make_unique< hpro::TBlockMatrix< value_t > >();
        auto  BC = ptrcast( C.get(), hpro::TBlockMatrix< value_t > );

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

//
// return copy of matrix with uniform low-rank matrices
// - TODO: add cluster basis as template argument to allow
//         different bases
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix< value_t > >
copy_uniform ( const hpro::TMatrix< value_t > &         M,
               hlr::matrix::cluster_basis< value_t > &  rowcb,
               hlr::matrix::cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );

        HLR_ASSERT( B->nblock_rows() == rowcb.nsons() );
        HLR_ASSERT( B->nblock_cols() == colcb.nsons() );
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,BM] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( BM->block( i, j ) ) )
                        {
                            auto  B_ij = copy_uniform( * BM->block( i, j ), * rowcb.son(i), * colcb.son(j) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
        
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

        auto  R  = cptrcast( &M, hpro::TRkMatrix< value_t > );

        auto  UA = rowcb.transform_forward( hpro::blas_mat_A< value_t >( R ) );
        auto  VB = colcb.transform_forward( hpro::blas_mat_B< value_t >( R ) );
        auto  S  = blas::prod( value_t(1), UA, blas::adjoint( VB ) );

        // auto  M1 = blas::prod( value_t(1), hpro::blas_mat_A< value_t >( R ), blas::adjoint( hpro::blas_mat_B< value_t >( R ) ) );
        // auto  T  = blas::prod( value_t(1), rowcb.basis(), S );
        // auto  M2 = blas::prod( value_t(1), T, blas::adjoint( colcb.basis() ) );

        // blas::add( value_t(-1), M2, M1 );
        
        // std::cout << blas::norm_F( M1 ) << std::endl;
        
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

//
// return copy of matrix with uniform low-rank matrices converted
// to standard lowrank matrices
//
// inline
// std::unique_ptr< hpro::TMatrix< value_t > >
// copy_nongeneric ( const hpro::TMatrix< value_t > &  M )
// {
//     if ( is_blocked( M ) )
//     {
//         auto  BM = cptrcast( &M, hpro::TBlockMatrix< value_t > );
//         auto  N  = std::make_unique< hpro::TBlockMatrix< value_t > >();
//         auto  B  = ptrcast( N.get(), hpro::TBlockMatrix< value_t > );

//         B->copy_struct_from( BM );

//         ::tbb::parallel_for(
//             ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
//                                             0, B->nblock_cols() ),
//             [&,B,BM] ( const ::tbb::blocked_range2d< uint > &  r )
//             {
//                 for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
//                 {
//                     for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
//                     {
//                         if ( ! is_null( BM->block( i, j ) ) )
//                         {
//                             auto  B_ij = copy_nongeneric( * BM->block( i, j ) );
                            
//                             B_ij->set_parent( B );
//                             B->set_block( i, j, B_ij.release() );
//                         }// if
//                     }// for
//                 }// for
//             } );
        
//         return N;
//     }// if
//     else if ( is_generic_lowrank( M ) )
//     {
//         auto  R = cptrcast( &M, lrmatrix );

//         if ( R->is_compressed() )
//         {
//             HLR_ERROR( "TODO" );
//         }// if

//         auto  SR = std::unique_ptr< hpro::TRkMatrix< value_t > >();
        
//         if ( R->value_type() == blas::value_type::rfp32 )
//         {
//             auto  U = blas::copy< hpro::real, float >( R->U< float >() );
//             auto  V = blas::copy< hpro::real, float >( R->V< float >() );

//             SR = std::make_unique< hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
//         }// if
//         else if ( R->value_type() == blas::value_type::rfp64 )
//         {
//             auto  U = blas::copy< hpro::real, double >( R->U< double >() );
//             auto  V = blas::copy< hpro::real, double >( R->V< double >() );

//             SR = std::make_unique< hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
//         }// if
//         else if ( R->value_type() == blas::value_type::cfp32 )
//         {
//             auto  U = blas::copy< hpro::complex, std::complex< float > >( R->U< std::complex< float > >() );
//             auto  V = blas::copy< hpro::complex, std::complex< float > >( R->V< std::complex< float > >() );

//             SR = std::make_unique< hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
//         }// if
//         else if ( R->value_type() == blas::value_type::cfp64 )
//         {
//             auto  U = blas::copy< hpro::complex, std::complex< double > >( R->U< std::complex< double > >() );
//             auto  V = blas::copy< hpro::complex, std::complex< double > >( R->V< std::complex< double > >() );

//             SR = std::make_unique< hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
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

//         auto  SD = std::unique_ptr< hpro::TDenseMatrix< value_t > >();

//         if ( D->value_type() == blas::value_type::rfp32 )
//         {
//             SD = std::make_unique< hpro::TDenseMatrix< value_t > >( M.row_is(), M.col_is(), std::move( blas::copy< hpro::real, float >( D->M< float >() ) ) );
//         }// if
//         else if ( D->value_type() == blas::value_type::rfp64 )
//         {
//             SD = std::make_unique< hpro::TDenseMatrix< value_t > >( M.row_is(), M.col_is(), std::move( blas::copy< hpro::real, double >( D->M< double >() ) ) );
//         }// if
//         else if ( D->value_type() == blas::value_type::cfp32 )
//         {
//             SD = std::make_unique< hpro::TDenseMatrix< value_t > >( M.row_is(), M.col_is(), std::move( blas::copy< hpro::complex, std::complex< float > >( D->M< std::complex< float > >() ) ) );
//         }// if
//         else if ( D->value_type() == blas::value_type::cfp64 )
//         {
//             SD = std::make_unique< hpro::TDenseMatrix< value_t > >( M.row_is(), M.col_is(), std::move( blas::copy< hpro::complex, std::complex< double > >( D->M< std::complex< double > >() ) ) );
//         }// if

//         SD->set_id( D->id() );

//         return SD;
//     }// if
//     else
//     {
//         return M.copy();
//     }// else
// }

}}}// namespace hlr::tbb::matrix

#endif // __HLR_TBB_MATRIX_HH

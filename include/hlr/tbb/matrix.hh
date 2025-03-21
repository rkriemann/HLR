#ifndef __HLR_TBB_MATRIX_HH
#define __HLR_TBB_MATRIX_HH
//
// Project     : HLR
// Module      : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <type_traits>
#include <deque>
#include <atomic>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/task_group.h>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/base/TTruncAcc.hh>

#include <hlr/seq/matrix.hh>
#include <hlr/matrix/restrict.hh>
#include <hlr/matrix/convert.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/lrsmatrix.hh>
#include <hlr/matrix/lrsvmatrix.hh>
#include <hlr/matrix/sparse_matrix.hh>
#include <hlr/tbb/compress.hh>

#include <hlr/tbb/detail/uniform_matrix.hh>
#include <hlr/tbb/detail/h2_matrix.hh>

#include <hlr/utils/timer.hh> // DEBUG

namespace hlr { namespace tbb { namespace matrix {

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
        const Hpro::TTruncAcc &      acc,
        const bool                   compress = false,
        const size_t                 nseq     = Hpro::CFG::Arith::max_seq_size )
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
        return seq::matrix::build( bct, coeff, lrapx, acc, compress );
        
    if ( bct->is_leaf() )
    {
        M = hlr::seq::matrix::build( bct, coeff, lrapx, acc, compress, nseq );
    }// if
    else if ( std::min( rowis.size(), colis.size() ) <= nseq )
    {
        M = hlr::seq::matrix::build( bct, coeff, lrapx, acc, compress, nseq );
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
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,bct,compress,nseq] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( bct->son( i, j ) ) )
                        {
                            auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc, compress, nseq );
                            
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
// same as above but use lrsvmatrix for lowrank matrices
//
template < coefficient_function_type  coeff_t,
           lowrank_approx_type        lrapx_t >
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build_sv ( const Hpro::TBlockCluster *  bct,
           const coeff_t &              coeff,
           const lrapx_t &              lrapx,
           const Hpro::TTruncAcc &      acc,
           const bool                   compress = false,
           const size_t                 nseq     = Hpro::CFG::Arith::max_seq_size )
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

    if ( bct->is_leaf() )
    {
        M = hlr::seq::matrix::build_sv( bct, coeff, lrapx, acc, compress, nseq );
    }// if
    else if ( std::min( rowis.size(), colis.size() ) <= nseq )
    {
        M = hlr::seq::matrix::build_sv( bct, coeff, lrapx, acc, compress, nseq );
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
                            auto  B_ij = build_sv( bct->son( i, j ), coeff, lrapx, acc, compress, nseq );
                            
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
    }// else

    // copy properties from the cluster
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

template < coefficient_function_type  coeff_t,
           lowrank_approx_type        lrapx_t >
size_t
mem_sv ( const Hpro::TBlockCluster *  bct,
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

    size_t      mem   = 0;
    const auto  rowis = bct->is().row_is();
    const auto  colis = bct->is().col_is();

    if ( bct->is_leaf() )
    {
        mem += hlr::seq::matrix::mem_sv( bct, coeff, lrapx, acc, nseq );
    }// if
    else if ( std::min( rowis.size(), colis.size() ) <= nseq )
    {
        mem += hlr::seq::matrix::mem_sv( bct, coeff, lrapx, acc, nseq );
    }// if
    else
    {
        //
        // recurse
        //

        auto  mtx = std::mutex();
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, bct->nrows(),
                                            0, bct->ncols() ),
            [&,bct,nseq] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( bct->son( i, j ) ) )
                        {
                            const auto  mem_ij = mem_sv( bct->son( i, j ), coeff, lrapx, acc, nseq );
                            
                            {
                                auto  lock = std::scoped_lock( mtx );

                                mem += mem_ij;
                            }
                        }// if
                    }// for
                }// for
            } );
    }// else

    return mem;
}

//
// build representation of nearfield of dense matrix with
// matrix structure defined by <bct>, matrix coefficients
// defined by <coeff>
//
template < coefficient_function_type coeff_t >
std::unique_ptr< Hpro::TMatrix< typename coeff_t::value_t > >
build_nearfield ( const Hpro::TBlockCluster *  bct,
                  const coeff_t &              coeff,
                  const size_t                 nseq = Hpro::CFG::Arith::max_seq_size )
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
        M = hlr::seq::matrix::build_nearfield( bct, coeff, nseq );
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
           approx::approximation_type approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build ( const Hpro::TBlockCluster &             bct,
        const Hpro::TSparseMatrix< value_t > &  S,
        const Hpro::TTruncAcc &                 acc,
        const approx_t &                        apx,
        const size_t                            nseq = Hpro::CFG::Arith::max_seq_size ) // ignored
{
    // std::cout << "build    : " << bct.to_string() << std::endl;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< Hpro::TMatrix< value_t > >  M;
    
    if ( bct.is_leaf() )
    {
        M = hlr::seq::matrix::build( bct, S, acc, apx, nseq );
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
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,nseq] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( bct.son( i, j ) ) )
                        {
                            auto  B_ij = build( *bct.son( i, j ), S, acc, apx, nseq );
                            
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
    }// else

    // M->set_cluster_force( bct );
    M->set_id( bct.id() );
    M->set_procs( bct.procs() );

    return M;
}
    
template < typename value_t,
           approx::approximation_type approx_t >
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

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,nbr,nbc,nseq] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
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
            } );
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
           approx::approximation_type approx_t >
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
        M = hlr::seq::matrix::build_sparse( bct, S, acc, apx, nseq );
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
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B,nseq] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( bct.son( i, j ) ) )
                        {
                            auto  B_ij = build_sparse( *bct.son( i, j ), S, acc, apx, nseq );
                            
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
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
    auto  mutex_H    = std::vector< std::mutex >( bc->id() + 1 );
    auto  mutex_U    = std::vector< std::mutex >( bc->id() + 1 );
    auto  mutex_coup = std::vector< std::mutex >( bc->id() + 1 );

    ::tbb::parallel_invoke(
        [&,bc] { detail::collect_clusters( bc->rowcl(), row_cls ); },
        [&,bc] { detail::collect_clusters( bc->colcl(), col_cls ); },
        [&,bc] { detail::build_block_map< value_t >( bc, row_map, col_map, mutex_H ); } // using mutex array here assuming id(cl) < id(bc)
    );

    //
    // intermix row/column clusters to free lowrank blocks as soon as possible
    //

    ::tbb::parallel_invoke(
        [&,compress] ()
        {
            ::tbb::task_group  tg;

            for ( auto  rowcl : row_cls )
            {
                tg.run( [&,rowcl,compress]
                { 
                    auto  rowcb = std::make_unique< hlr::matrix::shared_cluster_basis< value_t > >( *rowcl );
            
                    rowcb->set_nsons( rowcl->nsons() );
                    rowcb->set_id( rowcl->id() );
                    row_cbs[ rowcb->id() ] = rowcb.get();

                    detail::build_uniform( rowcl, rowcb.release(),
                                           coeff, lrapx, basisapx, acc, compress,
                                           row_map, mat_map_H, mat_map_U, row_coup, col_coup,
                                           apply_normal,
                                           mutex_H, mutex_U, mutex_coup );
                } );
            }// for

            tg.wait();
        },

        [&,compress] ()
        {
            ::tbb::task_group  tg;

            for ( auto  colcl : col_cls )
            {
                tg.run( [&,colcl,compress]
                { 
                    auto  colcb = std::make_unique< hlr::matrix::shared_cluster_basis< value_t > >( *colcl );
            
                    colcb->set_nsons( colcl->nsons() );
                    colcb->set_id( colcl->id() );
                    col_cbs[ colcb->id() ] = colcb.get();
            
                    detail::build_uniform( colcl, colcb.release(),
                                           coeff, lrapx, basisapx, acc, compress,
                                           col_map, mat_map_H, mat_map_U, col_coup, row_coup,
                                           apply_adjoint,
                                           mutex_H, mutex_U, mutex_coup );
                } );
            }// for

            tg.wait();
        }
    );
    
    // check if all low rank blocks are gone
    for ( auto  M : mat_map_H )
    {
        HLR_ASSERT( ! hlr::matrix::is_lowrank( M ) );
    }// for
    
    auto  rowcb_root = std::unique_ptr< hlr::matrix::shared_cluster_basis< value_t > >( row_cbs[ bc->rowcl()->id() ] );
    auto  colcb_root = std::unique_ptr< hlr::matrix::shared_cluster_basis< value_t > >( col_cbs[ bc->colcl()->id() ] );
    auto  M_root     = std::unique_ptr< Hpro::TMatrix< value_t > >( mat_map_U[ bc->id() ] );

    HLR_ASSERT( ! is_null( M_root ) && ! is_null( rowcb_root ) && ! is_null( colcb_root ) );

    ::tbb::parallel_invoke(
        [&,bc] () { detail::fix_hierarchy( bc->rowcl(), rowcb_root.get(), row_cbs ); },
        [&,bc] () { detail::fix_hierarchy( bc->colcl(), colcb_root.get(), col_cbs ); },
        [&,bc] () { detail::fix_hierarchy( bc, M_root.get(), mat_map_U ); }
    );
    
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
    auto  mutex_H    = std::vector< std::mutex >( bc->id() + 1 );
    auto  mutex_U    = std::vector< std::mutex >( bc->id() + 1 );

    ::tbb::parallel_invoke(
        [&,bc] { detail::collect_clusters( bc->rowcl(), row_cls ); },
        [&,bc] { detail::collect_clusters( bc->colcl(), col_cls ); },
        [&,bc] { detail::build_block_map< value_t >( bc, row_map, col_map, mutex_H ); } // using mutex array here assuming id(cl) < id(bc)
    );
    
    //
    // intermix row/column clusters to free lowrank blocks as soon as possible
    //

    ::tbb::parallel_invoke(
        [&,compress] ()
        {
            ::tbb::task_group  tg;

            for ( auto  rowcl : row_cls )
            {
                tg.run( [&,rowcl,compress]
                { 
                    auto  rowcb = std::make_unique< hlr::matrix::shared_cluster_basis< value_t > >( *rowcl );
            
                    rowcb->set_nsons( rowcl->nsons() );
                    rowcb->set_id( rowcl->id() );
                    row_cbs[ rowcb->id() ] = rowcb.get();

                    detail::build_uniform_sep( rowcl, rowcb.release(),
                                               coeff, lrapx, basisapx, acc, compress,
                                               row_map, mat_map_H, mat_map_U,
                                               apply_normal,
                                               mutex_H, mutex_U );
                } );
            }// for

            tg.wait();
        },

        [&,compress] ()
        {
            ::tbb::task_group  tg;

            for ( auto  colcl : col_cls )
            {
                tg.run( [&,colcl,compress]
                { 
                    auto  colcb = std::make_unique< hlr::matrix::shared_cluster_basis< value_t > >( *colcl );
            
                    colcb->set_nsons( colcl->nsons() );
                    colcb->set_id( colcl->id() );
                    col_cbs[ colcb->id() ] = colcb.get();
            
                    detail::build_uniform_sep( colcl, colcb.release(),
                                               coeff, lrapx, basisapx, acc, compress,
                                               col_map, mat_map_H, mat_map_U,
                                               apply_adjoint,
                                               mutex_H, mutex_U );
                } );
            }// for

            tg.wait();
        }
    );
    
    // check if all low rank blocks are gone
    for ( auto  M : mat_map_H )
    {
        HLR_ASSERT( ! hlr::matrix::is_lowrank( M ) );
    }// for
    
    auto  rowcb_root = std::unique_ptr< hlr::matrix::shared_cluster_basis< value_t > >( row_cbs[ bc->rowcl()->id() ] );
    auto  colcb_root = std::unique_ptr< hlr::matrix::shared_cluster_basis< value_t > >( col_cbs[ bc->colcl()->id() ] );
    auto  M_root     = std::unique_ptr< Hpro::TMatrix< value_t > >( mat_map_U[ bc->id() ] );

    HLR_ASSERT( ! is_null( M_root ) && ! is_null( rowcb_root ) && ! is_null( colcb_root ) );

    ::tbb::parallel_invoke(
        [&,bc] () { detail::fix_hierarchy( bc->rowcl(), rowcb_root.get(), row_cbs ); },
        [&,bc] () { detail::fix_hierarchy( bc->colcl(), colcb_root.get(), col_cbs ); },
        [&,bc] () { detail::fix_hierarchy( bc, M_root.get(), mat_map_U ); }
    );
    
    return { std::move( rowcb_root ), std::move( colcb_root ), std::move( M_root ) };
}

// template < coefficient_function_type  coeff_t,
//            lowrank_approx_type        lrapx_t,
//            approx::approximation_type basisapx_t >
// std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
//             std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
//             std::unique_ptr< Hpro::TMatrix< Hpro::value_type_t< coeff_t > > > >
// build_uniform_lvl ( const Hpro::TBlockCluster *  bct,
//                     const coeff_t &              coeff,
//                     const lrapx_t &              lrapx,
//                     const basisapx_t &           basisapx,
//                     const Hpro::TTruncAcc &      acc,
//                     const bool                   compress = false,
//                     const size_t                 /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
// {
//     auto  [ rowcb, colcb, A ] = detail::build_uniform_lvl( bct, coeff, lrapx, basisapx, acc, compress );

//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *rowcb, id ); }
//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *colcb, id ); }

//     return { std::move( rowcb ), std::move( colcb ), std::move( A ) };
// }

// template < coefficient_function_type  coeff_t,
//            lowrank_approx_type        lrapx_t,
//            approx::approximation_type basisapx_t >
// std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
//             std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
//             std::unique_ptr< Hpro::TMatrix< Hpro::value_type_t< coeff_t > > > >
// build_uniform_lvl_sep ( const Hpro::TBlockCluster *  bct,
//                         const coeff_t &              coeff,
//                         const lrapx_t &              lrapx,
//                         const basisapx_t &           basisapx,
//                         const Hpro::TTruncAcc &      acc,
//                         const bool                   compress = false,
//                         const size_t                 /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
// {
//     auto  [ rowcb, colcb, A ] = detail::build_uniform_lvl_sep( bct, coeff, lrapx, basisapx, acc, compress );

//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *rowcb, id ); }
//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *colcb, id ); }

//     return { std::move( rowcb ), std::move( colcb ), std::move( A ) };
// }

// template < approx::approximation_type basisapx_t >
// std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
//             std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
//             std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
// build_uniform_lvl ( const Hpro::TMatrix< typename basisapx_t::value_t > & A,
//                     const basisapx_t &                                    basisapx,
//                     const Hpro::TTruncAcc &                               acc,
//                     const size_t                                          /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
// {
//     using value_t       = typename basisapx_t::value_t;
//     using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

//     auto  rowcb = std::make_unique< cluster_basis >( A.row_is() );
//     auto  colcb = std::make_unique< cluster_basis >( A.col_is() );

//     if ( is_blocked( A ) )
//     {
//         rowcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
//         colcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
//     }// if

//     detail::init_cluster_bases( A, *rowcb, *colcb );

//     auto  M = detail::build_uniform_lvl( A, basisapx, acc, *rowcb, *colcb );

//     return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
// }

//
// build representation of dense matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and low-rank blocks computed by <lrapx>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//

// template < coefficient_function_type coeff_t,
//            lowrank_approx_type lrapx_t,
//            approx::approximation_type basisapx_t >
// std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
//             std::unique_ptr< hlr::matrix::shared_cluster_basis< Hpro::value_type_t< coeff_t > > >,
//             std::unique_ptr< Hpro::TMatrix< Hpro::value_type_t< coeff_t > > > >
// build_uniform_rec ( const Hpro::TBlockCluster *  bct,
//                     const coeff_t &              coeff,
//                     const lrapx_t &              lrapx,
//                     const basisapx_t &           basisapx,
//                     const Hpro::TTruncAcc &      acc,
//                     const size_t                 /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
// {
//     static_assert( std::is_same_v< Hpro::value_type_t< coeff_t >, Hpro::value_type_t< lrapx_t > >,
//                    "coefficient function and low-rank approximation must have equal value type" );
//     static_assert( std::is_same_v< Hpro::value_type_t< coeff_t >, Hpro::value_type_t< basisapx_t > >,
//                    "coefficient function and basis approximation must have equal value type" );
    
//     HLR_ASSERT( bct != nullptr );

//     using value_t       = Hpro::value_type_t< coeff_t >;
//     using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

//     auto  rowcb  = std::make_unique< cluster_basis >( bct->is().row_is() );
//     auto  colcb  = std::make_unique< cluster_basis >( bct->is().col_is() );

//     rowcb->set_nsons( bct->rowcl()->nsons() );
//     colcb->set_nsons( bct->colcl()->nsons() );

//     detail::init_cluster_bases( bct, *rowcb, *colcb );
    
//     auto  constr = detail::rec_uniform_construction( basisapx );
//     auto  M      = constr.build( bct, coeff, lrapx, acc, *rowcb, *colcb );

//     return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
// }

// //
// // build uniform-H version from given H-matrix <A>
// // - low-rank blocks are converted to uniform low-rank matrices and
// //   shared bases are constructed on-the-fly
// //
// template < approx::approximation_type basisapx_t >
// std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
//             std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
//             std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
// build_uniform_rec ( const Hpro::TMatrix< typename basisapx_t::value_t > &    A,
//                     const basisapx_t &                                       basisapx,
//                     const Hpro::TTruncAcc &                                  acc,
//                     const bool                                               compress,
//                     const size_t                                             /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
// {
//     using value_t       = typename basisapx_t::value_t;
//     using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

//     //
//     // mapping of index sets to lowrank matrices 
//     //

//     auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
//     auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

//     if ( is_blocked( A ) )
//     {
//         rowcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
//         colcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
//     }// if

//     detail::init_cluster_bases( A, *rowcb, *colcb );
    
//     auto  lr_row_map   = detail::lr_coupling_map_t< value_t >();
//     auto  lr_col_map   = detail::lr_coupling_map_t< value_t >();
//     auto  lrsv_row_map = detail::lrsv_mat_map_t< value_t >();
//     auto  lrsv_col_map = detail::lrsv_mat_map_t< value_t >();

//     detail::build_mat_map( A, *rowcb, *colcb, lr_row_map, lr_col_map, lrsv_row_map, lrsv_col_map );

//     //
//     // build cluster bases
//     //

//     if ( lrsv_row_map.empty() )
//     {
//         ::tbb::parallel_invoke(
//             [&] () { detail::build_cluster_basis( *rowcb, basisapx, acc, lr_row_map, false, compress ); },
//             [&] () { detail::build_cluster_basis( *colcb, basisapx, acc, lr_col_map, true,  compress );  }
//         );
//     }// if
//     else
//     {
//         ::tbb::parallel_invoke(
//             [&] () { detail::build_cluster_basis( *rowcb, basisapx, acc, lrsv_row_map, false, compress ); },
//             [&] () { detail::build_cluster_basis( *colcb, basisapx, acc, lrsv_col_map, true,  compress );  }
//         );
//     }// else

//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *rowcb, id ); }
//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *colcb, id ); }

//     //
//     // construct uniform lowrank matrices with given cluster bases
//     //
    
//     auto  M = detail::build_uniform( A, *rowcb, *colcb, acc, compress );

//     return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
// }

// //
// // build uniform-H version from given H-matrix <A>
// // - low-rank blocks are converted to uniform low-rank matrices and
// //   shared bases are constructed on-the-fly
// //
// template < approx::approximation_type basisapx_t >
// std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
//             std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
//             std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
// build_uniform_rec_sep ( const Hpro::TMatrix< typename basisapx_t::value_t > &    A,
//                         const basisapx_t &                                       basisapx,
//                         const Hpro::TTruncAcc &                                  acc,
//                         const bool                                               compress,
//                         const size_t                                             /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
// {
//     using value_t       = typename basisapx_t::value_t;
//     using cluster_basis = hlr::matrix::shared_cluster_basis< value_t >;

//     //
//     // mapping of index sets to lowrank matrices 
//     //

//     auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
//     auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

//     if ( is_blocked( A ) )
//     {
//         rowcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
//         colcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
//     }// if

//     detail::init_cluster_bases( A, *rowcb, *colcb );
    
//     auto  lr_row_map   = detail::lr_coupling_map_t< value_t >();
//     auto  lr_col_map   = detail::lr_coupling_map_t< value_t >();
//     auto  lrsv_row_map = detail::lrsv_mat_map_t< value_t >();
//     auto  lrsv_col_map = detail::lrsv_mat_map_t< value_t >();

//     detail::build_mat_map( A, *rowcb, *colcb, lr_row_map, lr_col_map, lrsv_row_map, lrsv_col_map );

//     //
//     // build cluster bases
//     //

//     if ( lrsv_row_map.empty() )
//     {
//         ::tbb::parallel_invoke(
//             [&] () { detail::build_cluster_basis( *rowcb, basisapx, acc, lr_row_map, false, compress ); },
//             [&] () { detail::build_cluster_basis( *colcb, basisapx, acc, lr_col_map, true,  compress );  }
//         );
//     }// if
//     else
//     {
//         ::tbb::parallel_invoke(
//             [&] () { detail::build_cluster_basis( *rowcb, basisapx, acc, lrsv_row_map, false, compress ); },
//             [&] () { detail::build_cluster_basis( *colcb, basisapx, acc, lrsv_col_map, true,  compress );  }
//         );
//     }// else

//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *rowcb, id ); }
//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *colcb, id ); }

//     //
//     // construct uniform lowrank matrices with given cluster bases
//     //
    
//     auto  M = detail::build_uniform_sep( A, *rowcb, *colcb, acc, compress );

//     return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
// }

namespace tlr
{
//
// special version for BLR format
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

template < approx::approximation_type basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::shared_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_uniform ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
                const basisapx_t &                                     basisapx,
                const Hpro::TTruncAcc &                                acc ) // ignored
{
    return detail::build_blr2( A, basisapx, acc );
}

}// namespace tlr

template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_h2 ( const Hpro::TMatrix< typename basisapx_t::value_t > &                      A,
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

    auto  pblocks = std::list< const hlr::matrix::uniform_lrmatrix< value_t > * >();
    auto  row_map = std::vector< std::list< const uniform_lrmatrix< value_t > * > >( srowcb.id() + 1 );
    auto  col_map = std::vector< std::list< const uniform_lrmatrix< value_t > * > >( scolcb.id() + 1 );

    detail::build_uniform_map( A, srowcb, scolcb, row_map, col_map );

    auto  nrowcb = std::unique_ptr< hlr::matrix::nested_cluster_basis< value_t > >();
    auto  ncolcb = std::unique_ptr< hlr::matrix::nested_cluster_basis< value_t > >();
    auto  Rr     = blas::matrix< value_t >();
    auto  Rc     = blas::matrix< value_t >();

    ::tbb::parallel_invoke(
        [&] () { std::tie( nrowcb, Rr ) = detail::build_nested_cluster_basis( srowcb, row_map, pblocks, basisapx, acc, false ); },
        [&] () { std::tie( ncolcb, Rc ) = detail::build_nested_cluster_basis( scolcb, col_map, pblocks, basisapx, acc, true  ); }
    );

    //
    // precompute transformation from shared to nested cluster basis
    //

    auto  row_trans = std::vector< blas::matrix< value_t > >( srowcb.id() + 1 );
    auto  col_trans = std::vector< blas::matrix< value_t > >( scolcb.id() + 1 );

    ::tbb::parallel_invoke(
        [&] () { detail::compute_transform( srowcb, *nrowcb, row_trans ); },
        [&] () { detail::compute_transform( scolcb, *ncolcb, col_trans ); }
    );
    
    //
    // construct uniform lowrank matrices with given cluster bases
    //
    
    auto  M = detail::build_h2( A, *nrowcb, *ncolcb, row_trans, col_trans, acc, compress );

    // only after building matrix
    if ( compress )
    {
        ::tbb::parallel_invoke(
            [&] () { matrix::compress< matrix::nested_cluster_basis< value_t > >( *nrowcb, acc ); },
            [&] () { matrix::compress< matrix::nested_cluster_basis< value_t > >( *ncolcb, acc ); }
        );
    }// if
    
    return  { std::move( nrowcb ), std::move( ncolcb ), std::move( M ) };
}
    
template < typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
            std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
build_h2_sep ( const Hpro::TMatrix< typename basisapx_t::value_t > &                      A,
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

    auto  pblocks = std::list< const hlr::matrix::uniform_lr2matrix< value_t > * >();
    auto  row_map = std::vector< std::list< const uniform_lr2matrix< value_t > * > >( srowcb.id() + 1 );
    auto  col_map = std::vector< std::list< const uniform_lr2matrix< value_t > * > >( scolcb.id() + 1 );

    detail::build_uniform_sep_map( A, srowcb, scolcb, row_map, col_map );

    auto  nrowcb = std::unique_ptr< hlr::matrix::nested_cluster_basis< value_t > >();
    auto  ncolcb = std::unique_ptr< hlr::matrix::nested_cluster_basis< value_t > >();
    auto  Rr     = blas::matrix< value_t >();
    auto  Rc     = blas::matrix< value_t >();

    ::tbb::parallel_invoke(
        [&] () { std::tie( nrowcb, Rr ) = detail::build_nested_cluster_basis_sep( srowcb, row_map, pblocks, basisapx, acc, false ); },
        [&] () { std::tie( ncolcb, Rc ) = detail::build_nested_cluster_basis_sep( scolcb, col_map, pblocks, basisapx, acc, true  ); }
    );

    //
    // precompute transformation from shared to nested cluster basis
    //

    auto  row_trans = std::vector< blas::matrix< value_t > >( srowcb.id() + 1 );
    auto  col_trans = std::vector< blas::matrix< value_t > >( scolcb.id() + 1 );

    ::tbb::parallel_invoke(
        [&] () { detail::compute_transform( srowcb, *nrowcb, row_trans ); },
        [&] () { detail::compute_transform( scolcb, *ncolcb, col_trans ); }
    );
    
    //
    // construct uniform lowrank matrices with given cluster bases
    //
    
    auto  M = detail::build_h2( A, *nrowcb, *ncolcb, row_trans, col_trans, acc, compress );
    
    // only after building matrix
    if ( compress )
    {
        ::tbb::parallel_invoke(
            [&] () { matrix::compress< matrix::nested_cluster_basis< value_t > >( *nrowcb, acc ); },
            [&] () { matrix::compress< matrix::nested_cluster_basis< value_t > >( *ncolcb, acc ); }
        );
    }// if
    
    return  { std::move( nrowcb ), std::move( ncolcb ), std::move( M ) };
}

// //
// // construct H² matrix with corresponding cluster bases out of given H-matrix
// //
// template < approx::approximation_type basisapx_t >
// std::tuple< std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
//             std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
//             std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
// build_h2_rec ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
//                const basisapx_t &                                     basisapx,
//                const Hpro::TTruncAcc &                                acc,
//                const bool                                             compress,
//                const size_t                                           /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
// {
//     using value_t       = typename basisapx_t::value_t;
//     using cluster_basis = hlr::matrix::nested_cluster_basis< value_t >;

//     //
//     // mapping of index sets to lowrank matrices 
//     //

//     auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
//     auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

//     if ( is_blocked( A ) )
//     {
//         rowcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
//         colcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
//     }// if
    
//     detail::init_cluster_bases( A, *rowcb, *colcb );
    
//     auto  row_map      = detail::lr_mat_map_t< value_t >();
//     auto  row_coupling = detail::coupling_map_t< value_t >();
//     auto  row_mtx      = std::mutex();
//     auto  col_map      = detail::lr_mat_map_t< value_t >();
//     auto  col_coupling = detail::coupling_map_t< value_t >();
//     auto  col_mtx      = std::mutex();
    
//     detail::build_mat_map( A, *rowcb, *colcb,
//                            row_map, row_coupling, row_mtx,
//                            col_map, col_coupling, col_mtx );
    
//     //
//     // build cluster bases
//     //

//     auto  empty_list = detail::lr_mat_list_t< value_t >();
    
//     ::tbb::parallel_invoke (
//         [&,compress] () { detail::build_nested_cluster_basis( *rowcb, basisapx, acc, row_map, row_coupling, empty_list, false, compress ); },
//         [&,compress] () { detail::build_nested_cluster_basis( *colcb, basisapx, acc, col_map, col_coupling, empty_list, true,  compress ); }
//     );

//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *rowcb, id ); }
//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *colcb, id ); }

//     //
//     // construct uniform lowrank matrices with given cluster bases
//     //
    
//     auto  M = detail::build_h2( A, *rowcb, *colcb, acc, compress );
    
//     return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
// }
    
// template < approx::approximation_type basisapx_t >
// std::tuple< std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
//             std::unique_ptr< hlr::matrix::nested_cluster_basis< typename basisapx_t::value_t > >,
//             std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > > >
// build_h2_rec_sep ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
//                    const basisapx_t &                                     basisapx,
//                    const Hpro::TTruncAcc &                                acc,
//                    const bool                                             compress,
//                    const size_t                                           /* nseq */ = Hpro::CFG::Arith::max_seq_size ) // ignored
// {
//     using value_t       = typename basisapx_t::value_t;
//     using cluster_basis = hlr::matrix::nested_cluster_basis< value_t >;

//     //
//     // mapping of index sets to lowrank matrices 
//     //

//     auto  rowcb  = std::make_unique< cluster_basis >( A.row_is() );
//     auto  colcb  = std::make_unique< cluster_basis >( A.col_is() );

//     if ( is_blocked( A ) )
//     {
//         rowcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_rows() );
//         colcb->set_nsons( cptrcast( &A, Hpro::TBlockMatrix< value_t > )->nblock_cols() );
//     }// if
    
//     detail::init_cluster_bases( A, *rowcb, *colcb );
    
//     auto  row_map      = detail::lr_mat_map_t< value_t >();
//     auto  row_coupling = detail::coupling_map_t< value_t >();
//     auto  row_mtx      = std::mutex();
//     auto  col_map      = detail::lr_mat_map_t< value_t >();
//     auto  col_coupling = detail::coupling_map_t< value_t >();
//     auto  col_mtx      = std::mutex();
    
//     detail::build_mat_map( A, *rowcb, *colcb,
//                            row_map, row_coupling, row_mtx,
//                            col_map, col_coupling, col_mtx );
    
//     //
//     // build cluster bases
//     //

//     auto  empty_list = detail::lr_mat_list_t< value_t >();
    
//     ::tbb::parallel_invoke (
//         [&,compress] () { detail::build_nested_cluster_basis( *rowcb, basisapx, acc, row_map, row_coupling, empty_list, false, compress ); },
//         [&,compress] () { detail::build_nested_cluster_basis( *colcb, basisapx, acc, col_map, col_coupling, empty_list, true,  compress ); }
//     );

//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *rowcb, id ); }
//     { int  id = 0;  hlr::seq::matrix::detail::set_ids( *colcb, id ); }

//     //
//     // construct uniform lowrank matrices with given cluster bases
//     //
    
//     auto  M = detail::build_h2_sep( A, *rowcb, *colcb, acc, compress );
    
//     return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
// }
    
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

        N->set_id( M.id() );
        
        return N;
    }// if
    else
    {
        // assuming non-structured block and hence no parallel copy needed
        auto  N = M.copy();

        N->set_id( M.id() );
        
        return N;
    }// else
}

//
// return truncated copy of matrix
// - copy operation is performed in parallel for sub blocks
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
copy ( const Hpro::TMatrix< value_t > &  M,
       const Hpro::TTruncAcc &           acc )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

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

        N->set_id( M.id() );
        
        return N;
    }// if
    else
    {
        // assuming non-structured block and hence no parallel copy needed
        auto  N = M.copy();

        N->truncate( acc( M.row_is(), M.col_is() ) );
        N->set_id( M.id() );

        return N;
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
                            auto  B_ij = copy_compressible( * BM->block( i, j ) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );

        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        auto  N = M.copy();
        
        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_lowrank_sv( M ) )
    {
        auto  N = M.copy();
        
        N->set_id( M.id() );
        
        return N;
    }// if
    // else if ( Hpro::is_lowrank( M ) )
    // {
    //     auto  R = cptrcast( &M, Hpro::TRkMatrix< value_t > );
    //     auto  U = blas::copy( blas::mat_U( R ) );
    //     auto  V = blas::copy( blas::mat_V( R ) );

    //     if ( false )
    //     {
    //         auto  RU = blas::matrix< value_t >( U.ncols(), U.ncols() );
    //         auto  RV = blas::matrix< value_t >( V.ncols(), V.ncols() );

    //         blas::qr( U, RU );
    //         blas::qr( V, RV );

    //         auto  S = blas::prod( RU, blas::adjoint( RV ) );
    //         auto  N = std::make_unique< matrix::lrsmatrix< value_t > >( R->row_is(), R->col_is(), std::move( U ), std::move( S ), std::move( V ) );

    //         N->set_id( M.id() );
        
    //         return N;
    //     }// if
    //     else
    //     {
    //         auto  N = std::make_unique< matrix::lrmatrix< value_t > >( R->row_is(), R->col_is(), std::move( U ), std::move( V ) );

    //         N->set_id( M.id() );
        
    //         return N;
    //     }// else
    // }// if
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
                            auto  B_ij = copy_mixedprec( * BM->block( i, j ) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );

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
// - copy operation is performed in parallel for sub blocks
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
    else if ( matrix::is_lowrank( M ) )
    {
        //
        // copy low-rank data into tiled form
        //

        auto  RM = cptrcast( & M, matrix::lrmatrix< value_t > );
        auto  R  = std::make_unique< hlr::matrix::tiled_lrmatrix< value_t > >( RM->row_is(),
                                                                               RM->col_is(),
                                                                               ntile,
                                                                               RM->U(),
                                                                               RM->V() );

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
// - copy operation is performed in parallel for sub blocks
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
        auto  U  = hlr::matrix::to_dense( RM->U() );
        auto  V  = hlr::matrix::to_dense( RM->V() );
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
std::unique_ptr< Hpro::TMatrix< value_t > >
copy_uniform ( const Hpro::TMatrix< value_t > &                M,
               hlr::matrix::shared_cluster_basis< value_t > &  rowcb,
               hlr::matrix::shared_cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

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
    else if ( matrix::is_lowrank( M ) )
    {
        //
        // project into row/column cluster basis:
        //
        //   M = A·B^H = (V·V^H·A) (U·U^H·B)^H
        //             = U · (U^H·A)·(V^H·B)^H · V^H
        //             = U · S · V^H   with  S = (U^H·A)·(V^H·B)^H

        auto  R  = cptrcast( &M, matrix::lrmatrix< value_t > );

        auto  UA = rowcb.transform_forward( R->U() );
        auto  VB = colcb.transform_forward( R->V() );
        auto  S  = blas::prod( value_t(1), UA, blas::adjoint( VB ) );

        // auto  M1 = blas::prod( value_t(1), Hpro::blas_mat_A< value_t >( R ), blas::adjoint( Hpro::blas_mat_B< value_t >( R ) ) );
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
            } );

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

}}}// namespace hlr::tbb::matrix

#endif // __HLR_TBB_MATRIX_HH

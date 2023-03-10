#ifndef __HLR_TBB_DETAIL_MATRIX_HH
#define __HLR_TBB_DETAIL_MATRIX_HH
//
// Project     : HLR
// Module      : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>

#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/lrsmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>

#include <hlr/tbb/detail/uniform_basis.hh>

namespace hlr { namespace tbb { namespace matrix { namespace detail {

namespace hpro = HLIB;

using namespace hlr::matrix;

using hlr::uniform::is_matrix_map_t;
using hlr::tbb::uniform::detail::compute_extended_basis;
using hlr::tbb::uniform::detail::update_coupling;

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
                    const Hpro::TTruncAcc &      acc )
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< typename coeff_t::value_t, typename basisapx_t::value_t >,
                   "coefficient function and basis approximation must have equal value type" );
    
    assert( bct != nullptr );

    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;
    using basis_map_t   = std::unordered_map< indexset, cluster_basis *, indexset_hash >;
    using lrmat_map_t   = std::unordered_map< indexset, std::list< Hpro::TRkMatrix< value_t > * >, indexset_hash >;
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
    auto  insert_hier = [&] ( const Hpro::TBlockCluster *         node,
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

        auto             rowcl = node->rowcl();
        auto             colcl = node->colcl();
        cluster_basis *  rowcb = nullptr;
        cluster_basis *  colcb = nullptr;
        auto             lock  = std::scoped_lock( cbmtx );
                    
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
        
        ::tbb::parallel_for_each(
            nodes,
            [&] ( auto  node )
            {
                auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();

                if ( node->is_leaf() )
                {
                    // handled above
                    if ( node->is_adm() )
                    {
                        M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( node, acc ) );

                        {
                            auto  lock = std::scoped_lock( lmtx );

                            if ( is_lowrank( *M ) )
                            {
                                auto  R = ptrcast( M.get(), Hpro::TRkMatrix< value_t > );
                                    
                                rowmap[ M->row_is() ].push_back( R );
                                colmap[ M->col_is() ].push_back( R );
                            }// if
                                
                            // store always to maintain affinity
                            lrmat.push_back( M.get() );
                        }
                            
                        M->set_id( node->id() );
                        M->set_procs( node->procs() );

                        // insert_hier( node, M );
                        // create_cb( node );
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
            } );

        nodes = std::move( children );
        
        ::tbb::parallel_invoke(

            [&] ()
            {
                //
                // construct row bases for all block rows constructed on this level
                //

                auto  rowiss = std::deque< indexset >();

                for ( auto  [ is, matrices ] : rowmap )
                    rowiss.push_back( is );

                ::tbb::parallel_for_each(
                    rowiss,
                    [&] ( auto  is )                           
                    {
                        auto  matrices = rowmap.at( is );
                    
                        if ( matrices.size() == 0 )
                            return;

                        //
                        // compute column basis for
                        //
                        //   ( U?????V???'  U?????V???'  U?????V???'  ??? ) =
                        //
                        //                  ??? V???'        ???
                        //   ( U??? U??? U??? ??? ) ???    V???'     ??? =
                        //                  ???       V???'  ???
                        //                  ???          ??? ???
                        //
                        //                  ??? Q?????R???             ???'
                        //   ( U??? U??? U??? ??? ) ???      Q?????R???        ??? =
                        //                  ???           Q?????R???   ???
                        //                  ???                 ??? ???
                        //
                        //                  ??????Q???     ??? ???R???     ??????'
                        //   ( U??? U??? U??? ??? ) ??????  Q???   ????????  R???   ?????? =
                        //                  ??????    Q??? ??? ???    R??? ??????
                        //                  ??????      ?????? ???      ?????????
                        //
                        // Since diag(Q_i) is orthogonal, it can be omitted for row bases
                        // computation, leaving
                        //
                        //                  ???R???     ???'                 
                        //   ( U??? U??? U??? ??? ) ???  R???   ??? = ( U?????R???' U?????R???' U?????R???' ??? )
                        //                  ???    R??? ???                  
                        //                  ???      ??????                  
                        //
                        // of which a column basis is computed.
                        //

                        //
                        // form U = ( U?????R???' U?????R???' U?????R???' ??? )
                        //
            
                        size_t  nrows_U = is.size();
                        size_t  ncols_U = 0;

                        for ( auto &  R : matrices )
                            ncols_U += R->rank();

                        auto    U   = blas::matrix< value_t >( nrows_U, ncols_U );
                        size_t  pos = 0;

                        for ( auto &  R : matrices )
                        {
                            // R = U??V' = W??T??X'
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
                    } );
            },

            [&] ()
            {
                //
                // construct column bases for all block columns constructed on this level
                //

                auto  coliss = std::deque< indexset >();
            
                for ( auto  [ is, matrices ] : colmap )
                    coliss.push_back( is );

                ::tbb::parallel_for_each(
                    coliss,
                    [&] ( auto  is )                           
                    {
                        auto  matrices = colmap.at( is );

                        if ( matrices.size() == 0 )
                            return;

                        //
                        // compute column basis for
                        //
                        //   ???U?????V???'??? 
                        //   ???U?????V???'???
                        //   ???U?????V???'???
                        //   ???  ???   ???
                        //
                        // or row basis of
                        //
                        //   ???U?????V???'???' 
                        //   ???U?????V???'??? = ( V?????U???'  V?????U???'  V?????U???'  ??? ) =
                        //   ???U?????V???'???
                        //   ???  ???   ???
                        //
                        //                  ??? U???      ???'
                        //   ( V??? V??? V??? ??? ) ???   U???    ??? =
                        //                  ???     U???  ???
                        //                  ???       ??? ???
                        //
                        //                  ??? Q?????R???               ???'
                        //   ( V??? V??? V??? ??? ) ???       Q?????R???         ??? =
                        //                  ???             Q?????R???   ???
                        //                  ???                   ??? ???
                        //
                        //                  ??????Q???     ??? ???R???     ??????'
                        //   ( V??? V??? V??? ??? ) ??????  Q???   ????????  R???   ?????? =
                        //                  ??????    Q??? ??? ???    R??? ??????
                        //                  ??????      ?????? ???      ?????????
                        //
                        // Since diag(Q_i) is orthogonal, it can be omitted for column bases
                        // computation, leaving
                        //
                        //                  ???R???     ???'                
                        //   ( V??? V??? V??? ??? ) ???  R???   ??? = ( V?????R???' V?????R???' V?????R???' ??? )
                        //                  ???    R??? ???                
                        //                  ???      ??????
                        //
                        // of which a column basis is computed.
                        //

                        //
                        // form matrix V = ( V?????R???' V?????R???' V?????R???' ??? )
                        //

                        size_t  nrows_V = is.size();
                        size_t  ncols_V = 0;

                        for ( auto &  R : matrices )
                            ncols_V += R->rank();

                        auto    V   = blas::matrix< value_t >( nrows_V, ncols_V );
                        size_t  pos = 0;

                        for ( auto &  R : matrices )
                        {
                            // R' = (U??V')' = V??U' = X??T'??W'
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
                    } );
            }
        );

        //
        // now convert all blocks on this level
        //

        ::tbb::parallel_for_each(
            lrmat,
            [&] ( auto  M )                           
            {
                auto  R     = ptrcast( M, Hpro::TRkMatrix< value_t > );
                auto  rowcb = rowcb_map.at( R->row_is() );
                auto  colcb = colcb_map.at( R->col_is() );
                auto  Un    = rowcb->basis();
                auto  Vn    = colcb->basis();

                //
                // R = U??V' ??? Un (Un' U V' Vn) Vn'
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
            } );
    }// while
    
    return { std::move( rowcb_root ),
             std::move( colcb_root ),
             std::move( M_root ) };
}

//
// level-wise construction of uniform-H matrix from given H-matrix
//
template < typename basisapx_t >
std::unique_ptr< Hpro::TMatrix< typename basisapx_t::value_t > >
build_uniform_lvl ( const Hpro::TMatrix< typename basisapx_t::value_t > &  A,
                    const basisapx_t &                               basisapx,
                    const Hpro::TTruncAcc &                          acc,
                    cluster_basis< typename basisapx_t::value_t > &  rowcb_root,
                    cluster_basis< typename basisapx_t::value_t > &  colcb_root )
{
    using value_t       = typename basisapx_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;
    using basis_map_t   = std::unordered_map< indexset, cluster_basis *, indexset_hash >;
    using lrmat_map_t   = std::unordered_map< indexset, std::list< const Hpro::TRkMatrix< value_t > * >, indexset_hash >;
    using bmat_map_t    = std::unordered_map< Hpro::idx_t, Hpro::TBlockMatrix< value_t > * >;

    //
    // go BFS-style through matrix and construct leaves per level
    // then convert lowrank to uniform lowrank while constructing bases
    //

    // TODO: handle case of global lowrank matrix
    HLR_ASSERT( ! is_lowrank( A ) );
    
    auto  rowcb_map = basis_map_t();
    auto  colcb_map = basis_map_t();

    auto  M_root    = std::unique_ptr< Hpro::TMatrix< value_t > >();

    auto  matrices  = std::list< const Hpro::TMatrix< value_t > * >{ &A };
    auto  bmat_map  = bmat_map_t();

    auto  bmtx      = std::mutex(); // for bmat_map
    auto  cmtx      = std::mutex(); // for children list
    auto  lmtx      = std::mutex(); // for row/col map lists
    auto  cbmtx     = std::mutex(); // for rowcb/colcb map lists

    //
    // level-wise iteration for matrix construction
    //
    
    rowcb_map[ A.row_is() ] = & rowcb_root;
    colcb_map[ A.col_is() ] = & colcb_root;
    
    while ( ! matrices.empty() )
    {
        auto  children = decltype( matrices )();
        auto  rowmap   = lrmat_map_t();
        auto  colmap   = lrmat_map_t();
        auto  lrmat    = std::deque< const Hpro::TRkMatrix< value_t > * >();
        
        ::tbb::parallel_for_each(
            matrices,
            [&] ( auto  mat )
            {
                auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();

                if ( is_lowrank( mat ) )
                {
                    auto  R    = cptrcast( mat, Hpro::TRkMatrix< value_t > );
                    auto  lock = std::scoped_lock( lmtx );
                        
                    rowmap[ R->row_is() ].push_back( R );
                    colmap[ R->col_is() ].push_back( R );
                    lrmat.push_back( R );
                }// if
                else if ( is_dense( mat ) )
                {
                    M = mat->copy();
                }// if
                else if ( is_blocked( mat ) )
                {
                    auto  B = cptrcast( mat, Hpro::TBlockMatrix< value_t > );
                
                    // collect sub-blocks
                    {
                        auto  lock = std::scoped_lock( cmtx );

                        for ( uint  i = 0; i < B->nblock_rows(); ++i )
                            for ( uint  j = 0; j < B->nblock_cols(); ++j )
                                if ( ! is_null( B->block( i, j ) ) )
                                    children.push_back( B->block( i, j ) );
                    }

                    M = B->copy_struct();

                    // remember all block matrices for setting up hierarchy
                    {
                        auto  lock = std::scoped_lock( bmtx );
                        
                        bmat_map[ mat->id() ] = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );
                    }
                }// else

                //
                // set up hierarchy (parent <-> M)
                //

                if ( ! is_null( M ) )
                {
                    if ( mat == &A )
                    {
                        M_root = std::move( M );
                    }// if
                    else
                    {
                        auto                  mat_parent = mat->parent();
                        Hpro::TBlockMatrix< value_t > *  M_parent   = nullptr;

                        {
                            auto  lock_bmap = std::scoped_lock( bmtx );
                            
                            M_parent = bmat_map.at( mat_parent->id() );
                        }
                        
                        for ( uint  i = 0; i < mat_parent->nblock_rows(); ++i )
                        {
                            for ( uint  j = 0; j < mat_parent->nblock_cols(); ++j )
                            {
                                if ( mat_parent->block( i, j ) == mat )
                                {
                                    M_parent->set_block( i, j, M.release() );
                                    break;
                                }// if
                            }// for
                        }// for
                    }// if
                }// if

                //
                // fill mapping for row/column cluster basis
                //

                {
                    auto  lock_cb = std::scoped_lock( cbmtx );
                    
                    HLR_ASSERT( rowcb_map.find( mat->row_is() ) != rowcb_map.end() );
            
                    auto  rowcb = rowcb_map[ mat->row_is() ];

                    for ( uint  i = 0; i < rowcb->nsons(); ++i )
                    {
                        auto  son_i = rowcb->son(i);
                
                        if ( ! is_null( son_i ) )
                            rowcb_map[ son_i->is() ] = son_i;
                    }// for
            
                    HLR_ASSERT( colcb_map.find( mat->col_is() ) != colcb_map.end() );
            
                    auto  colcb = colcb_map[ mat->col_is() ];

                    for ( uint  i = 0; i < colcb->nsons(); ++i )
                    {
                        auto  son_i = colcb->son(i);
                
                        if ( ! is_null( son_i ) )
                            colcb_map[ son_i->is() ] = son_i;
                    }// for
                }
            } );

        matrices = std::move( children );
        
        ::tbb::parallel_invoke(

            [&] ()
            {
                //
                // construct row bases for all block rows constructed on this level
                //

                auto  rowiss = std::deque< indexset >();

                for ( auto  [ is, matrices ] : rowmap )
                    rowiss.push_back( is );

                ::tbb::parallel_for_each(
                    rowiss,
                    [&] ( auto  is )                           
                    {
                        auto  matrices = rowmap.at( is );
                    
                        if ( matrices.size() == 0 )
                            return;

                        //
                        // form U = ( U?????R???' U?????R???' U?????R???' ??? )
                        //
            
                        size_t  nrows_U = is.size();
                        size_t  ncols_U = 0;

                        for ( auto &  R : matrices )
                            ncols_U += R->rank();

                        auto    U       = blas::matrix< value_t >( nrows_U, ncols_U );
                        size_t  pos     = 0;
                        auto    pos_mtx = std::mutex();
                        
                        ::tbb::parallel_for_each(
                            matrices,
                            [&] ( auto &  R )
                            {
                                // R = U??V' = W??T??X'
                                auto  U_i = blas::mat_U< value_t >( R );
                                auto  V_i = blas::copy( blas::mat_V< value_t >( R ) );
                                auto  R_i = blas::matrix< value_t >();
                                auto  k   = R->rank();
                
                                blas::qr( V_i, R_i );

                                auto  UR_i  = blas::prod( U_i, blas::adjoint( R_i ) );

                                {
                                    auto  lock  = std::scoped_lock( pos_mtx );
                                    auto  U_sub = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k - 1 ) );

                                    blas::copy( UR_i, U_sub );
                
                                    pos += k;
                                }
                            } );

                        //
                        // QR of S and computation of row basis
                        //

                        auto  Un = basisapx.column_basis( U, acc );
            
                        // finally assign to cluster basis object
                        // (no change to "rowcb_map", therefore no lock)
                        rowcb_map.at( is )->set_basis( std::move( Un ) );
                    } );
            },

            [&] ()
            {
                //
                // construct column bases for all block columns constructed on this level
                //

                auto  coliss = std::deque< indexset >();
            
                for ( auto  [ is, matrices ] : colmap )
                    coliss.push_back( is );

                ::tbb::parallel_for_each(
                    coliss,
                    [&] ( auto  is )                           
                    {
                        auto  matrices = colmap.at( is );

                        if ( matrices.size() == 0 )
                            return;

                        //
                        // form matrix V = ( V?????R???' V?????R???' V?????R???' ??? )
                        //

                        size_t  nrows_V = is.size();
                        size_t  ncols_V = 0;

                        for ( auto &  R : matrices )
                            ncols_V += R->rank();

                        auto    V       = blas::matrix< value_t >( nrows_V, ncols_V );
                        size_t  pos     = 0;
                        auto    pos_mtx = std::mutex();
                        
                        ::tbb::parallel_for_each (
                            matrices,
                            [&] ( auto &  R )
                            {
                                // R' = (U??V')' = V??U' = X??T'??W'
                                auto  V_i = blas::copy( blas::mat_V< value_t >( R ) );
                                auto  U_i = blas::copy( blas::mat_U< value_t >( R ) );
                                auto  R_i = blas::matrix< value_t >();
                                auto  k   = R->rank();
                
                                blas::qr( U_i, R_i );

                                auto  VR_i  = blas::prod( V_i, blas::adjoint( R_i ) );

                                {
                                    auto  lock = std::scoped_lock( pos_mtx );
                                    auto  V_sub = blas::matrix< value_t >( V, blas::range::all, blas::range( pos, pos + k - 1 ) );

                                    blas::copy( VR_i, V_sub );
                
                                    pos += k;
                                }
                            } );

                        auto  Vn = basisapx.column_basis( V, acc );

                        // finally assign to cluster basis object
                        // (no change to "colcb_map", therefore no lock)
                        colcb_map.at( is )->set_basis( std::move( Vn ) );
                    } );
            }
        );

        //
        // now convert all blocks on this level
        //

        ::tbb::parallel_for_each(
            lrmat,
            [&] ( auto  R )
            {
                auto  rowcb = rowcb_map.at( R->row_is() );
                auto  colcb = colcb_map.at( R->col_is() );
                auto  Un    = rowcb->basis();
                auto  Vn    = colcb->basis();

                //
                // R = U??V' ??? Un (Un' U V' Vn) Vn'
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
                auto  R_parent = R->parent();
                auto  U_parent = bmat_map.at( R_parent->id() );

                for ( uint  i = 0; i < R_parent->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < R_parent->nblock_cols(); ++j )
                    {
                        if ( R_parent->block( i, j ) == R )
                        {
                            U_parent->set_block( i, j, RU.release() );
                            break;
                        }// if
                    }// for
                }// for
            } );
    }// while
    
    return M_root;
}

//
// recursively build uniform H-matrix while also constructing row/column cluster basis
// by updating bases after constructing low-rank blocks
//
using  mutex_map_t   = std::unordered_map< indexset, std::mutex, indexset_hash >;

// struct rec_basis_data_t
// {
//     // maps indexsets to set of uniform matrices sharing corresponding cluster basis
//     // and their mutexes
//     matrix_map_t   rowmap, colmap;
//     std::mutex     rowmapmtx, colmapmtx;

//     // mutexes for the lists within rowmap/colmap
//     mutex_map_t    rowmtxs, colmtxs;

//     // maps indexsets to set of LRS lowrank matrices contributing to cluster basis
//     matrix_map_t   rowlrmap, collrmap;
//     std::mutex     rowlrmapmtx, collrmapmtx;

//     //
//     // add given lowrank matrix to update lists
//     //
//     void
//     add_update ( Hpro::TMatrix< value_t > *  M )
//     {
//         HLR_ASSERT( ! is_null( M ) );
        
//         {
//             auto  is   = M->row_is();
//             auto  lock = std::scoped_lock( rowlrmapmtx );

//             rowlrmap[ is ].push_back( M );
//         }

//         {
//             auto  is   = M->col_is();
//             auto  lock = std::scoped_lock( collrmapmtx );

//             collrmap[ is ].push_back( M );
//         }
//     }

//     //
//     // update row cluster basis with all collected update blocks
//     //
//     template < typename cluster_basis,
//                typename basis_approx >
//     void
//     update_row_basis ( cluster_basis &          rowcb,
//                        cluster_basis &          colcb,
//                        const basis_approx &     basisapx,
//                        const Hpro::TTruncAcc &  acc )
//     {
//         using  value_t = typename cluster_basis::value_t;

//         // if ( rowcb.mutex().try_lock() )
//         {
//             do
//             {
//                 lrsmatrix< value_t > *  R = nullptr;

//                 //
//                 // TODO: update with _all_ matrices in list at once
//                 //
                
//                 {
//                     auto  is   = rowcb.is();
//                     auto  lock = std::scoped_lock( rowlrmapmtx );

//                     // check, if no updates are available
//                     // (check/return inside locked region to avoid race condition)
//                     if ( rowlrmap[ is ].empty() )
//                     {
//                         // finally unlock mutex
//                         // rowcb.mutex().unlock();
//                         return;
//                     }// if

//                     R = ptrcast( rowlrmap[ is ].front(), lrsmatrix< value_t > );
//                     rowlrmap[ is ].pop_front();
//                 }

//                 //
//                 // update cluster bases
//                 //

//                 auto  S = blas::matrix< value_t >();

//                 {
//                     auto  lock = std::scoped_lock( R->mutex() );
                    
//                     // For basis update we only need coupling consistent with
//                     // basis U and with correct norm. Any (orthogonal) changes
//                     // to column basis do not affect norm or row basis coupling.
//                     S = std::move( blas::copy( R->S() ) );
//                 }

//                 // since R can only be in single basis update list, no other modifications of U are possible
//                 auto  U  = R->U();
//                 auto  Un = compute_extended_row_basis( rowcb, U, S, acc, basisapx );

//                 update_row_coupling( rowcb, Un );

//                 rowcb.set_basis( std::move( Un ) );
                
//                 //
//                 // update R by Un??(Un'??U??S)
//                 //

//                 auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), U );

//                 {
//                     // to keep consistency, guard any modifications to R
//                     auto  lock = std::scoped_lock( R->mutex() );
                    
//                     // reset row basis to save memory as no longer needed
//                     // HACK: also reset row basis to signal usage of shared row basis
//                     R->clear_row_basis();

//                     // test if R should be replaced by uniform matrix
//                     if ( R->V().ncols() == 0 )
//                     {
//                         // std::cout << "converting " << R->id() << " " << R->block_is().to_string() << std::endl;
                        
//                         const auto  colis = R->col_is();
                        
//                         // lock to prevent concurrent changes to R coupling matrix
//                         // (busy lock to avoid dead lock)
//                         while ( ! colmtxs[ colis ].try_lock() ) {}

//                         auto  T  = blas::prod( TU, R->S() );
//                         auto  RU = new uniform_lrmatrix< value_t >( R->row_is(), R->col_is(), rowcb, colcb, std::move( T ) );

//                         // R is deprecated now
//                         R->set_coeff_unsafe( std::move( blas::matrix< value_t >() ) );

//                         // replace but don't delete as R may be used in matrix list
//                         if ( ! is_null( R->parent() ) )
//                             R->parent()->replace_block( R, RU );

//                         {
//                             // insert RU into set of matrices sharing column basis
//                             auto  clock = std::scoped_lock( colmapmtx );
                    
//                             colmap[ colis ].push_back( RU );
//                         }
                        
//                         colmtxs[ colis ].unlock();

//                         {
//                             // insert RU into set of matrices sharing row basis
//                             auto  rlock = std::scoped_lock( rowmapmtx );
                    
//                             rowmap[ rowcb.is() ].push_back( RU );
//                         }
//                     }// if
//                     else
//                     {
//                         // std::cout << "not converting " << R->id() << " " << R->block_is().to_string()  << std::endl;
                        
//                         R->set_coeff_unsafe( std::move( blas::prod( TU, R->S() ) ) );

//                         // insert RU into set of matrices sharing row basis
//                         auto  rlock = std::scoped_lock( rowmapmtx );
                    
//                         rowmap[ rowcb.is() ].push_back( R );
//                     }// else
//                 }
                
//             } while ( true );
//         }// if
//     }

//     //
//     // update row cluster basis with all collected update blocks
//     //
//     template < typename cluster_basis,
//                typename basis_approx >
//     void
//     update_col_basis ( cluster_basis &          colcb,
//                        cluster_basis &          rowcb,
//                        const basis_approx &     basisapx,
//                        const Hpro::TTruncAcc &  acc )
//     {
//         using  value_t = typename cluster_basis::value_t;

//         // if ( colcb.mutex().try_lock() )
//         {
//             do
//             {
//                 lrsmatrix< value_t > *  R = nullptr;
                
//                 //
//                 // TODO: update with _all_ matrices in list at once
//                 //

//                 {
//                     auto  is   = colcb.is();
//                     auto  lock = std::scoped_lock( collrmapmtx );

//                     // check, if no updates are available
//                     if ( collrmap[ is ].empty() )
//                     {
//                         // finally unlock mutex
//                         // colcb.mutex().unlock();
//                         return;
//                     }// if

//                     R = ptrcast( collrmap[ is ].front(), lrsmatrix< value_t > );
//                     collrmap[ is ].pop_front();
//                 }

//                 //
//                 // update cluster bases
//                 //

//                 auto  S  = blas::matrix< value_t >();

//                 {
//                     auto  lock = std::scoped_lock( R->mutex() );

//                     // For basis update we only need coupling consistent with
//                     // basis V and with correct norm. Any (orthogonal) changes
//                     // to row basis do not affect norm or column basis coupling.
//                     S = std::move( blas::copy( R->S() ) );
//                 }
                
//                 // since R can only be in single basis update list, no other modifications of V are possible
//                 auto  V  = R->V();
//                 auto  Vn = compute_extended_col_basis( colcb, S, V, acc, basisapx );

//                 update_col_coupling( colcb, Vn );
                
//                 colcb.set_basis( std::move( Vn ) );
                
//                 //
//                 // update R by (S??V'??Vn)??Vn'
//                 //

//                 auto  TV = blas::prod( blas::adjoint( colcb.basis() ), V );

//                 {
//                     // to keep consistency, guard any modifications to R
//                     auto  lock = std::scoped_lock( R->mutex() );

//                     // reset column basis to save memory as no longer needed
//                     // HACK: also reset column basis to signal usage of shared column basis
//                     R->clear_col_basis();

//                     // test if R should be replaced by uniform matrix
//                     if ( R->U().ncols() == 0 )
//                     {
//                         // std::cout << "converting " << R->id() << " " << R->block_is().to_string()  << std::endl;
                        
//                         const auto  rowis = R->row_is();
                        
//                         // lock to prevent concurrent changes to R coupling matrix
//                         // (busy lock to avoid dead lock)
//                         while ( ! rowmtxs[ rowis ].try_lock() ) {}

//                         auto  T  = blas::prod( R->S(), blas::adjoint( TV ) );
//                         auto  RU = new uniform_lrmatrix< value_t >( R->row_is(), R->col_is(), rowcb, colcb, std::move( T ) );

//                         // R is deprecated now
//                         R->set_coeff_unsafe( std::move( blas::matrix< value_t >() ) );

//                         // replace but don't delete as R may be used in matrix list
//                         if ( ! is_null( R->parent() ) )
//                             R->parent()->replace_block( R, RU );

//                         {
//                             // insert RU into set of matrices sharing row basis
//                             auto  rlock = std::scoped_lock( rowmapmtx );
                    
//                             rowmap[ rowis ].push_back( RU );
//                         }
                        
//                         rowmtxs[ rowis ].unlock();

//                         {
//                             // insert RU into set of matrices sharing column basis
//                             auto  clock = std::scoped_lock( colmapmtx );
                    
//                             colmap[ colcb.is() ].push_back( RU );
//                         }
//                     }// if
//                     else
//                     {
//                         // std::cout << "not converting " << R->id() << " " << R->block_is().to_string()  << std::endl;
                        
//                         R->set_coeff_unsafe( std::move( blas::prod( R->S(), blas::adjoint( TV ) ) ) );

//                         // insert R into set of matrices sharing column basis to be used during basis updates
//                         auto  clock = std::scoped_lock( colmapmtx );
                    
//                         colmap[ colcb.is() ].push_back( R );
//                     }// else
//                 }

//             } while ( true );
//         }// if
//     }

//     //
//     // extend row basis <cb> by block W??T??X' (X is not needed for computation)
//     //
//     // - identical to implementation in "arith/detail/uniform.hh" but thread-safe,
//     //   hence, for details look into original code
//     //
//     template < typename value_t,
//                typename basis_approx_t >
//     blas::matrix< value_t >
//     compute_extended_row_basis ( const cluster_basis< value_t > &  cb,
//                                  const blas::matrix< value_t > &   W,
//                                  const blas::matrix< value_t > &   T,
//                                  const Hpro::TTruncAcc &           acc,
//                                  const basis_approx_t &            basisapx )
//     {
//         using  real_t = Hpro::real_type_t< value_t >;

//         //
//         // collect all scaled coupling matrices
//         //
        
//         auto    couplings = std::list< blas::matrix< value_t > >();
//         size_t  nrows_S   = T.ncols();

//         {
//             auto  lock = std::scoped_lock( rowmapmtx );
            
//             if ( rowmap.find( cb.is() ) != rowmap.end() )
//             {
//                 auto  lock_is = std::scoped_lock( rowmtxs[ cb.is() ] );

//                 // TODO: maybe first copy list to local list and then proceed
//                 //       after unlocking mutex as soon as possible
                
//                 for ( auto  M_i : rowmap.at( cb.is() ) )
//                 {
//                     if ( is_uniform_lowrank( M_i ) )
//                     {
//                         auto        lock_i = std::scoped_lock( M_i->mutex() );
//                         const auto  R_i    = cptrcast( M_i, uniform_lrmatrix< value_t > );
//                         const auto  rank   = R_i->col_rank();
//                         auto        S_i    = blas::copy( R_i->coeff() );
//                         auto        norm   = norm::spectral( S_i );
                        
//                         if ( norm != real_t(0) )
//                         {
//                             blas::scale( value_t(1) / norm, S_i );
//                             couplings.push_back( std::move( S_i ) );
//                             nrows_S += rank;
//                         }// if
//                     }// if
//                     else if ( is_lowrankS( M_i ) )
//                     {
//                         // ASSUMPTION: U from M_i = U??S??V' is identical to shared cluster basis
//                         auto        lock_i = std::scoped_lock( M_i->mutex() );
//                         const auto  R_i    = cptrcast( M_i, lrsmatrix< value_t > );
//                         auto        S_i    = blas::copy( R_i->S() );
//                         const auto  rank   = S_i.ncols();
//                         auto        norm   = ( rank == 0 ? real_t(0) : norm::spectral( S_i ) );
                        
//                         if ( norm != real_t(0) )
//                         {
//                             blas::scale( value_t(1) / norm, S_i );
//                             couplings.push_back( std::move( S_i ) );
//                             nrows_S += rank;
//                         }// if
//                     }// if
//                 }// for
//             }// if
//         }

//         if ( couplings.empty() )
//         {
//             return std::move( blas::copy( W ) );
//         }// if
//         else
//         {
//             auto  U  = cb.basis();
//             auto  Ue = blas::join_row< value_t >( { U, W } );

//             // assemble all scaled coupling matrices into joined matrix
//             auto    S   = blas::matrix< value_t >( nrows_S, Ue.ncols() );
//             size_t  pos = 0;

//             for ( auto  S_i : couplings )
//             {
//                 const auto  rank = S_i.ncols();
//                 auto        S_sub = blas::matrix< value_t >( S,
//                                                              blas::range( pos, pos + rank-1 ),
//                                                              blas::range( 0, U.ncols() - 1 ) );
                        
//                 blas::copy( blas::adjoint( S_i ), S_sub );
//                 pos += rank;
//             }// for

//             //
//             // and add part from W??T??X'
//             //
        
//             const auto  rank = T.ncols();
//             auto        S_i  = blas::copy( T );
//             auto        norm = norm::spectral( T );
            
//             if ( norm != real_t(0) )
//                 blas::scale( value_t(1) / norm, S_i );
            
//             auto  S_sub = blas::matrix< value_t >( S,
//                                                    blas::range( pos, pos + rank-1 ),
//                                                    blas::range( U.ncols(), Ue.ncols() - 1 ) );
            
//             blas::copy( blas::adjoint( S_i ), S_sub );

//             //
//             // form product Ue??S and compute column basis
//             //
            
//             auto  R = blas::matrix< value_t >();
        
//             blas::qr( S, R, false );

//             auto  UeR = blas::prod( Ue, blas::adjoint( R ) );
//             auto  Un  = basisapx.column_basis( UeR, acc );

//             return  Un;
//         }// else
//     }

//     //
//     // extend column basis <cb> by block W??T??X' (W is not needed for computation)
//     //
//     // - identical to implementation in "arith/detail/uniform.hh" but thread-safe,
//     //   hence, for details look into original code
//     //
//     template < typename value_t,
//                typename approx_t >
//     blas::matrix< value_t >
//     compute_extended_col_basis ( const cluster_basis< value_t > &  cb,
//                                  const blas::matrix< value_t > &   T,
//                                  const blas::matrix< value_t > &   X,
//                                  const Hpro::TTruncAcc &           acc,
//                                  const approx_t &                  approx )
//     {
//         using  real_t = Hpro::real_type_t< value_t >;

//         //
//         // collect all scaled coupling matrices
//         //
        
//         auto    couplings = std::list< blas::matrix< value_t > >();
//         size_t  nrows_S   = T.nrows();
    
//         {
//             auto  lock = std::scoped_lock( colmapmtx );
            
//             if ( colmap.find( cb.is() ) != colmap.end() )
//             {
//                 auto  lock_is = std::scoped_lock( colmtxs[ cb.is() ] );
                
//                 // TODO: maybe first copy list to local list and then proceed
//                 //       after unlocking mutex as soon as possible
                
//                 for ( auto  M_i : colmap.at( cb.is() ) )
//                 {
//                     if ( is_uniform_lowrank( M_i ) )
//                     {
//                         auto        lock_i = std::scoped_lock( M_i->mutex() );
//                         const auto  R_i    = cptrcast( M_i, uniform_lrmatrix< value_t > );
//                         const auto  rank   = R_i->row_rank();
//                         auto        S_i    = blas::copy( R_i->coeff() );
//                         auto        norm   = norm::spectral( S_i );

//                         if ( norm != real_t(0) )
//                         {
//                             blas::scale( value_t(1) / norm, S_i );
//                             couplings.push_back( std::move( S_i ) );
//                             nrows_S += rank;
//                         }// if
//                     }// if
//                     else if ( is_lowrankS( M_i ) )
//                     {
//                         // ASSUMPTION: V from M_i = U??S??V' is identical to shared cluster basis
//                         auto        lock_i = std::scoped_lock( M_i->mutex() );
//                         const auto  R_i    = cptrcast( M_i, lrsmatrix< value_t > );
//                         auto        S_i    = blas::copy( R_i->S() );
//                         const auto  rank   = R_i->nrows();
//                         auto        norm   = ( rank == 0 ? real_t(0) : norm::spectral( S_i ) );

//                         if ( norm != real_t(0) )
//                         {
//                             blas::scale( value_t(1) / norm, S_i );
//                             couplings.push_back( std::move( S_i ) );
//                             nrows_S += rank;
//                         }// if
//                     }// if
//                 }// for
//             }// if
//         }

//         if ( couplings.empty() )
//         {
//             return std::move( blas::copy( X ) );
//         }// if
//         else
//         {
//             auto  V  = cb.basis();
//             auto  Ve = blas::join_row< value_t >( { V, X } );
    
//             // assemble all scaled coupling matrices into joined matrix
//             auto    S   = blas::matrix< value_t >( nrows_S, Ve.ncols() );
//             size_t  pos = 0;

//             for ( auto  S_i : couplings )
//             {
//                 const auto  rank  = S_i.ncols();
//                 auto        S_sub = blas::matrix< value_t >( S,
//                                                              blas::range( pos, pos + rank-1 ),
//                                                              blas::range( 0, V.ncols() - 1 ) );

//                 blas::copy( S_i, S_sub );
//                 pos += rank;
//             }// for

//             //
//             // add part from W??T??X'
//             //
        
//             const auto  rank = T.nrows();
//             auto        S_i  = blas::copy( T );
//             auto        norm = norm::spectral( T );
            
//             if ( norm != real_t(0) )
//                 blas::scale( value_t(1) / norm, S_i );
            
//             auto  S_sub = blas::matrix< value_t >( S,
//                                                    blas::range( pos, pos + rank-1 ),
//                                                    blas::range( V.ncols(), Ve.ncols() - 1 ) );
            
//             blas::copy( S_i, S_sub );

//             //
//             // form product Ve??S' and compute column basis
//             //
            
//             auto  R = blas::matrix< value_t >();

//             blas::qr( S, R, false );

//             auto  VeR = blas::prod( Ve, blas::adjoint( R ) );
//             auto  Vn  = approx.column_basis( VeR, acc );
        
//             return  Vn;
//         }// else
//     }

//     //
//     // update coupling matrices for all blocks sharing row basis <cb> to new basis <Un>
//     //
//     template < typename value_t >
//     void
//     update_row_coupling ( const cluster_basis< value_t > &  cb,
//                           const blas::matrix< value_t > &   Un )
//     {
//         auto  lock = std::scoped_lock( rowmapmtx );
        
//         if ( rowmap.find( cb.is() ) != rowmap.end() )
//         {
//             auto  lock_is = std::scoped_lock( rowmtxs[ cb.is() ] );
//             auto  U       = cb.basis();
//             auto  TU      = blas::prod( blas::adjoint( Un ), U );

//             for ( auto  M_i : rowmap.at( cb.is() ) )
//             {
//                 if ( is_uniform_lowrank( M_i ) )
//                 {
//                     auto  lock_i = std::scoped_lock( M_i->mutex() );
//                     auto  R_i    = ptrcast( M_i, uniform_lrmatrix< value_t > );
//                     auto  Sn_i   = blas::prod( TU, R_i->coeff() );

//                     R_i->set_coeff_unsafe( std::move( Sn_i ) );
//                 }// if
//                 else if ( is_lowrankS( M_i ) )
//                 {
//                     auto  lock_i = std::scoped_lock( M_i->mutex() );
//                     auto  R_i    = ptrcast( M_i, lrsmatrix< value_t > );

//                     if ( R_i->S().ncols() == 0 )
//                         continue;
                    
//                     auto  Sn_i   = blas::prod( TU, R_i->S() );
                
//                     R_i->set_coeff_unsafe( std::move( Sn_i ) );
//                 }// if
//             }// for
//         }// if
//     }

//     //
//     // update coupling matrices for all blocks sharing column basis <cb> to new basis <Vn>
//     //
//     template < typename value_t >
//     void
//     update_col_coupling ( const cluster_basis< value_t > &  cb,
//                           const blas::matrix< value_t > &   Vn )
//     {
//         auto  lock = std::scoped_lock( colmapmtx );
        
//         if ( colmap.find( cb.is() ) != colmap.end() )
//         {
//             auto  lock_is = std::scoped_lock( colmtxs[ cb.is() ] );
//             auto  V       = cb.basis();
//             auto  TV      = blas::prod( blas::adjoint( Vn ), V );

//             for ( auto  M_i : colmap.at( cb.is() ) )
//             {
//                 if ( is_uniform_lowrank( M_i ) )
//                 {
//                     auto  lock_i = std::scoped_lock( M_i->mutex() );
//                     auto  R_i    = ptrcast( M_i, uniform_lrmatrix< value_t > );
//                     auto  Sn_i   = blas::prod( R_i->coeff(), blas::adjoint( TV ) );
                
//                     R_i->set_coeff_unsafe( std::move( Sn_i ) );
//                 }// if
//                 else if ( is_lowrankS( M_i ) )
//                 {
//                     auto  lock_i = std::scoped_lock( M_i->mutex() );
//                     auto  R_i    = ptrcast( M_i, lrsmatrix< value_t > );

//                     if ( R_i->S().ncols() == 0 )
//                         continue;
                    
//                     auto  Sn_i   = blas::prod( R_i->S(), blas::adjoint( TV ) );
                
//                     R_i->set_coeff_unsafe( std::move( Sn_i ) );
//                 }// if
//             }// for
//         }// if
//     }
// };

// template < typename coeff_t,
//            typename lrapx_t,
//            typename basisapx_t >
// std::unique_ptr< Hpro::TMatrix< value_t > >
// build_uniform_rec ( const Hpro::TBlockCluster *                   bct,
//                     const coeff_t &                               coeff,
//                     const lrapx_t &                               lrapx,
//                     const basisapx_t &                            basisapx,
//                     const Hpro::TTruncAcc &                       acc,
//                     Hpro::TBlockMatrix< value_t > *                          parent,
//                     cluster_basis< typename coeff_t::value_t > &  rowcb,
//                     cluster_basis< typename coeff_t::value_t > &  colcb,
//                     rec_basis_data_t &                            basis_data )
// {
//     using value_t = typename coeff_t::value_t;

//     //
//     // local function to put matrix <M> into hierarchy
//     //
//     auto  add_to_parent = [bct,parent] ( Hpro::TMatrix< value_t > *  M )
//     {
//         if ( ! is_null_all( parent ) )
//         {
//             HLR_ASSERT( ! is_null( bct->parent() ) );

//             // also set remaining attributes
//             M->set_id( bct->id() );
//             M->set_procs( bct->procs() );
            
//             for ( uint  i = 0; i < bct->parent()->nrows(); ++i )
//             {
//                 for ( uint  j = 0; j < bct->parent()->ncols(); ++j )
//                 {
//                     if ( bct->parent()->son( i, j ) == bct )
//                     {
//                         parent->set_block( i, j, M );
//                         return;
//                     }// if
//                 }// for
//             }// for
//         }// if
//     };
    
//     //
//     // decide upon cluster type, how to construct matrix
//     //

//     if ( bct->is_leaf() )
//     {
//         if ( bct->is_adm() )
//         {
//             auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( bct, acc ) );

//             if ( is_lowrank( *M ) )
//             {
//                 //
//                 // compute LRS representation W??T??X' = U??V' = M
//                 //

//                 auto  R  = ptrcast( M.get(), Hpro::TRkMatrix< value_t > );
//                 auto  W  = std::move( blas::mat_U< value_t >( R ) );
//                 auto  X  = std::move( blas::mat_V< value_t >( R ) );
//                 auto  Rw = blas::matrix< value_t >();
//                 auto  Rx = blas::matrix< value_t >();

//                 blas::qr( W, Rw );
//                 blas::qr( X, Rx );

//                 auto  T  = blas::prod( Rw, blas::adjoint( Rx ) );
//                 auto  RU = std::make_unique< lrsmatrix< value_t > >( R->row_is(), R->col_is(), std::move( W ), std::move( T ), std::move( X ) );

//                 // immediately put into hierarchy so it can be replace by basis update procedures
//                 add_to_parent( RU.get() );
                
//                 // rowmap[ RU->row_is() ].push_back( RU.get() );
//                 // colmap[ RU->col_is() ].push_back( RU.get() );

//                 basis_data.add_update( RU.get() );
//                 basis_data.update_row_basis( rowcb, colcb, basisapx, acc );
//                 basis_data.update_col_basis( colcb, rowcb, basisapx, acc );
                
//                 //
//                 // ATTENTION: RU is only correct _after_ the update of the cluster bases
//                 //
                
//                 RU.release();
//             }// if
//             else
//                 add_to_parent( M.release() );
//         }// if
//         else
//         {
//             auto  M = coeff.build( bct->is().row_is(), bct->is().col_is() );

//             add_to_parent( M.release() );
//         }// else
//     }// if
//     else
//     {
//         auto  M = std::make_unique< Hpro::TBlockMatrix< value_t > >( bct );

//         // make sure, block structure is correct
//         if (( M->nblock_rows() != bct->nrows() ) ||
//             ( M->nblock_cols() != bct->ncols() ))
//             M->set_block_struct( bct->nrows(), bct->ncols() );

//         //
//         // build cluster basis sons
//         //
//         for ( uint  i = 0; i < M->nblock_rows(); ++i )
//         {
//             auto  rowcb_i = rowcb.son( i );

//             for ( uint  j = 0; j < M->nblock_cols(); ++j )
//             {
//                 auto  colcb_j = colcb.son( j );
                
//                 if ( ! is_null( bct->son( i, j ) ) )
//                 {
//                     if ( is_null( rowcb_i ) )
//                     {
//                         rowcb_i = new cluster_basis< value_t >( bct->son( i, j )->is().row_is() );
//                         rowcb_i->set_nsons( bct->son( i, j )->rowcl()->nsons() );
//                         rowcb.set_son( i, rowcb_i );
//                     }// if
            
//                     if ( is_null( colcb_j ) )
//                     {
//                         colcb_j = new cluster_basis< value_t >( bct->son( i, j )->is().col_is() );
//                         colcb_j->set_nsons( bct->son( i, j )->colcl()->nsons() );
//                         colcb.set_son( j, colcb_j );
//                     }// if
//                 }// if
//             }// for
//         }// for

//         ::tbb::parallel_for(
//             ::tbb::blocked_range2d< uint >( 0, M->nblock_rows(),
//                                             0, M->nblock_cols() ),
//             [&,bct] ( const ::tbb::blocked_range2d< uint > &  r )
//             {
//                 for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
//                 {
//                     auto  rowcb_i = rowcb.son( i );
                    
//                     for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
//                     {
//                         auto  colcb_j = colcb.son( j );
                
//                         if ( ! is_null( bct->son( i, j ) ) )
//                         {
//                             HLR_ASSERT( ! is_null_all( rowcb_i, colcb_j ) );
                            
//                             build_uniform_rec( bct->son( i, j ), coeff, lrapx, basisapx, acc, M.get(), *rowcb_i, *colcb_j, basis_data );
//                         }// if
//                     }// for
//                 }// for
//             } );

//         // make value type consistent in block matrix and sub blocks
//         M->adjust_value_type();

//         // top-level matrix is actually returned to outer function
//         if ( is_null( parent ) )
//             return M;
//         else 
//             add_to_parent( M.release() );
//     }// else

//     // return dummy
//     return std::unique_ptr< Hpro::TMatrix< value_t > >();
// }

template < typename basisapx_t >
struct rec_uniform_construction
{
    using  value_t = typename basisapx_t::value_t;
    
    // maps indexsets to set of uniform matrices sharing corresponding cluster basis
    // and their mutexes
    is_matrix_map_t< value_t >  rowmap, colmap;
    std::mutex                  rowmapmtx, colmapmtx;

    // used basis approximation algorithm
    const basisapx_t &          basisapx;

    //
    // ctor
    //
    rec_uniform_construction ( const basisapx_t &  abasisapx )
            : basisapx( abasisapx )
    {}
    
    template < typename coeff_t,
               typename lrapx_t >
    std::unique_ptr< Hpro::TMatrix< value_t > >
    build ( const Hpro::TBlockCluster *                   bct,
            const Hpro::TTruncAcc &                       acc,
            const coeff_t &                               coeff,
            const lrapx_t &                               lrapx,
            cluster_basis< typename coeff_t::value_t > &  rowcb,
            cluster_basis< typename coeff_t::value_t > &  colcb )
    {
        using value_t = typename coeff_t::value_t;

        using namespace hlr::matrix;
        
        //
        // decide upon cluster type, how to construct matrix
        //

        auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
        if ( bct->is_leaf() )
        {
            if ( bct->is_adm() )
            {
                M = std::unique_ptr< Hpro::TMatrix< value_t > >( lrapx.build( bct, acc ) );

                if ( is_lowrank( *M ) )
                {
                    //
                    // compute LRS representation W??T??X' = U??V' = M
                    //

                    auto  R  = ptrcast( M.get(), Hpro::TRkMatrix< value_t > );
                    auto  W  = std::move( blas::mat_U< value_t >( R ) ); // reuse storage from R
                    auto  X  = std::move( blas::mat_V< value_t >( R ) );
                    auto  Rw = blas::matrix< value_t >();
                    auto  Rx = blas::matrix< value_t >();

                    ::tbb::parallel_invoke( [&] () { blas::qr( W, Rw ); },
                                            [&] () { blas::qr( X, Rx ); } );

                    HLR_ASSERT( Rw.ncols() != 0 );
                    HLR_ASSERT( Rx.ncols() != 0 );
                
                    auto  T = blas::prod( Rw, blas::adjoint( Rx ) );

                    ::tbb::this_task_arena::isolate( [&] ()
                    {
                        auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
                
                        ::tbb::parallel_invoke(
                            [&] ()
                            {
                                auto  Un = compute_extended_basis( rowcb, W, T, acc, basisapx, rowmap, rowmapmtx, apply_adjoint );
                        
                                update( rowcb, Un, rowmap, rowmapmtx, false );
                                rowcb.set_basis( std::move( Un ) );
                            },
                
                            [&] ()
                            {
                                auto  Vn = compute_extended_basis( colcb, X, T, acc, basisapx, colmap, colmapmtx, apply_normal );
                        
                                update_coupling( colcb, Vn, colmap, colmapmtx, true );
                                colcb.set_basis( std::move( Vn ) );
                            }
                        );

                        //
                        // transform T into new bases
                        //

                        auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), W );
                        auto  TV = blas::prod( blas::adjoint( colcb.basis() ), X );
                        auto  TS = blas::prod( TU, T );
                        auto  S  = blas::prod( TS, blas::adjoint( TV ) );

                        M = std::make_unique< uniform_lrmatrix< value_t > >( M->row_is(), M->col_is(), rowcb, colcb, std::move( S ) );

                        {
                            auto  lock_is = std::scoped_lock( rowmapmtx, colmapmtx );

                            rowmap[ rowcb.is() ].push_back( M.get() );
                            colmap[ colcb.is() ].push_back( M.get() );
                        }
                    } );
                }// if
            }// if
            else
            {
                M = coeff.build( bct->is().row_is(), bct->is().col_is() );

                if ( is_dense( *M ) )
                {
                    auto  D  = cptrcast( M.get(), Hpro::TDenseMatrix< value_t > );
                    auto  DD = blas::copy( blas::mat( D ) );

                    return  M = std::move( std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( DD ) ) );
                }// if
            }// else
        }// if
        else
        {
            M = std::make_unique< Hpro::TBlockMatrix< value_t > >( bct );

            auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );
        
            // make sure, block structure is correct
            B->set_block_struct( bct->nrows(), bct->ncols() );

            // recurse
            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                                0, B->nblock_cols() ),
                [&,bct,B] ( const ::tbb::blocked_range2d< uint > &  r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    {
                        auto  rowcb_i = rowcb.son( i );
                    
                        for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                        {
                            auto  colcb_j = colcb.son( j );
                
                            if ( ! is_null( bct->son( i, j ) ) )
                            {
                                HLR_ASSERT( ! is_null_all( rowcb_i, colcb_j ) );
                            
                                auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc, *rowcb_i, *colcb_j );

                                B->set_block( i, j, B_ij.release() );
                            }// if
                        }// for
                    }// for
                } );

            // make value type consistent in block matrix and sub blocks
            B->adjust_value_type();
        }// else

        M->set_id( bct->id() );
        M->set_procs( bct->procs() );
    
        return M;
    }

    std::unique_ptr< Hpro::TMatrix< value_t > >
    build ( const Hpro::TMatrix< value_t > &  A,
            const Hpro::TTruncAcc &           acc,
            cluster_basis< value_t > &        rowcb,
            cluster_basis< value_t > &        colcb )
    {
        using namespace hlr::matrix;

        //
        // decide upon cluster type, how to construct matrix
        //

        auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
        if ( is_lowrank( A ) )
        {
            //
            // compute LRS representation W??T??X' = U??V' = M
            //

            auto  R  = cptrcast( &A, Hpro::TRkMatrix< value_t > );
            auto  W  = blas::copy( blas::mat_U< value_t >( R ) );
            auto  X  = blas::copy( blas::mat_V< value_t >( R ) );
            auto  Rw = blas::matrix< value_t >();
            auto  Rx = blas::matrix< value_t >();

            ::tbb::parallel_invoke( [&] () { blas::qr( W, Rw ); },
                                    [&] () { blas::qr( X, Rx ); } );

            HLR_ASSERT( Rw.ncols() != 0 );
            HLR_ASSERT( Rx.ncols() != 0 );
                
            auto  T     = blas::prod( Rw, blas::adjoint( Rx ) );

            ::tbb::this_task_arena::isolate( [&] ()
            {
                auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
            
                ::tbb::parallel_invoke(
                    [&] ()
                    {
                        auto  Un = compute_extended_basis( rowcb, W, T, acc, basisapx, rowmap, rowmapmtx, apply_adjoint );
                
                        update_coupling( rowcb, Un, rowmap, rowmapmtx, false );
                        rowcb.set_basis( std::move( Un ) );
                    },
        
                    [&] ()
                    {
                        auto  Vn = compute_extended_basis( colcb, X, T, acc, basisapx, colmap, colmapmtx, apply_normal );
                
                        update_coupling( colcb, Vn, colmap, colmapmtx, true );
                        colcb.set_basis( std::move( Vn ) );
                    }
                );

                //
                // transform T into new bases
                //

                auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), W );
                auto  TV = blas::prod( blas::adjoint( colcb.basis() ), X );
                auto  TS = blas::prod( TU, T );
                auto  S  = blas::prod( TS, blas::adjoint( TV ) );

                M = std::make_unique< uniform_lrmatrix< value_t > >( A.row_is(), A.col_is(), rowcb, colcb, std::move( S ) );

                {
                    auto  lock_map = std::scoped_lock( rowmapmtx, colmapmtx );

                    rowmap[ rowcb.is() ].push_back( M.get() );
                    colmap[ colcb.is() ].push_back( M.get() );
                }
            } );
        }// if
        else if ( is_blocked( A ) )
        {
            auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
            M = std::make_unique< Hpro::TBlockMatrix< value_t > >();

            auto  BM = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

            BM->copy_struct_from( BA );

            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( 0, BA->nblock_rows(),
                                                0, BA->nblock_cols() ),
                [&,BA,BM] ( const ::tbb::blocked_range2d< uint > &  r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    {
                        auto  rowcb_i = rowcb.son( i );
                    
                        for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                        {
                            auto  colcb_j = colcb.son( j );
                
                            if ( ! is_null( BA->block( i, j ) ) )
                            {
                                HLR_ASSERT( ! is_null_any( rowcb_i, colcb_j ) );
                            
                                auto  B_ij = build( *BA->block( i, j ), acc, *rowcb_i, *colcb_j );

                                BM->set_block( i, j, B_ij.release() );
                            }// if
                        }// for
                    }// for
                } );
        }// if
        else if ( is_dense( A ) )
        {
            auto  D  = cptrcast( &A, Hpro::TDenseMatrix< value_t > );
            auto  DD = blas::copy( blas::mat( D ) );

            return  std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( DD ) );
        }// if
        else
        {
            M = A.copy();
        }// else

        M->set_id( A.id() );
        M->set_procs( A.procs() );
    
        return M;
    }
};


template < typename value_t >
void
init_cluster_bases ( const Hpro::TBlockCluster *  bct,
                     cluster_basis< value_t > &   rowcb,
                     cluster_basis< value_t > &   colcb )
{
    //
    // decide upon cluster type, how to construct matrix
    //

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
init_cluster_bases ( const Hpro::TMatrix< value_t > &  M,
                     cluster_basis< value_t > &        rowcb,
                     cluster_basis< value_t > &        colcb )
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

}}}}// namespace hlr::tbb::matrix::detail

#endif // __HLR_TBB_DETAIL_MATRIX_HH

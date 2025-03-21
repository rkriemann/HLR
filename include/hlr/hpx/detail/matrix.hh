#ifndef __HLR_HPX_DETAIL_MATRIX_HH
#define __HLR_HPX_DETAIL_MATRIX_HH
//
// Project     : HLR
// Module      : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpx/parallel/task_block.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>

#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/lrsmatrix.hh>

namespace hlr { namespace hpx { namespace matrix { namespace detail {

using ::hpx::parallel::v2::define_task_block;
using ::hpx::for_each;
using ::hpx::execution::par;

namespace hpro = HLIB;

using namespace hlr::matrix;

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
                    const hpro::TTruncAcc &      acc )
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< typename coeff_t::value_t, typename basisapx_t::value_t >,
                   "coefficient function and basis approximation must have equal value type" );
    
    assert( bct != nullptr );

    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;
    using basis_map_t   = std::unordered_map< indexset, cluster_basis *, indexset_hash >;
    using lrmat_map_t   = std::unordered_map< indexset, std::list< hpro::TRkMatrix * >, indexset_hash >;
    using bmat_map_t    = std::unordered_map< hpro::idx_t, hpro::TBlockMatrix * >;

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

    auto  M_root     = std::unique_ptr< hpro::TMatrix >();

    auto  nodes      = std::deque< const hpro::TBlockCluster * >{ bct };
    auto  bmat_map   = bmat_map_t();

    auto  bmtx       = std::mutex(); // for bmat_map
    auto  cmtx       = std::mutex(); // for children list
    auto  lmtx       = std::mutex(); // for row/col map lists
    auto  cbmtx      = std::mutex(); // for rowcb/colcb map lists

    //
    // local function to set up hierarchy (parent <-> M)
    //
    auto  insert_hier = [&] ( const hpro::TBlockCluster *         node,
                              std::unique_ptr< hpro::TMatrix > &  M )
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
    auto  create_cb = [&] ( const hpro::TBlockCluster *  node )
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
        auto  lrmat    = std::deque< hpro::TRkMatrix * >();
        
        for_each( par, nodes.begin(), nodes.end(), 
                  [&] ( auto  node )
                  {
                      auto  M = std::unique_ptr< hpro::TMatrix >();
                      
                      if ( node->is_leaf() )
                      {
                          // handled above
                          if ( node->is_adm() )
                          {
                              M = std::unique_ptr< hpro::TMatrix >( lrapx.build( node, acc ) );
                              
                              if ( is_lowrank( *M ) )
                              {
                                  auto  R    = ptrcast( M.get(), hpro::TRkMatrix );
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

                          M = std::make_unique< hpro::TBlockMatrix >( node );
        
                          auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

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
        
        define_task_block( [&] ( auto &  tb )
        {
            tb.run( [&] ()
            {
                //
                // construct row bases for all block rows constructed on this level
                //
                    
                auto  rowiss = std::deque< indexset >();
                    
                for ( auto  [ is, matrices ] : rowmap )
                    rowiss.push_back( is );

                for_each( par, rowiss.begin(), rowiss.end(), 
                          [&] ( auto  is )
                          {
                              auto  matrices = rowmap.at( is );
                    
                              if ( matrices.size() == 0 )
                                  return;

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
                          } );
            } );

            tb.run( [&] ()
            {
                //
                // construct column bases for all block columns constructed on this level
                //
                    
                auto  coliss = std::deque< indexset >();
                    
                for ( auto  [ is, matrices ] : colmap )
                    coliss.push_back( is );

                for_each( par, coliss.begin(), coliss.end(), 
                          [&] ( auto  is )
                          {
                              auto  matrices = colmap.at( is );

                              if ( matrices.size() == 0 )
                                  return;

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
                          } );
            } );
        } );

        //
        // now convert all blocks on this level
        //

        for_each( par, lrmat.begin(), lrmat.end(),
                  [&] ( auto  R )
                  {
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
                  } );
    }// while
    
    return { std::move( rowcb_root ),
             std::move( colcb_root ),
             std::move( M_root ) };
}

//
// recursively build uniform H-matrix while also constructing row/column cluster basis
// by updating bases after constructing low-rank blocks
//
using  matrix_list_t = std::vector< hpro::TMatrix * >;
using  matrix_map_t  = std::unordered_map< indexset, matrix_list_t, indexset_hash >;
using  mutex_map_t   = std::unordered_map< indexset, std::mutex, indexset_hash >;

struct rec_basis_data_t
{
    // maps indexsets to set of uniform matrices sharing corresponding cluster basis
    // and their mutexes
    matrix_map_t   rowmap, colmap;
    std::mutex     rowmapmtx, colmapmtx;

    // mutexes for the lists within rowmap/colmap
    // mutex_map_t    rowmtxs, colmtxs;

    //
    // extend row basis <cb> by block W·T·X' (X is not needed for computation)
    //
    // - identical to implementation in "arith/detail/uniform.hh" but thread-safe,
    //   hence, for details look into original code
    //
    template < typename value_t,
               typename basis_approx_t >
    blas::matrix< value_t >
    compute_extended_row_basis ( const cluster_basis< value_t > &  cb,
                                 const blas::matrix< value_t > &   W,
                                 const blas::matrix< value_t > &   T,
                                 const hpro::TTruncAcc &           acc,
                                 const basis_approx_t &            basisapx )
    {
        using  real_t = hpro::real_type_t< value_t >;

        // zero basis implies empty matrix list
        if ( cb.basis().ncols() == 0 )
            return std::move( blas::copy( W ) );
            
        //
        // collect all scaled coupling matrices
        //
        
        auto    couplings = std::list< blas::matrix< value_t > >();
        size_t  nrows_S   = T.ncols();
        auto    uni_mats  = matrix_list_t();
        auto    mat_ids   = std::vector< int >();

        {
            auto  lock = std::scoped_lock( rowmapmtx );

            HLR_ASSERT( rowmap.find( cb.is() ) != rowmap.end() );
            
            for ( auto  M_i : rowmap.at( cb.is() ) )
            {
                uni_mats.push_back( M_i );
                mat_ids.push_back( M_i->id() );
            }// for
        }

        for ( auto  M_i : uni_mats )
        {
            const auto  R_i = cptrcast( M_i, uniform_lrmatrix< value_t > );
            auto        S_i = blas::matrix< value_t >();
                        
            {
                auto  lock_i = std::scoped_lock( M_i->mutex() );

                S_i = std::move( blas::copy( R_i->coeff() ) );
            }
                        
            HLR_ASSERT( S_i.nrows() == cb.basis().ncols() );
            
            auto  norm = norm::spectral( S_i );
                        
            if ( norm != real_t(0) )
            {
                blas::scale( value_t(1) / norm, S_i );
                nrows_S += S_i.ncols();
                couplings.push_back( std::move( S_i ) );
            }// if
        }// for

        //
        // assemble all scaled coupling matrices into joined matrix
        //

        auto    U   = cb.basis();
        auto    Ue  = blas::join_row< value_t >( { U, W } );
        auto    S   = blas::matrix< value_t >( nrows_S, Ue.ncols() );
        size_t  pos = 0;

        int  id_i = 0;
            
        for ( auto  S_i : couplings )
        {
            HLR_ASSERT( pos + S_i.ncols() <= S.nrows() );
            HLR_ASSERT( S_i.nrows() == U.ncols() );
            
            auto  S_sub = blas::matrix< value_t >( S,
                                                   blas::range( pos, pos + S_i.ncols()-1 ),
                                                   blas::range( 0, U.ncols() - 1 ) );
                        
            blas::copy( blas::adjoint( S_i ), S_sub );
            pos += S_i.ncols();

            id_i++;
        }// for

        //
        // add part from W·T·X'
        //
        
        auto  S_i  = blas::copy( T );
        auto  norm = norm::spectral( T );
            
        if ( norm != real_t(0) )
            blas::scale( value_t(1) / norm, S_i );
            
        HLR_ASSERT( pos + S_i.ncols() <= S.nrows() );
        HLR_ASSERT( S_i.nrows() == Ue.ncols() - U.ncols() );
        
        auto  S_sub = blas::matrix< value_t >( S,
                                               blas::range( pos, pos + T.ncols()-1 ),
                                               blas::range( U.ncols(), Ue.ncols() - 1 ) );
            
        blas::copy( blas::adjoint( S_i ), S_sub );
        
        //
        // form product Ue·S and compute column basis
        //
            
        auto  R = blas::matrix< value_t >();
        
        blas::qr( S, R, false );

        auto  UeR = blas::prod( Ue, blas::adjoint( R ) );
        auto  Un  = basisapx.column_basis( UeR, acc );

        return  Un;
    }

    //
    // extend column basis <cb> by block W·T·X' (W is not needed for computation)
    //
    // - identical to implementation in "arith/detail/uniform.hh" but thread-safe,
    //   hence, for details look into original code
    //
    template < typename value_t,
               typename approx_t >
    blas::matrix< value_t >
    compute_extended_col_basis ( const cluster_basis< value_t > &  cb,
                                 const blas::matrix< value_t > &   X,
                                 const blas::matrix< value_t > &   T,
                                 const hpro::TTruncAcc &           acc,
                                 const approx_t &                  approx )
    {
        using  real_t = hpro::real_type_t< value_t >;

        // non-zero matrix implies non-empty matrix list
        if ( cb.basis().ncols() == 0 )
            return std::move( blas::copy( X ) );
            
        //
        // collect all scaled coupling matrices
        //
        
        auto    couplings = std::list< blas::matrix< value_t > >();
        size_t  nrows_S   = T.nrows();
        auto    uni_mats  = matrix_list_t();
        auto    mat_ids   = std::vector< int >();

        {
            auto  lock = std::scoped_lock( colmapmtx );
                    
            HLR_ASSERT( colmap.find( cb.is() ) != colmap.end() );
            
            for ( auto  M_i : colmap.at( cb.is() ) )
            {
                uni_mats.push_back( M_i );
                mat_ids.push_back( M_i->id() );
            }// for
        }

        for ( auto  M_i : uni_mats )
        {
            const auto  R_i = cptrcast( M_i, uniform_lrmatrix< value_t > );
            auto        S_i = blas::matrix< value_t >();

            {
                auto  lock_i = std::scoped_lock( M_i->mutex() );
                
                S_i = std::move( blas::copy( R_i->coeff() ) );
            }
                        
            HLR_ASSERT( S_i.ncols() == cb.basis().ncols() );
            
            auto  norm = norm::spectral( S_i );

            if ( norm != real_t(0) )
            {
                blas::scale( value_t(1) / norm, S_i );
                nrows_S += S_i.nrows();
                couplings.push_back( std::move( S_i ) );
            }// if
        }// for
        
        //
        // assemble all scaled coupling matrices into joined matrix
        //
        
        auto    V   = cb.basis();
        auto    Ve  = blas::join_row< value_t >( { V, X } );
        auto    S   = blas::matrix< value_t >( nrows_S, Ve.ncols() );
        size_t  pos = 0;

        int  id_i = 0;
            
        for ( auto  S_i : couplings )
        {
            HLR_ASSERT( pos + S_i.nrows() <= S.nrows() );
            HLR_ASSERT( S_i.ncols() == V.ncols() );
            
            auto  S_sub = blas::matrix< value_t >( S,
                                                   blas::range( pos, pos + S_i.nrows()-1 ),
                                                   blas::range( 0, V.ncols() - 1 ) );

            blas::copy( S_i, S_sub );
            pos += S_i.nrows();

            id_i++;
        }// for

        //
        // add part from W·T·X'
        //
        
        auto  S_i  = blas::copy( T );
        auto  norm = norm::spectral( T );
            
        if ( norm != real_t(0) )
            blas::scale( value_t(1) / norm, S_i );
            
        HLR_ASSERT( pos + S_i.nrows() <= S.nrows() );
        HLR_ASSERT( S_i.ncols() == Ve.ncols() - V.ncols() );
        
        auto  S_sub = blas::matrix< value_t >( S,
                                               blas::range( pos, pos + T.nrows()-1 ),
                                               blas::range( V.ncols(), Ve.ncols() - 1 ) );
            
        blas::copy( S_i, S_sub );
        
        //
        // form product Ve·S' and compute column basis
        //
            
        auto  R = blas::matrix< value_t >();

        blas::qr( S, R, false );

        auto  VeR = blas::prod( Ve, blas::adjoint( R ) );
        auto  Vn  = approx.column_basis( VeR, acc );
        
        return  Vn;
    }

    //
    // update coupling matrices for all blocks sharing row basis <cb> to new basis <Un>
    //
    template < typename value_t >
    void
    update_row_coupling ( const cluster_basis< value_t > &  cb,
                          const blas::matrix< value_t > &   Un )
    {
        if ( cb.basis().ncols() == 0 )
            return;
            
        // auto  lock_is  = std::scoped_lock( rowmtxs[ cb.is() ] );
        auto  uni_mats = matrix_list_t();

        {
            auto  lock = std::scoped_lock( rowmapmtx );
                    
            HLR_ASSERT( rowmap.find( cb.is() ) != rowmap.end() );
            
            for ( auto  M_i : rowmap.at( cb.is() ) )
                uni_mats.push_back( M_i );
        }
        
        auto  U  = cb.basis();
        auto  TU = blas::prod( blas::adjoint( Un ), U );

        for ( auto  M_i : uni_mats )
        {
            auto  lock_i = std::scoped_lock( M_i->mutex() );
            auto  R_i    = ptrcast( M_i, uniform_lrmatrix< value_t > );
            auto  S_i    = blas::prod( TU, R_i->coeff() );

            R_i->set_coeff_unsafe( std::move( S_i ) );
        }// for
    }

    //
    // update coupling matrices for all blocks sharing column basis <cb> to new basis <Vn>
    //
    template < typename value_t >
    void
    update_col_coupling ( const cluster_basis< value_t > &  cb,
                          const blas::matrix< value_t > &   Vn )
    {
        if ( cb.basis().ncols() == 0 )
            return;
        
        // auto  lock_is  = std::scoped_lock( colmtxs[ cb.is() ] );
        auto  uni_mats = matrix_list_t();

        {
            auto  lock = std::scoped_lock( colmapmtx );
                    
            HLR_ASSERT( colmap.find( cb.is() ) != colmap.end() );

            for ( auto  M_i : colmap.at( cb.is() ) )
                uni_mats.push_back( M_i );
        }
            
        auto  V  = cb.basis();
        auto  TV = blas::prod( blas::adjoint( Vn ), V );

        for ( auto  M_i : uni_mats )
        {
            auto  lock_i = std::scoped_lock( M_i->mutex() );
            auto  R_i    = ptrcast( M_i, uniform_lrmatrix< value_t > );
            auto  S_i    = blas::prod( R_i->coeff(), blas::adjoint( TV ) );
                
            R_i->set_coeff_unsafe( std::move( S_i ) );
        }// for
    }
};

template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::unique_ptr< hpro::TMatrix >
build_uniform_rec ( const hpro::TBlockCluster *                   bct,
                    const coeff_t &                               coeff,
                    const lrapx_t &                               lrapx,
                    const basisapx_t &                            basisapx,
                    const hpro::TTruncAcc &                       acc,
                    cluster_basis< typename coeff_t::value_t > &  rowcb,
                    cluster_basis< typename coeff_t::value_t > &  colcb,
                    rec_basis_data_t &                            basis_data )
{
    using value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto  M = std::unique_ptr< hpro::TMatrix >();
    
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< hpro::TMatrix >( lrapx.build( bct, acc ) );

            if ( is_lowrank( *M ) )
            {
                //
                // compute LRS representation W·T·X' = U·V' = M
                //

                auto  R  = ptrcast( M.get(), hpro::TRkMatrix );
                auto  W  = std::move( blas::mat_U< value_t >( R ) ); // reuse storage from R
                auto  X  = std::move( blas::mat_V< value_t >( R ) );
                auto  Rw = blas::matrix< value_t >();
                auto  Rx = blas::matrix< value_t >();

                define_task_block( [&] ( auto &  tb )
                {
                    tb.run( [&] () { blas::qr( W, Rw ); } );
                    tb.run( [&] () { blas::qr( X, Rx ); } );
                } );
                
                HLR_ASSERT( Rw.ncols() != 0 );
                HLR_ASSERT( Rx.ncols() != 0 );
                
                auto  T       = blas::prod( Rw, blas::adjoint( Rx ) );
                auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
                
                // define_task_block( [&] ( auto &  tb )
                {
                    // tb.run( [&] ()
                    {
                        auto  Un = basis_data.compute_extended_row_basis( rowcb, W, T, acc, basisapx );
                        
                        basis_data.update_row_coupling( rowcb, Un );
                        rowcb.set_basis( std::move( Un ) );
                    } // );
                    
                    // tb.run( [&] ()
                    {
                        auto  Vn = basis_data.compute_extended_col_basis( colcb, X, T, acc, basisapx );
                        
                        basis_data.update_col_coupling( colcb, Vn );
                        colcb.set_basis( std::move( Vn ) );
                    } // );
                } // );

                //
                // transform T into new bases
                //

                auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), W );
                auto  TV = blas::prod( blas::adjoint( colcb.basis() ), X );
                auto  TS = blas::prod( TU, T );
                auto  S  = blas::prod( TS, blas::adjoint( TV ) );

                M = std::make_unique< uniform_lrmatrix< value_t > >( M->row_is(), M->col_is(), rowcb, colcb, std::move( S ) );

                {
                    auto  lock_is = std::scoped_lock( basis_data.rowmapmtx, basis_data.colmapmtx );

                    basis_data.rowmap[ rowcb.is() ].push_back( M.get() );
                    basis_data.colmap[ colcb.is() ].push_back( M.get() );
                }
            }// if
        }// if
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
        B->set_block_struct( bct->nrows(), bct->ncols() );

        define_task_block( [&] ( auto &  tb )
        {
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( ! is_null( bct->son( i, j ) ) )
                    {
                        tb.run( [&,B,i,j] ()
                        {
                            auto  rowcb_i = rowcb.son( i );
                            auto  colcb_j = colcb.son( j );
                            
                            HLR_ASSERT( ! is_null_any( rowcb_i, colcb_j ) );
                            
                            auto  B_ij = build_uniform_rec( bct->son( i, j ), coeff, lrapx, basisapx, acc, *rowcb_i, *colcb_j, basis_data );
                            
                            B->set_block( i, j, B_ij.release() );
                        } );
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

template < typename basisapx_t >
std::unique_ptr< hpro::TMatrix >
build_uniform_rec ( const hpro::TMatrix &                            A,
                    const basisapx_t &                               basisapx,
                    const hpro::TTruncAcc &                          acc,
                    cluster_basis< typename basisapx_t::value_t > &  rowcb,
                    cluster_basis< typename basisapx_t::value_t > &  colcb,
                    rec_basis_data_t &                               basis_data )
{
    using value_t = typename basisapx_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    auto  M = std::unique_ptr< hpro::TMatrix >();
    
    if ( is_lowrank( A ) )
    {
        //
        // compute LRS representation W·T·X' = U·V' = M
        //

        auto  R  = cptrcast( &A, hpro::TRkMatrix );
        auto  W  = blas::copy( blas::mat_U< value_t >( R ) );
        auto  X  = blas::copy( blas::mat_V< value_t >( R ) );
        auto  Rw = blas::matrix< value_t >();
        auto  Rx = blas::matrix< value_t >();

        define_task_block( [&] ( auto &  tb )
        {
            tb.run( [&] () { blas::qr( W, Rw ); } );
            tb.run( [&] () { blas::qr( X, Rx ); } );
        } );

        HLR_ASSERT( Rw.ncols() != 0 );
        HLR_ASSERT( Rx.ncols() != 0 );
                
        auto  T       = blas::prod( Rw, blas::adjoint( Rx ) );
        auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
        
        // define_task_block( [&] ( auto &  tb )
        {
            // tb.run( [&] ()
            {
                auto  Un = basis_data.compute_extended_row_basis( rowcb, W, T, acc, basisapx );
                
                basis_data.update_row_coupling( rowcb, Un );
                rowcb.set_basis( std::move( Un ) );
            } // );
            
            // tb.run( [&] ()
            {
                auto  Vn = basis_data.compute_extended_col_basis( colcb, X, T, acc, basisapx );
                
                basis_data.update_col_coupling( colcb, Vn );
                colcb.set_basis( std::move( Vn ) );
            } // );
        } // );

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
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        
        M = std::make_unique< hpro::TBlockMatrix >();

        auto  BM = ptrcast( M.get(), hpro::TBlockMatrix );

        BM->copy_struct_from( BA );

        define_task_block( [&,BA,BM] ( auto &  tb )
        {
            for ( uint  i = 0; i < BA->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < BA->nblock_cols(); ++j )
                {
                    if ( ! is_null( BA->block( i, j ) ) )
                    {
                        tb.run( [&,BA,BM,i,j] ()
                        {
                            auto  rowcb_i = rowcb.son( i );
                            auto  colcb_j = colcb.son( j );
                            
                            HLR_ASSERT( ! is_null_any( rowcb_i, colcb_j ) );
                            
                            auto  B_ij = build_uniform_rec( *BA->block( i, j ), basisapx, acc, *rowcb_i, *colcb_j, basis_data );

                            BM->set_block( i, j, B_ij.release() );
                        } );
                    }// if
                }// for
            }// for
        } );

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
init_cluster_bases ( const hpro::TBlockCluster *  bct,
                     cluster_basis< value_t > &   rowcb,
                     cluster_basis< value_t > &   colcb )
{
    //
    // decide upon cluster type, how to construct matrix
    //

    auto  M = std::unique_ptr< hpro::TMatrix >();
    
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
init_cluster_bases ( const hpro::TMatrix &       M,
                     cluster_basis< value_t > &  rowcb,
                     cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix );
        
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
                            rowcb_i->set_nsons( cptrcast( M_ij, hpro::TBlockMatrix )->nblock_rows() );
                        
                        if ( is_null( colcb_j ) )
                        {
                            colcb_j = new cluster_basis< value_t >( M_ij->col_is() );
                            colcb.set_son( j, colcb_j );
                        }// if
            
                        if ( is_blocked( M_ij ) && ( colcb_j->nsons() == 0 ))
                            colcb_j->set_nsons( cptrcast( M_ij, hpro::TBlockMatrix )->nblock_cols() );
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

}}}}// namespace hlr::hpx::matrix::detail

#endif // __HLR_HPX_DETAIL_MATRIX_HH

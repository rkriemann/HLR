#ifndef __HLR_HPX_DETAIL_UNIFORM_ACCU_HH
#define __HLR_HPX_DETAIL_UNIFORM_ACCU_HH
//
// Project     : HLR
// Module      : arith/uniform_accu
// Description : arithmetic functions for uniform matrices using accumulators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <algorithm>

#include <hpx/parallel/task_block.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>

#include <boost/format.hpp>

#include <hpro/base/System.hh>

#include <hlr/arith/multiply.hh>
#include <hlr/arith/blas.hh>
#include <hlr/arith/solve.hh>
#include <hlr/arith/detail/uniform.hh>
#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/lrsmatrix.hh>
#include <hlr/matrix/restrict.hh>
#include <hlr/utils/hash.hh>
#include <hlr/utils/tensor.hh>

namespace hlr { namespace hpx { namespace uniform { namespace accu { namespace detail {

using ::hpx::parallel::v2::define_task_block;
using ::hpx::for_each;
using ::hpx::execution::par;

using  hlr::matrix::cluster_basis;
using  hlr::matrix::is_uniform_lowrank;
using  hlr::matrix::is_uniform_lowrank_all;
using  hlr::matrix::uniform_lrmatrix;

// maps index set to set of blocks sharing it
using  uniform_map_t = std::unordered_map< indexset, std::list< hpro::TMatrix * >, indexset_hash >;

// maps index set to product U×V' of inner matrix product
using  inner_map_t   = std::unordered_map< indexset, blas::matrix< hpro::real >, indexset_hash >;

struct accumulator
{
    //
    // represents an update, i.e., matrix product
    //
    struct update
    {
        const matop_t          op_A;
        const hpro::TMatrix *  A;
        const matop_t          op_B;
        const hpro::TMatrix *  B;
    };
    
    // represents set of updates
    using  update_list = std::list< update >;

    // accumulated computed updates
    std::unique_ptr< hpro::TMatrix >   matrix;

    // accumulated pending (recursive) updates
    update_list                        pending;

    // cached products
    inner_map_t *                      prod_inner;
    std::mutex *                       prod_inner_mtx;
    
    //
    // ctors
    //

    accumulator ( inner_map_t *                        aprod_inner     = nullptr,
                  std::mutex *                         aprod_inner_mtx = nullptr )
            : prod_inner( aprod_inner )
            , prod_inner_mtx( aprod_inner_mtx )
    {
        if ( ! is_null( prod_inner ) )
            HLR_ASSERT( ! is_null( prod_inner_mtx ) );
    }
    
    accumulator ( std::unique_ptr< hpro::TMatrix > &&  amatrix,
                  update_list &&                       apending,
                  inner_map_t *                        aprod_inner     = nullptr,
                  std::mutex *                         aprod_inner_mtx = nullptr )
            : matrix( std::move( amatrix ) )
            , pending( std::move( apending ) )
            , prod_inner( aprod_inner )
            , prod_inner_mtx( aprod_inner_mtx )
    {
        if ( ! is_null( prod_inner ) )
            HLR_ASSERT( ! is_null( prod_inner_mtx ) );
    }
    
    //
    // remove update matrix
    //
    void
    clear_matrix ()
    {
        matrix.reset( nullptr );
    }

    //
    // release matrix
    //
    hpro::TMatrix *
    release_matrix ()
    {
        return matrix.release();
    }

    //
    // add update A×B
    //
    void
    add_update ( const hpro::TMatrix &  A,
                 const hpro::TMatrix &  B )
    {
        pending.push_back( { apply_normal, &A, apply_normal, &B } );
    }

    void
    add_update ( const matop_t          op_A,
                 const hpro::TMatrix &  A,
                 const matop_t          op_B,
                 const hpro::TMatrix &  B )
    {
        pending.push_back( { op_A, &A, op_B, &B } );
    }
    
    //
    // apply accumulated updates and free accumulator matrix
    //
    template < typename value_t,
               typename approx_t >
    void
    apply ( const value_t            alpha,
            hpro::TMatrix &          M,
            const hpro::TTruncAcc &  acc,
            const approx_t &         approx )
    {
        if ( ! is_null( matrix ) )
            hlr::add( alpha, *matrix, M, acc, approx );

        clear_matrix();
    }
    
    //
    // return restriction of updates to block (i,j) of given block matrix
    //
    template < typename value_t >
    accumulator
    restrict ( const uint                  i,
               const uint                  j,
               const hpro::TBlockMatrix &  M ) const
    {
        auto  U_ij = std::unique_ptr< hpro::TMatrix >();
        auto  P_ij = update_list();
        
        if ( ! is_null( matrix ) )
        {
            HLR_ASSERT( ! is_null( M.block( i, j ) ) );
            
            U_ij = hlr::matrix::restrict( *matrix, M.block( i, j )->block_is() );

            // prefer dense if destination is dense
            if ( check_dense( *M.block( i, j ) ) && ! is_dense( *U_ij ) )
                U_ij = std::move( hlr::matrix::convert_to_dense< value_t >( *U_ij ) );
        }// if

        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            // filter out non-recursive updates
            if ( ! is_blocked_all( A, B ) )
                continue;
                                            
            auto  BA = cptrcast( A, hpro::TBlockMatrix );
            auto  BB = cptrcast( B, hpro::TBlockMatrix );
                                            
            for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
            {
                auto  A_il = BA->block( i, l, op_A );
                auto  B_lj = BB->block( l, j, op_B );
                                                
                if ( is_null_any( A_il, B_lj ) )
                    continue;
                                                
                P_ij.push_back( { op_A, A_il, op_B, B_lj } );
            }// for
        }// for

        return accumulator{ std::move( U_ij ), std::move( P_ij ), prod_inner, prod_inner_mtx };
    }

    //
    // return restriction of updates to all sub blocks of given block matrix
    //
    template < typename value_t >
    tensor2< accumulator >
    restrict ( const hpro::TBlockMatrix &  M ) const
    {
        tensor2< accumulator >  sub_accu( M.nblock_rows(), M.nblock_cols() );
        
        for ( uint  i = 0; i < M.nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < M.nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( M.block( i, j ) ) );

                sub_accu(i,j) = restrict< value_t >( i, j, M );
            }// for
        }// for

        return sub_accu;
    }
    
    //
    // evaluate all computable updates to matrix M
    //
    template < typename value_t,
               typename approx_t >
    void
    eval ( const value_t                    alpha,
           const hpro::TMatrix &            M,
           const hpro::TTruncAcc &          acc,
           const approx_t &                 approx )
    {
        //
        // first check for dense handling
        //

        bool  handle_dense = check_dense( M );

        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            if ( is_dense_all( A, B ) ||
                 ( is_blocked( A ) && is_dense(   B ) ) ||
                 ( is_dense(   A ) && is_blocked( B ) ))
            {
                handle_dense = true;
                break;
            }// if
        }// for
        
        if ( ! handle_dense )
        {
            //
            // filter out different variants of uniform factors
            //
            
            update_list  pending_uniAB;
            update_list  pending_uniA;
            update_list  pending_uniB;
        
            for ( auto  it = pending.begin(); it != pending.end(); )
            {
                if ( is_uniform_lowrank_all( (*it).A, (*it).B ) )
                {
                    pending_uniAB.push_back( *it );
                    it = pending.erase( it );
                }// if
                else
                    ++it;
            }// for
            
            for ( auto  it = pending.begin(); it != pending.end(); )
            {
                if ( is_uniform_lowrank( (*it).A ) )
                {
                    pending_uniA.push_back( *it );
                    it = pending.erase( it );
                }// if
                else
                    ++it;
            }// for

            for ( auto  it = pending.begin(); it != pending.end(); )
            {
                if ( is_uniform_lowrank( (*it).B ) )
                {
                    pending_uniB.push_back( *it );
                    it = pending.erase( it );
                }// if
                else
                    ++it;
            }// for

            //
            // initialize matrices
            // - for definition of matrices see sections below
            //
            
            auto  U = blas::matrix< value_t >();
            auto  V = blas::matrix< value_t >();
            auto  R = blas::matrix< value_t >();
            auto  Z = blas::matrix< value_t >();
            auto  Y = blas::matrix< value_t >();

            if ( ! pending_uniAB.empty() )
            {
                auto  [ op_A, A, op_B, B ] = pending_uniAB.front();
                
                U = cptrcast( A, uniform_lrmatrix< value_t > )->row_basis( op_A );
                V = cptrcast( B, uniform_lrmatrix< value_t > )->col_basis( op_B );
            }// if
            
            if ( ! pending_uniA.empty() )
            {
                auto  [ op_A, A, op_B, B ] = pending_uniA.front();
                
                if ( U.ncols() == 0 )
                    U = cptrcast( A, uniform_lrmatrix< value_t > )->row_basis( op_A );
                
                Z = std::move( blas::matrix< value_t >( M.ncols(), U.ncols() ) );
            }// if

            if ( ! pending_uniB.empty() )
            {
                auto  [ op_A, A, op_B, B ] = pending_uniB.front();

                if ( V.ncols() == 0 )
                    V = cptrcast( B, hlr::matrix::uniform_lrmatrix< value_t > )->col_basis( op_B );

                Y = std::move( blas::matrix< value_t >( M.nrows(), V.ncols() ) );
            }// if

            //
            // handle different uniform product variants in parallel
            //

            define_task_block( [&] ( auto &  tb )
            {
                tb.run( [&,this] ()
                {
                    if ( ! pending_uniAB.empty() )
                    {
                        std::vector< update >  vpending;

                        vpending.reserve( pending_uniAB.size() );
                        std::copy( std::begin( pending_uniAB ), std::end( pending_uniAB ), std::back_inserter( vpending ) );
                        R = std::move( sum_uni_uni< value_t >( 0, vpending.size(), vpending ) );
                    }// if
                } );

                tb.run( [&,this] ()
                {
                    if ( ! pending_uniA.empty() )
                    {
                        std::vector< update >  vpending;

                        vpending.reserve( pending_uniA.size() );
                        std::copy( std::begin( pending_uniA ),  std::end( pending_uniA ),  std::back_inserter( vpending ) );
                        Z = std::move( sum_uni_any< value_t >( 0, vpending.size(), vpending ) );
                    }// if
                } );

                tb.run( [&,this] ()
                {
                    if ( ! pending_uniB.empty() )
                    {
                        std::vector< update >  vpending;

                        vpending.reserve( pending_uniB.size() );
                        std::copy( std::begin( pending_uniB ),  std::end( pending_uniB ),  std::back_inserter( vpending ) );
                        Y = std::move( sum_any_uni< value_t >( 0, vpending.size(), vpending ) );
                    }// if
                } );
            } );

            //
            // sum up individual updates
            //

            if ( R.ncols() > 0 )
            {
                if ( Z.ncols() > 0 )
                {
                    // Z = Z + V·R'
                    blas::prod( value_t(1), V, blas::adjoint( R ), value_t(1), Z );
                }// if
                else if ( Y.ncols() > 0 )
                {
                    // Y = Y + U·R
                    blas::prod( value_t(1), U, R, value_t(1), Y );
                }// if
                else
                {
                    //
                    // just update with U·R·V'
                    //
                    if ( is_null( matrix ) )
                        matrix = std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(),
                                                                      std::move( blas::prod( alpha, U, R ) ),
                                                                      std::move( blas::copy( V ) ) );
                    else
                    {
                        auto  US = blas::prod( U, R );
                        auto  T  = hpro::TRkMatrix( M.row_is(), M.col_is(), US, V );
                
                        hlr::add( alpha, T, *matrix, acc, approx );
                    }// else
                }// else
            }// if

            if ( Z.ncols() > 0 )
            {
                if ( is_null( matrix ) )
                {
                    blas::scale( alpha, Z );
                    matrix = std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( blas::copy( U ) ), std::move( Z ) );
                }// if
                else
                {
                    auto  T = hpro::TRkMatrix( M.row_is(), M.col_is(), U, Z );
                    
                    hlr::add( alpha, T, *matrix, acc, approx );
                }// else
            }// if
            
            if ( Y.ncols() > 0 )
            {
                if ( is_null( matrix ) )
                {
                    blas::scale( alpha, Y );
                    matrix = std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( Y ), std::move( blas::copy( V ) ) );
                }// if
                else
                {
                    auto  T = hpro::TRkMatrix( M.row_is(), M.col_is(), Y, V );
                    
                    hlr::add( alpha, T, *matrix, acc, approx );
                }// else
            }// if
        }// if

        //
        // handle remaining computable updates, i.e., one factor is a leaf block
        //
        
        update_list  pending_comp;
        
        for ( auto  it = pending.begin(); it != pending.end(); )
        {
            if ( ! is_blocked_all( (*it).A, (*it).B ) )
            {
                pending_comp.push_back( *it );
                it = pending.erase( it );
            }// if
            else
                ++it;
        }// for
        
        if ( ! pending_comp.empty() )
        {
            std::vector< update >  vpending;

            vpending.reserve( pending_comp.size() );
            std::copy( std::begin( pending_comp ), std::end( pending_comp ), std::back_inserter( vpending ) );

            auto  T = std::move( sum_pending< value_t >( 0, vpending.size(), vpending, handle_dense, acc, approx ) );

            if ( is_null( matrix ) )
            {
                matrix = std::move( T );
            }// if
            else if ( ! is_dense( *matrix ) && is_dense( *T ) )
            {
                // prefer dense format to avoid unnecessary truncations
                hlr::add( value_t(1), *matrix, *T, acc, approx );
                matrix = std::move( T );
            }// if
            else
            {
                hlr::add( value_t(1), *T, *matrix, acc, approx );
            }// else
        }// if
        
        //
        // test if leaf = blocked × blocked is in remaining updates and set up
        // temporary block matrix for recursion
        //

        auto  BC = std::unique_ptr< hpro::TBlockMatrix >(); // for recursive handling

        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            if ( is_blocked_all( *A, *B, M ) )
                continue;
        
            if ( is_blocked_all( A, B ) )
            {
                //
                // if M is a leaf and A _and_ B are blocked, a temporary matrix
                // is created for further recursive update handling
                //

                if ( ! is_null( BC ) )
                    continue;
                
                // TODO: non low-rank M
                if ( ! ( is_lowrank( M ) || hlr::matrix::is_lowrankS( M ) || is_uniform_lowrank( M ) ) )
                    HLR_ERROR( "unsupported matrix type: " + M.typestr() );
                
                auto  BA = cptrcast( A, hpro::TBlockMatrix );
                auto  BB = cptrcast( B, hpro::TBlockMatrix );
                
                BC = std::make_unique< hpro::TBlockMatrix >( A->row_is( op_A ), B->col_is( op_B ) );

                BC->set_block_struct( BA->nblock_rows( op_A ), BB->nblock_cols( op_B ) );

                for ( uint  i = 0; i < BC->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                    {
                        HLR_ASSERT( ! is_null_any( BA->block( i, 0, op_A ), BB->block( 0, j, op_B ) ) );
                        
                        if ( handle_dense )
                            BC->set_block( i, j, new hpro::TDenseMatrix( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                         BB->block( 0, j, op_B )->col_is( op_B ),
                                                                         hpro::value_type_v< value_t > ) );
                        else
                            BC->set_block( i, j, new hpro::TRkMatrix( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                      BB->block( 0, j, op_B )->col_is( op_B ),
                                                                      hpro::value_type_v< value_t > ) );
                    }// for
                }// for

                break;
            }// if
        }// for
        
        //
        // now handle recursive updates if M is a leaf block
        //
    
        if ( ! is_null( BC ) )
        {
            //
            // first, split update matrix into subblock updates
            // (to release matrix before recursion)
            //

            auto  sub_accu = restrict< value_t >( *BC );

            matrix.reset( nullptr );
        
            //
            // apply recursive updates
            //

            define_task_block( [&,alpha] ( auto &  tb )
            {
                for ( uint  i = 0; i < BC->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                    {
                        tb.run( [&,alpha,i,j] ()
                        {
                            sub_accu(i,j).eval( alpha, *BC->block(i,j), acc, approx );

                            // replace block in BC by accumulator matrix for agglomeration below
                            BC->delete_block( i, j );
                            BC->set_block( i, j, sub_accu(i,j).release_matrix() );
                        } );
                    }// for
                }// for
            } );

            //
            // finally convert subblocks to single low-rank matrix for new accumulated updates
            //

            if ( handle_dense )
                matrix = seq::matrix::convert_to_dense< value_t >( *BC );
            else
                matrix = seq::matrix::convert_to_lowrank( *BC, acc, approx );
        }// if
    }

    //
    // compute sum of uniform × uniform updates
    //
    template < typename value_t >
    blas::matrix< value_t >
    sum_uni_uni ( const int                lb,
                  const int                ub,
                  std::vector< update > &  updates )
    {
        if ( ub - lb <= 1 )
        {
            //
            // for uniform x uniform only coefficients need to be added without
            // truncation due to shared bases:
            //
            //   U ( Σ_i S_i X_i' W_i T_i ) V' = U R V'  with R = Σ_i S_i X_i' W_i T_i
            //

            auto  [ op_A, A, op_B, B ] = updates[ lb ];

            auto  RA = cptrcast( A, uniform_lrmatrix< value_t > );
            auto  RB = cptrcast( B, uniform_lrmatrix< value_t > );

            auto  S  = RA->coeff();
            auto  X  = RA->col_basis( op_A );
            auto  W  = RB->row_basis( op_B );
            auto  T  = RB->coeff();
            auto  XW = blas::matrix< value_t >();

            if ( ! is_null( prod_inner ) )
            {
                std::scoped_lock  lock( * prod_inner_mtx );
                                                       
                if ( prod_inner->find( RA->col_is( op_A ) ) == prod_inner->end() )
                    prod_inner->emplace( RA->col_is( op_A ), std::move( blas::prod( blas::adjoint( X ), W ) ) );
                                                       
                XW = prod_inner->at( RA->col_is( op_A ) );
            }// if
            else
                XW = std::move( blas::prod( blas::adjoint( X ), W ) );
                                               
            auto  SXW = blas::prod( blas::mat_view( op_A, S ), XW );

            return std::move( blas::prod( SXW, blas::mat_view( op_B, T ) ) );
        }// if
        else
        {
            const int  mid = ( ub + lb ) / 2;
            auto       R1  = blas::matrix< value_t >();
            auto       R2  = blas::matrix< value_t >();

            define_task_block( [&] ( auto &  tb )
            {
                tb.run( [&,lb,mid] () { R1 = sum_uni_uni< value_t >( lb, mid, updates ); } );
                tb.run( [&,mid,ub] () { R2 = sum_uni_uni< value_t >( mid, ub, updates ); } );
            } );
            
            blas::add( value_t(1), R1, R2 );

            return std::move( R2 );
        }// else
    }

    //
    // compute sum of uniform × anything updates
    //
    template < typename value_t >
    blas::matrix< value_t >
    sum_uni_any ( const int                lb,
                  const int                ub,
                  std::vector< update > &  updates )
    {
        if ( ub - lb <= 1 )
        {
            //
            // now handle all uniform x non-uniform
            //
            //   U ( Σ_i S_i X_i' ) × B_i = U ( Σ_i S_i ( X_i' × B_i ) )
            //                            = U ( Σ_i S_i Z_i' )   with Z_i = B_i' × X_i
            //                            = U ( Σ_i (Z_i B_i'))'
            //                            = U Z'                 with Z   = Σ_i (Z_i S_i')
            //
            
            auto  [ op_A, A, op_B, B ] = updates[ lb ];

            auto  RA  = cptrcast( A, hlr::matrix::uniform_lrmatrix< value_t > );
            auto  S   = RA->coeff();
            auto  X   = RA->col_basis( op_A );
            auto  Z_i = std::move( blas::matrix< value_t >( B->ncols(), X.ncols() ) );
                        
            hlr::multiply( value_t(1), blas::adjoint( op_B ), *B, X, Z_i );

            return blas::prod( Z_i, blas::adjoint( S ) );
        }// if
        else
        {
            const int  mid = ( ub + lb ) / 2;
            auto       Z1  = blas::matrix< value_t >();
            auto       Z2  = blas::matrix< value_t >();

            define_task_block( [&] ( auto &  tb )
            {
                tb.run( [&,lb,mid] () { Z1 = sum_uni_any< value_t >( lb, mid, updates ); } );
                tb.run( [&,mid,ub] () { Z2 = sum_uni_any< value_t >( mid, ub, updates ); } );
            } );

            blas::add( value_t(1), Z1, Z2 );

            return std::move( Z2 );
        }// else
    }
    
    //
    // compute sum of anything × uniform updates
    //
    template < typename value_t >
    blas::matrix< value_t >
    sum_any_uni ( const int                lb,
                  const int                ub,
                  std::vector< update > &  updates )
    {
        if ( ub - lb <= 1 )
        {
            //
            // now handle all non-uniform x uniform
            //
            //   A_i × ( Σ_i W_i S_i ) V' = ( Σ_i ( A_i × W_i ) S_i ) V'
            //                            = ( Σ_i Y_i S_i ) V' with Y_i = A_i × W_i
            //                            = ( Σ_i (Y_i S_i) ) V'
            //                            = Y V'
            //

            auto  [ op_A, A, op_B, B ] = updates[ lb ];

            auto  RB  = cptrcast( B, hlr::matrix::uniform_lrmatrix< value_t > );
            auto  W   = RB->row_basis( op_B );
            auto  S   = RB->coeff();
            auto  Y_i = blas::matrix< value_t >( A->nrows(), W.ncols() );
            
            hlr::multiply( value_t(1), op_A, *A, W, Y_i );
            
            return blas::prod( Y_i, S );
        }// if
        else
        {
            const int  mid = ( ub + lb ) / 2;
            auto       Y1  = blas::matrix< value_t >();
            auto       Y2  = blas::matrix< value_t >();

            define_task_block( [&] ( auto &  tb )
            {
                tb.run( [&,lb,mid] () { Y1 = sum_any_uni< value_t >( lb, mid, updates ); } );
                tb.run( [&,mid,ub] () { Y2 = sum_any_uni< value_t >( mid, ub, updates ); } );
            } );

            blas::add( value_t(1), Y1, Y2 );

            return std::move( Y2 );
        }// else
    }
    
    //
    // compute sum of general updates with non-leaf types
    // for at least one factor
    //
    template < typename value_t,
               typename approx_t >
    std::unique_ptr< hpro::TMatrix >
    sum_pending ( const int                lb,
                  const int                ub,
                  std::vector< update > &  updates,
                  const bool               handle_dense,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
    {
        if ( ub - lb <= 1 )
        {
            auto  [ op_A, A, op_B, B ] = updates[ lb ];
            auto  T                    = std::unique_ptr< hpro::TMatrix >();

            if ( handle_dense ||
                 is_dense_all( A, B ) ||
                 ( is_blocked( A ) && is_dense(   B ) ) ||
                 ( is_dense(   A ) && is_blocked( B ) ))
                T = std::make_unique< hpro::TDenseMatrix >( A->row_is( op_A ), B->col_is( op_B ), hpro::value_type_v< value_t > );
            else
            {
                std::cout << "!!! : " << A->typestr() << " x " << B->typestr() << std::endl;
                T = std::make_unique< hpro::TRkMatrix >( A->row_is( op_A ), B->col_is( op_B ), hpro::value_type_v< value_t > );
            }// else

            hlr::multiply< value_t >( value_t(1), op_A, *A, op_B, *B, *T, acc, approx );

            return T;
        }// if
        else
        {
            const int  mid = ( ub + lb ) / 2;
            auto       T1  = std::unique_ptr< hpro::TMatrix >();
            auto       T2  = std::unique_ptr< hpro::TMatrix >();

            define_task_block( [&] ( auto &  tb )
            {
                tb.run( [&,lb,mid,handle_dense] () { T1 = sum_pending< value_t >( lb, mid, updates, handle_dense, acc, approx ); } );
                tb.run( [&,mid,ub,handle_dense] () { T2 = sum_pending< value_t >( mid, ub, updates, handle_dense, acc, approx ); } );
            } );

            if ( is_dense( *T1 ) )
            {
                hlr::add( value_t(1), *T2, *T1 );

                return T1;
            }// if
            else if ( is_dense( *T2 ) )
            {
                hlr::add( value_t(1), *T1, *T2 );

                return T2;
            }// if
            else
            {
                hlr::add( value_t(1), *T2, *T1, acc, approx );

                return T1;
            }// else
        }// else
    }
    
    //
    // return true if given matrix is dense
    // - for structured matrices, only next level is tested
    //
    bool
    check_dense ( const hpro::TMatrix &  M ) const
    {
        // return false;
        if ( is_dense( M ) )
        {
            return true;
        }// if
        else if ( is_blocked( M ) )
        {
            //
            // test if all subblocks are dense
            //

            auto  B = cptrcast( &M, hpro::TBlockMatrix );

            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( ! is_null( B->block( i, j ) ) && ! is_dense( B->block( i, j ) ) )
                         return false;
                }// for
            }// for

            return true;
        }// if
        else
        {
            return false;
        }// else
    }
};

//
// structure for handling basis updates
//

using  matrix_list_t = std::vector< hpro::TMatrix * >;
using  matrix_map_t  = std::unordered_map< indexset, matrix_list_t, indexset_hash >;

struct rec_basis_data_t
{
    // maps indexsets to set of uniform matrices sharing corresponding cluster basis
    matrix_map_t   rowmap, colmap;

    //
    // ctor
    //
    rec_basis_data_t ( hpro::TMatrix &  A )
    {
        auto  blocks = std::list< hpro::TMatrix * >{ &A };

        while ( ! blocks.empty() )
        {
            auto  subblocks = decltype( blocks )();

            for ( auto  M : blocks )
            {
                if ( is_blocked( M ) )
                {
                    auto  BM = ptrcast( M, hpro::TBlockMatrix );

                    for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                        for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                            if ( ! is_null( BM->block( i, j ) ) )
                                subblocks.push_back( BM->block( i, j ) );
                }// if
                else if ( is_uniform_lowrank( M ) )
                {
                    rowmap[ M->row_is() ].push_back( M );
                    colmap[ M->col_is() ].push_back( M );
                }// if
            }// for

            blocks = std::move( subblocks );
        }// while
    }

    //
    // extend row basis <cb> by block W·T·X' (X is not needed for computation)
    //
    // - identical to implementation in "arith/detail/uniform.hh" but thread-safe,
    //   hence, for details look into original code
    //
    template < typename value_t,
               typename basisapx_t >
    blas::matrix< value_t >
    compute_extended_basis ( uniform_lrmatrix< value_t > &     M,
                             const cluster_basis< value_t > &  cb,
                             const blas::matrix< value_t > &   W,
                             const blas::matrix< value_t > &   T,
                             const hpro::TTruncAcc &           acc,
                             const basisapx_t &                basisapx,
                             matrix_map_t &                    matmap,
                             const matop_t                     op )
    {
        using  real_t = hpro::real_type_t< value_t >;

        // zero basis implies empty matrix list
        if ( cb.basis().ncols() == 0 )
            return std::move( blas::copy( W ) );
            
        //
        // collect scaled coupling matrices and filter out zero couplings
        //

        HLR_ASSERT( matmap.find( cb.is() ) != matmap.end() );

        auto    uni_mats  = matmap.at( cb.is() );
        auto    couplings = std::list< blas::matrix< value_t > >();
        size_t  nrows_S   = T.ncols();
        auto    cmtx      = std::mutex();

        for_each( par, uni_mats.begin(), uni_mats.end(),
                  [&] ( auto  M_i )
                  {
                      if ( M_i == &M )
                          return;
            
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
                  } );

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
               typename basisapx_t >
    blas::matrix< value_t >
    compute_extended_row_basis ( uniform_lrmatrix< value_t > &     M,
                                 const cluster_basis< value_t > &  cb,
                                 const blas::matrix< value_t > &   W,
                                 const blas::matrix< value_t > &   T,
                                 const hpro::TTruncAcc &           acc,
                                 const basisapx_t &                basisapx )
    {
        return compute_extended_basis( M, cb, W, T, acc, basisapx, rowmap, apply_adjoint );
    }

    //
    // extend column basis <cb> by block W·T·X' (W is not needed for computation)
    //
    // - identical to implementation in "arith/detail/uniform.hh" but thread-safe,
    //   hence, for details look into original code
    //
    template < typename value_t,
               typename basisapx_t >
    blas::matrix< value_t >
    compute_extended_col_basis ( uniform_lrmatrix< value_t > &     M,
                                 const cluster_basis< value_t > &  cb,
                                 const blas::matrix< value_t > &   X,
                                 const blas::matrix< value_t > &   T,
                                 const hpro::TTruncAcc &           acc,
                                 const basisapx_t &                basisapx )
    {
        return compute_extended_basis( M, cb, X, T, acc, basisapx, colmap, apply_normal );
    }

    //
    // update coupling matrices for all blocks sharing basis <cb> to new basis <Un>
    //
    template < typename value_t >
    void
    update_coupling ( uniform_lrmatrix< value_t > &     M,
                      const cluster_basis< value_t > &  cb,
                      const blas::matrix< value_t > &   Un,
                      matrix_map_t &                    matmap,
                      const bool                        cols )
    {
        if ( cb.basis().ncols() == 0 )
            return;
            
        HLR_ASSERT( matmap.find( cb.is() ) != matmap.end() );
            
        auto  uni_mats = matmap.at( cb.is() );
        auto  U        = cb.basis();
        auto  TU       = blas::prod( blas::adjoint( Un ), U );

        for_each( par, uni_mats.begin(), uni_mats.end(),
                  [&] ( auto  M_i )
                  {
                      if ( M_i == &M )
                          return;
            
                      auto  lock = std::scoped_lock( M_i->mutex() );
                      auto  R_i  = ptrcast( M_i, uniform_lrmatrix< value_t > );
                      auto  S_i  = ( cols
                                     ? blas::prod( R_i->coeff(), blas::adjoint( TU ) )
                                     : blas::prod( TU, R_i->coeff() ) );

                      R_i->set_coeff_unsafe( std::move( S_i ) );
                  } );
    }

    template < typename value_t >
    void
    update_row_coupling ( uniform_lrmatrix< value_t > &     M,
                          const cluster_basis< value_t > &  cb,
                          const blas::matrix< value_t > &   Un )
    {
        update_coupling( M, cb, Un, rowmap, false );
    }
    
    template < typename value_t >
    void
    update_col_coupling ( uniform_lrmatrix< value_t > &     M,
                          const cluster_basis< value_t > &  cb,
                          const blas::matrix< value_t > &   Vn )
    {
        update_coupling( M, cb, Vn, colmap, true );
    }
};

//
// recursive LU factorization
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           hpro::TMatrix &          M,
           accumulator &            accu,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx,
           rec_basis_data_t &       basis_data )
{
    //
    // evaluate all computable updates to M
    //

    accu.eval( value_t(1), M, acc, approx );

    //
    // recurse
    //

    if ( is_blocked( M ) )
    {
        auto  BM       = ptrcast( &M, hpro::TBlockMatrix );
        auto  sub_accu = accu.restrict< value_t >( *BM );

        accu.clear_matrix();

        define_task_block( [&] ( auto &  tb )
        {
            for ( uint  i = 0; i < BM->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                {
                    if ( is_null( BM->block( i, j ) ) )
                        continue;
                    
                    tb.run( [&,alpha,BM,i,j] ()
                    {
                        multiply( alpha, * BM->block( i, j ), sub_accu( i, j ), acc, approx, basis_data );
                    } );
                }// for
            }// for
        } );
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( M ) )
    {
        //
        // update local matrix as standard low-rank matrix
        //

        auto    U = ptrcast( &M, hlr::matrix::uniform_lrmatrix< value_t > );
        auto &  rowcb = U->row_cb();
        auto &  colcb = U->col_cb();
        auto    R = hpro::TRkMatrix( U->row_is(), U->col_is(), hpro::value_type_v< value_t > );

        {
            std::scoped_lock  lock( U->mutex(), rowcb.mutex(), colcb.mutex() );

            R.set_lrmat( std::move( blas::prod( U->row_basis(), U->coeff() ) ),
                         std::move( blas::copy( U->col_basis() ) ) );
        }
        
        // no recursive updates left, apply accumulated updates and solve
        accu.apply( alpha, R, acc, approx );
        
        //
        // now replace M by R and update row/column bases
        //

        auto  W  = std::move( blas::mat_U< value_t >( R ) );
        auto  X  = std::move( blas::mat_V< value_t >( R ) );
        auto  RW = blas::matrix< value_t >();
        auto  RX = blas::matrix< value_t >();

        define_task_block( [&] ( auto &  tb )
        {
            tb.run( [&] () { blas::qr( W, RW ); } );
            tb.run( [&] () { blas::qr( X, RX ); } );
        } );

        auto  T       = blas::prod( RW, blas::adjoint( RX ) );
        auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
                
        define_task_block( [&] ( auto &  tb )
        {
            tb.run( [&] ()
            {
                auto  Un = basis_data.compute_extended_row_basis( *U, rowcb, W, T, acc, approx );
                
                basis_data.update_row_coupling( *U, rowcb, Un );
                rowcb.set_basis( std::move( Un ) );
            } );
                
            tb.run( [&] ()
            {
                auto  Vn = basis_data.compute_extended_col_basis( *U, colcb, X, T, acc, approx );
                
                basis_data.update_col_coupling( *U, colcb, Vn );
                colcb.set_basis( std::move( Vn ) );
            } );
        } );

        //
        // transform M into new bases
        //

        auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), W );
        auto  TV = blas::prod( blas::adjoint( colcb.basis() ), X );
        auto  TS = blas::prod( TU, T );
        auto  S  = blas::prod( TS, blas::adjoint( TV ) );
        
        U->set_coeff( std::move( S ) );
    }// if
    else
    {
        accu.apply( alpha, M, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  accumulator &            accu,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx,
                  const uniform_map_t &    rowmap,
                  const uniform_map_t &    colmap ) //, hpro::TMatrix &          REF )
{
    // std::cout << M.id() << std::endl;
    
    // apply computable updates
    accu.eval( value_t(1), M, acc, approx );
    
    if ( is_blocked_all( L, M ) )
    {
        auto  BL = cptrcast( &L, hpro::TBlockMatrix );
        auto  BM =  ptrcast( &M, hpro::TBlockMatrix );
        // auto  BREF = ptrcast( &REF, hpro::TBlockMatrix );
        
        //
        // first, split accumulated updates U and recursive updates upd_rec
        // into subblock updates
        // - to release U before recursion and by that avoid memory
        //   consumption dependent on hierarchy depth
        //

        auto  sub_accu = accu.restrict< value_t >( *BM );

        accu.clear_matrix();

        if ( side == from_left )
        {
            for ( uint i = 0; i < BM->nblock_rows(); ++i )
            {
                const auto  L_ii = BL->block( i, i );
            
                for ( uint j = 0; j < BM->nblock_cols(); ++j )
                    solve_lower_tri< value_t >( side, diag, *L_ii, *BM->block(i,j),
                                                sub_accu(i,j), acc, approx, rowmap, colmap ); // *BREF->block( i, j ) );

                for ( uint  k = i+1; k < BM->nblock_rows(); ++k )
                    for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                        sub_accu(k,j).add_update( *BL->block(k,i), *BM->block(i,j) );
            }// for
        }// if
        else
        {
            HLR_ASSERT( false );
        }// else
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( M ) )
    {
        //
        // update and solve local matrix
        //

        auto  UM = ptrcast( &M, hlr::matrix::uniform_lrmatrix< value_t > );
        auto  R  = hpro::TRkMatrix( UM->row_is(), UM->col_is(), std::move( blas::prod( UM->row_basis(), UM->coeff() ) ), std::move( blas::copy( UM->col_basis() ) ) );

        // no recursive updates left, apply accumulated updates and solve
        accu.apply( value_t(-1), R, acc, approx );
        
        hlr::solve_lower_tri< value_t >( side, diag, L, R, acc, approx );

        //
        // now replace M by R and update row/column bases
        //

        auto  W  = blas::mat_U< value_t >( R );
        auto  X  = blas::mat_V< value_t >( R );
        auto  RW = blas::matrix< value_t >();
        auto  RX = blas::matrix< value_t >();

        blas::qr( W, RW );
        blas::qr( X, RX );

        auto  T = blas::prod( RW, blas::adjoint( RX ) );
                    
        hlr::uniform::detail::update_row_col_basis( *UM, W, T, X, acc, approx, rowmap, colmap );
    }// if
    else
    {
        accu.apply( value_t(-1), M, acc, approx );

        hlr::solve_lower_tri< value_t >( side, diag, L, M, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    U,
                  hpro::TMatrix &          M,
                  accumulator &            accu,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx,
                  const uniform_map_t &    rowmap,
                  const uniform_map_t &    colmap ) //, hpro::TMatrix &          REF )
{
    // std::cout << M.id() << std::endl;

    // apply computable updates
    accu.eval( value_t(1), M, acc, approx );
    
    if ( is_blocked_all( U, M ) )
    {
        auto  BU = cptrcast( &U, hpro::TBlockMatrix );
        auto  BM =  ptrcast( &M, hpro::TBlockMatrix );
        // auto  BREF = ptrcast( &REF, hpro::TBlockMatrix );
        
        //
        // first, split accumulated updates U and recursive updates upd_rec
        // into subblock updates
        // - to release U before recursion and by that avoid memory
        //   consumption dependent on hierarchy depth
        //

        auto  sub_accu = accu.restrict< value_t >( *BM );

        accu.clear_matrix();

        if ( side == from_left )
        {
            HLR_ASSERT( false );
        }// if
        else
        {
            for ( uint j = 0; j < BM->nblock_cols(); ++j )
            {
                const auto  U_jj = BU->block( j, j );
            
                for ( uint i = 0; i < BM->nblock_rows(); ++i )
                    solve_upper_tri< value_t >( side, diag, *U_jj, *BM->block( i, j ),
                                                sub_accu(i,j), acc, approx, rowmap, colmap ); // *BREF->block( i, j ) );
            
                for ( uint  k = j+1; k < BM->nblock_cols(); ++k )
                    for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                        sub_accu(i,k).add_update( *BM->block(i,j), *BU->block(j,k) );
            }// for
        }// else
    }// if
    else if ( hlr::matrix::is_uniform_lowrank( M ) )
    {
        //
        // update and solve local matrix
        //

        auto  UM = ptrcast( &M, hlr::matrix::uniform_lrmatrix< value_t > );
        auto  R  = hlr::matrix::convert_to_lowrank< value_t >( M );

        // no recursive updates left, apply accumulated updates and solve
        accu.apply( value_t(-1), *R, acc, approx );

        hlr::solve_upper_tri< value_t >( side, diag, U, *R, acc, approx );

        //
        // now replace M by R and update row/column bases
        //

        auto  W  = blas::mat_U< value_t >( *R );
        auto  X  = blas::mat_V< value_t >( *R );
        auto  RW = blas::matrix< value_t >();
        auto  RX = blas::matrix< value_t >();

        blas::qr( W, RW );
        blas::qr( X, RX );

        auto  T = blas::prod( RW, blas::adjoint( RX ) );
        
        hlr::uniform::detail::update_row_col_basis( *UM, W, T, X, acc, approx, rowmap, colmap );
    }// if
    else
    {
        accu.apply( value_t(-1), M, acc, approx );

        hlr::solve_upper_tri< value_t >( side, diag, U, M, acc, approx );
    }// else
}

//
// recursive LU factorization
//
template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          A,
     accumulator &            accu,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx,
     const uniform_map_t &    rowmap,
     const uniform_map_t &    colmap )
// hpro::TMatrix &          REF )
{
    //
    // evaluate all computable updates to M
    //

    // std::cout << A.id() << std::endl;

    accu.eval( value_t(1), A, acc, approx );

    //
    // (recursive) LU factorization
    //

    if ( is_blocked( A ) )
    {
        auto  BA   = ptrcast( &A,   hpro::TBlockMatrix );
        // auto  BREF = ptrcast( &REF, hpro::TBlockMatrix );

        //
        // first, split accumulated updates U and recursive updates upd_rec
        // into subblock updates
        // - to release U before recursion and by that avoid memory
        //   consumption dependent on hierarchy depth
        //

        auto  sub_accu = accu.restrict< value_t >( *BA );

        accu.clear_matrix();

        //
        // recursive LU factorization but add updates to accumulator
        // instead of applying them
        //
        
        for ( uint  i = 0; i < std::min( BA->nblock_rows(), BA->nblock_cols() ); ++i )
        {
            HLR_ASSERT( ! is_null( BA->block( i, i ) ) );
            
            lu< value_t >( * BA->block( i, i ), sub_accu(i,i), acc, approx, rowmap, colmap ); // , *BREF->block( i, i ) );

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                if ( ! is_null( BA->block( j, i ) ) )
                    solve_upper_tri< value_t >( from_right, general_diag,
                                                *BA->block( i, i ), *BA->block( j, i ),
                                                sub_accu(j,i), acc, approx, rowmap, colmap ); // *BREF->block( j, i ) );
            }// for

            for ( uint  j = i+1; j < BA->nblock_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                    solve_lower_tri< value_t >( from_left, unit_diag,
                                                *BA->block( i, i ), *BA->block( i, j ),
                                                sub_accu(i,j), acc, approx, rowmap, colmap ); // *BREF->block( i, j ) );
            }// for

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
                for ( uint  l = i+1; l < BA->nblock_cols(); ++l )
                    if ( ! is_null_any( BA->block( j, i ), BA->block( i, l ) ) )
                        sub_accu(j,l).add_update( *BA->block( j, i ), *BA->block( i, l ) );
        }// for
    }// if
    else if ( is_dense( A ) )
    {
        auto  D = ptrcast( &A, hpro::TDenseMatrix );

        accu.apply( value_t(-1), A, acc, approx );
        
        invert< value_t >( *D );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

}}}}}// namespace hlr::hpx::uniform::accu::detail

#endif // __HLR_HPX_DETAIL_UNIFORM_ACCU_HH

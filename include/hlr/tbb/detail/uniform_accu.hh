#ifndef __HLR_TBB_DETAIL_UNIFORM_ACCU_HH
#define __HLR_TBB_DETAIL_UNIFORM_ACCU_HH
//
// Project     : HLR
// Module      : arith/detail/uniform_accu
// Description : arithmetic functions for uniform matrices using accumulators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <tbb/task_arena.h>

#include <hlr/arith/multiply.hh>
#include <hlr/arith/blas.hh>
#include <hlr/arith/solve.hh>
#include <hlr/arith/detail/uniform.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/restrict.hh>
#include <hlr/utils/hash.hh>
#include <hlr/utils/tensor.hh>

#include <hlr/tbb/detail/uniform_basis.hh>

namespace hlr { namespace tbb { namespace uniform { namespace accu { namespace detail {

using  hlr::matrix::shared_cluster_basis;
using  hlr::matrix::is_uniform_lowrank;
using  hlr::matrix::is_uniform_lowrank_all;
using  hlr::matrix::uniform_lrmatrix;
using  hlr::uniform::is_matrix_map_t;
using  hlr::tbb::uniform::detail::compute_extended_basis;
using  hlr::tbb::uniform::detail::update_coupling;

// maps index set to product U×V' of inner matrix product
template < typename value_t >
using  inner_map_t   = std::unordered_map< indexset, blas::matrix< value_t >, indexset_hash >;

//
// matrix update accumulator with parallel update handling
//
template < typename value_t >
struct accumulator
{
    //
    // represents an update, i.e., matrix product
    //
    struct update
    {
        const matop_t                     op_A;
        const Hpro::TMatrix< value_t > *  A;
        const matop_t                     op_B;
        const Hpro::TMatrix< value_t > *  B;
    };
    
    // represents set of updates
    using  update_list = std::list< update >;

    // accumulated computed updates
    std::unique_ptr< Hpro::TMatrix< value_t > >   matrix;

    // accumulated pending (recursive) updates
    update_list                        pending;

    // cached products
    inner_map_t< value_t > *           prod_inner;
    std::mutex *                       prod_inner_mtx;
    
    //
    // ctors
    //

    accumulator ( inner_map_t< value_t > *             aprod_inner     = nullptr,
                  std::mutex *                         aprod_inner_mtx = nullptr )
            : prod_inner( aprod_inner )
            , prod_inner_mtx( aprod_inner_mtx )
    {
        if ( ! is_null( prod_inner ) )
            HLR_ASSERT( ! is_null( prod_inner_mtx ) );
    }
    
    accumulator ( std::unique_ptr< Hpro::TMatrix< value_t > > &&  amatrix,
                  update_list &&                       apending,
                  inner_map_t< value_t > *             aprod_inner     = nullptr,
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
    Hpro::TMatrix< value_t > *
    release_matrix ()
    {
        return matrix.release();
    }

    //
    // add update A×B
    //
    void
    add_update ( const Hpro::TMatrix< value_t > &  A,
                 const Hpro::TMatrix< value_t > &  B )
    {
        pending.push_back( { apply_normal, &A, apply_normal, &B } );
    }

    void
    add_update ( const matop_t                     op_A,
                 const Hpro::TMatrix< value_t > &  A,
                 const matop_t                     op_B,
                 const Hpro::TMatrix< value_t > &  B )
    {
        pending.push_back( { op_A, &A, op_B, &B } );
    }
    
    //
    // apply accumulated updates and free accumulator matrix
    //
    template < typename approx_t >
    void
    apply ( const value_t               alpha,
            Hpro::TMatrix< value_t > &  M,
            const Hpro::TTruncAcc &     acc,
            const approx_t &            approx )
    {
        if ( ! is_null( matrix ) )
            hlr::add( alpha, *matrix, M, acc, approx );

        clear_matrix();
    }
    
    //
    // return restriction of updates to block (i,j) of given block matrix
    //
    accumulator
    restrict ( const uint                             i,
               const uint                             j,
               const Hpro::TBlockMatrix< value_t > &  M ) const
    {
        auto  U_ij = std::unique_ptr< Hpro::TMatrix< value_t > >();
        auto  P_ij = update_list();
        
        if ( ! is_null( matrix ) )
        {
            HLR_ASSERT( ! is_null( M.block( i, j ) ) );
            
            U_ij = hlr::matrix::restrict( *matrix, M.block( i, j )->block_is() );

            // prefer dense if destination is dense
            if ( check_dense( *M.block( i, j ) ) && ! hlr::matrix::is_dense( *U_ij ) )
                U_ij = std::move( hlr::matrix::convert_to_dense< value_t >( *U_ij ) );
        }// if

        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            // filter out non-recursive updates
            if ( ! is_blocked_all( A, B ) )
                continue;
                                            
            auto  BA = cptrcast( A, Hpro::TBlockMatrix< value_t > );
            auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );
                                            
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
    tensor2< accumulator >
    restrict ( const Hpro::TBlockMatrix< value_t > &  M ) const
    {
        tensor2< accumulator >  sub_accu( M.nblock_rows(), M.nblock_cols() );
        
        for ( uint  i = 0; i < M.nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < M.nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( M.block( i, j ) ) );

                sub_accu(i,j) = restrict( i, j, M );
            }// for
        }// for

        return sub_accu;
    }
    
    //
    // evaluate all computable updates to matrix M
    //
    template < typename approx_t >
    void
    eval ( const value_t                     alpha,
           const Hpro::TMatrix< value_t > &  M,
           const Hpro::TTruncAcc &           acc,
           const approx_t &                  approx )
    {
        //
        // first check for dense handling
        //

        bool  handle_dense = check_dense( M );

        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            if ( hlr::matrix::is_dense_all( A, B ) ||
                 ( is_blocked( A ) && hlr::matrix::is_dense( B ) ) ||
                 ( is_blocked( B ) && hlr::matrix::is_dense( A )  ))
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
            
            auto  pending_uniAB = std::vector< update >();
            auto  pending_uniA  = std::vector< update >();
            auto  pending_uniB  = std::vector< update >();
        
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

            ::tbb::parallel_invoke(

                [&,this] ()
                {
                    if ( ! pending_uniAB.empty() )
                        R = std::move( sum_uni_uni( 0, pending_uniAB.size(), pending_uniAB ) );
                },

                [&,this] ()
                {
                    if ( ! pending_uniA.empty() )
                        Z = std::move( sum_uni_any( 0, pending_uniA.size(), pending_uniA ) );
                },

                [&,this] ()
                {
                    if ( ! pending_uniB.empty() )
                        Y = std::move( sum_any_uni( 0, pending_uniB.size(), pending_uniB ) );
                }
            );

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
                        matrix = std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(),
                                                                                  std::move( blas::prod( alpha, U, R ) ),
                                                                                  std::move( blas::copy( V ) ) );
                    else
                    {
                        auto  US = blas::prod( U, R );
                        auto  T  = matrix::lrmatrix< value_t >( M.row_is(), M.col_is(), US, V );
                
                        hlr::add( alpha, T, *matrix, acc, approx );
                    }// else
                }// else
            }// if

            if ( Z.ncols() > 0 )
            {
                if ( is_null( matrix ) )
                {
                    blas::scale( alpha, Z );
                    matrix = std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( blas::copy( U ) ), std::move( Z ) );
                }// if
                else
                {
                    auto  T = matrix::lrmatrix< value_t >( M.row_is(), M.col_is(), U, Z );
                    
                    hlr::add( alpha, T, *matrix, acc, approx );
                }// else
            }// if
            
            if ( Y.ncols() > 0 )
            {
                if ( is_null( matrix ) )
                {
                    blas::scale( alpha, Y );
                    matrix = std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( Y ), std::move( blas::copy( V ) ) );
                }// if
                else
                {
                    auto  T = matrix::lrmatrix< value_t >( M.row_is(), M.col_is(), Y, V );
                    
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

            auto  T = std::move( sum_pending( 0, vpending.size(), vpending, handle_dense, acc, approx ) );

            if ( is_null( matrix ) )
            {
                matrix = std::move( T );
            }// if
            else if ( ! hlr::matrix::is_dense( *matrix ) && hlr::matrix::is_dense( *T ) )
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

        auto  BC = std::unique_ptr< Hpro::TBlockMatrix< value_t > >(); // for recursive handling

        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            if ( hlr::is_blocked_all( *A, *B, M ) )
                continue;
        
            if ( hlr::is_blocked_all( *A, *B ) )
            {
                //
                // if M is a leaf and A _and_ B are blocked, a temporary matrix
                // is created for further recursive update handling
                //

                if ( ! is_null( BC ) )
                    continue;
                
                // TODO: non low-rank M
                if ( ! ( matrix::is_lowrank( M ) || matrix::is_lowrankS( M ) || is_uniform_lowrank( M ) ) )
                    HLR_ERROR( "unsupported matrix type: " + M.typestr() );
                
                auto  BA = cptrcast( A, Hpro::TBlockMatrix< value_t > );
                auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );
                
                BC = std::make_unique< Hpro::TBlockMatrix< value_t > >( A->row_is( op_A ), B->col_is( op_B ) );

                BC->set_block_struct( BA->nblock_rows( op_A ), BB->nblock_cols( op_B ) );

                for ( uint  i = 0; i < BC->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                    {
                        HLR_ASSERT( ! is_null_any( BA->block( i, 0, op_A ), BB->block( 0, j, op_B ) ) );
                        
                        if ( handle_dense )
                            BC->set_block( i, j, new matrix::dense_matrix< value_t >( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                                      BB->block( 0, j, op_B )->col_is( op_B ) ) );
                        else
                            BC->set_block( i, j, new matrix::lrmatrix< value_t >( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                                  BB->block( 0, j, op_B )->col_is( op_B ) ) );
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

            auto  sub_accu = restrict( *BC );

            matrix.reset( nullptr );
        
            //
            // apply recursive updates
            //

            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                                0, BC->nblock_cols() ),
                [&,alpha] ( const auto &  r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    {
                        for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                        {
                            sub_accu(i,j).eval( alpha, *BC->block(i,j), acc, approx );

                            // replace block in BC by accumulator matrix for agglomeration below
                            BC->delete_block( i, j );
                            BC->set_block( i, j, sub_accu(i,j).release_matrix() );
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

            ::tbb::parallel_invoke( [&,lb,mid] () { R1 = sum_uni_uni( lb, mid, updates ); },
                                    [&,mid,ub] () { R2 = sum_uni_uni( mid, ub, updates ); } );
            
            blas::add( value_t(1), R1, R2 );

            return std::move( R2 );
        }// else
    }

    //
    // compute sum of uniform × anything updates
    //
    blas::matrix< value_t >
    sum_uni_any ( const int                lb,
                  const int                ub,
                  std::vector< update > &  updates )
    {
        if ( ub - lb <= 1 )
        {
            //
            // U ( Σ_i S_i X_i' ) × B_i = U ( Σ_i S_i ( X_i' × B_i ) )
            //                          = U ( Σ_i S_i Z_i' )   with Z_i = B_i' × X_i
            //                          = U ( Σ_i (Z_i B_i'))'
            //                          = U Z'                 with Z   = Σ_i (Z_i S_i')
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

            ::tbb::parallel_invoke( [&,lb,mid] () { Z1 = sum_uni_any( lb, mid, updates ); },
                                    [&,mid,ub] () { Z2 = sum_uni_any( mid, ub, updates ); } );

            blas::add( value_t(1), Z1, Z2 );

            return std::move( Z2 );
        }// else
    }
    
    //
    // compute sum of anything × uniform updates
    //
    blas::matrix< value_t >
    sum_any_uni ( const int                lb,
                  const int                ub,
                  std::vector< update > &  updates )
    {
        if ( ub - lb <= 1 )
        {
            //
            // A_i × ( Σ_i W_i S_i ) V' = ( Σ_i ( A_i × W_i ) S_i ) V'
            //                          = ( Σ_i Y_i S_i ) V' with Y_i = A_i × W_i
            //                          = ( Σ_i (Y_i S_i) ) V'
            //                          = Y V'
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

            ::tbb::parallel_invoke( [&,lb,mid] () { Y1 = sum_any_uni( lb, mid, updates ); },
                                    [&,mid,ub] () { Y2 = sum_any_uni( mid, ub, updates ); } );

            blas::add( value_t(1), Y1, Y2 );

            return std::move( Y2 );
        }// else
    }
    
    //
    // compute sum of general updates with non-leaf types
    // for at least one factor
    //
    template < typename approx_t >
    std::unique_ptr< Hpro::TMatrix< value_t > >
    sum_pending ( const int                lb,
                  const int                ub,
                  std::vector< update > &  updates,
                  const bool               handle_dense,
                  const Hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
    {
        if ( ub - lb <= 1 )
        {
            auto  [ op_A, A, op_B, B ] = updates[ lb ];
            auto  T                    = std::unique_ptr< Hpro::TMatrix< value_t > >();

            if ( handle_dense ||
                 hlr::matrix::is_dense_all( A, B ) ||
                 ( is_blocked( A ) && hlr::matrix::is_dense(   B ) ) ||
                 ( hlr::matrix::is_dense(   A ) && is_blocked( B ) ))
                T = std::make_unique< matrix::dense_matrix< value_t > >( A->row_is( op_A ), B->col_is( op_B ) );
            else
            {
                std::cout << "!!! : " << A->typestr() << " x " << B->typestr() << std::endl;
                T = std::make_unique< matrix::lrmatrix< value_t > >( A->row_is( op_A ), B->col_is( op_B ) );
            }// else

            hlr::multiply< value_t >( value_t(1), op_A, *A, op_B, *B, *T, acc, approx );

            return T;
        }// if
        else
        {
            const int  mid = ( ub + lb ) / 2;
            auto       T1  = std::unique_ptr< Hpro::TMatrix< value_t > >();
            auto       T2  = std::unique_ptr< Hpro::TMatrix< value_t > >();

            ::tbb::parallel_invoke( [&,lb,mid,handle_dense] () { T1 = sum_pending( lb, mid, updates, handle_dense, acc, approx ); },
                                    [&,mid,ub,handle_dense] () { T2 = sum_pending( mid, ub, updates, handle_dense, acc, approx ); } );

            if ( hlr::matrix::is_dense( *T1 ) )
            {
                hlr::add( value_t(1), *T2, *T1 );

                return T1;
            }// if
            else if ( hlr::matrix::is_dense( *T2 ) )
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
    check_dense ( const Hpro::TMatrix< value_t > &  M ) const
    {
        // return false;
        if ( hlr::matrix::is_dense( M ) )
        {
            return true;
        }// if
        else if ( is_blocked( M ) )
        {
            //
            // test if all subblocks are dense
            //

            auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( ! is_null( B->block( i, j ) ) && ! hlr::matrix::is_dense( B->block( i, j ) ) )
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
// build block mappings (indexset to matrix block)
//
template < typename value_t >
void
build_block_maps ( Hpro::TMatrix< value_t > &    A,
                   is_matrix_map_t< value_t > &  rowmap,
                   is_matrix_map_t< value_t > &  colmap )
{
    auto  blocks = std::list< Hpro::TMatrix< value_t > * >{ &A };

    while ( ! blocks.empty() )
    {
        auto  subblocks = decltype( blocks )();

        for ( auto  M : blocks )
        {
            if ( is_blocked( M ) )
            {
                auto  BM = ptrcast( M, Hpro::TBlockMatrix< value_t > );

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
// implements recursive matrix multiplication with accumulator
//
template < typename value_t >
struct rec_matrix_mult
{
    // maps indexsets to set of uniform matrices sharing corresponding cluster basis
    is_matrix_map_t< value_t >   rowmap, colmap;

    //
    // ctor
    //
    rec_matrix_mult ( Hpro::TMatrix< value_t > &  A )
    {
        build_block_maps( A, rowmap, colmap );
    }

    template < typename approx_t >
    void
    multiply ( const value_t               alpha,
               Hpro::TMatrix< value_t > &  M,
               accumulator< value_t > &    accu,
               const Hpro::TTruncAcc &     acc,
               const approx_t &            approx )
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
            auto  BM       = ptrcast( &M, Hpro::TBlockMatrix< value_t > );
            auto  sub_accu = accu.restrict( *BM );

            accu.clear_matrix();

            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( 0, BM->nblock_rows(),
                                                0, BM->nblock_cols() ),
                [&,alpha] ( const auto &  r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    {
                        for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                        {
                            if ( is_null( BM->block( i, j ) ) )
                                continue;

                            multiply( alpha, * BM->block( i, j ), sub_accu( i, j ), acc, approx );
                        }// for
                    }// for
                } );
        }// if
        else if ( hlr::matrix::is_uniform_lowrank( M ) )
        {
            //
            // update local matrix as standard low-rank matrix
            //

            auto    U     = ptrcast( &M, hlr::matrix::uniform_lrmatrix< value_t > );
            auto &  rowcb = U->row_cb();
            auto &  colcb = U->col_cb();
            auto    R     = matrix::lrmatrix< value_t >( U->row_is(), U->col_is() );

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

            auto  W  = R.U(); // std::move( blas::mat_U< value_t >( R ) );
            auto  X  = R.V(); // std::move( blas::mat_V< value_t >( R ) );
            auto  RW = blas::matrix< value_t >();
            auto  RX = blas::matrix< value_t >();

            ::tbb::parallel_invoke( [&] () { blas::qr( W, RW ); },
                                    [&] () { blas::qr( X, RX ); } );

            auto  T  = blas::prod( RW, blas::adjoint( RX ) );

            ::tbb::this_task_arena::isolate( [&] ()
            {
                auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
                
                ::tbb::parallel_invoke(
                    [&] ()
                    {
                        auto  mtx = std::mutex(); // just dummy
                        auto  Un  = compute_extended_basis( rowcb, W, T, acc, approx, rowmap, mtx, apply_adjoint, U );
                        
                        update_coupling( rowcb, Un, rowmap, mtx, false, U );
                        rowcb.set_basis( std::move( Un ) );
                    },
                
                    [&] ()
                    {
                        auto  mtx = std::mutex(); // just dummy
                        auto  Vn  = compute_extended_basis( colcb, X, T, acc, approx, colmap, mtx, apply_normal, U );
                        
                        update_coupling( colcb, Vn, colmap, mtx, true, U );
                        colcb.set_basis( std::move( Vn ) );
                    }
                );

                //
                // transform M into new bases
                //

                auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), W );
                auto  TV = blas::prod( blas::adjoint( colcb.basis() ), X );
                auto  TS = blas::prod( TU, T );
                auto  S  = blas::prod( TS, blas::adjoint( TV ) );

                U->set_coeff( std::move( S ) );
            } );
        }// if
        else
        {
            accu.apply( alpha, M, acc, approx );
        }// else
    }
};

//
// implements recursive LU factorization with accumulator
//
template < typename value_t >
struct rec_lu_factorization
{
    // maps indexsets to set of uniform matrices sharing corresponding cluster basis
    is_matrix_map_t< value_t >   rowmap_L, colmap_L;
    is_matrix_map_t< value_t >   rowmap_U, colmap_U;

    // mutices for mappings
    std::mutex                   rowmap_L_mtx, colmap_L_mtx;
    std::mutex                   rowmap_U_mtx, colmap_U_mtx;
    
    //
    // ctor
    //
    rec_lu_factorization ( Hpro::TMatrix< value_t > &  L,
                           Hpro::TMatrix< value_t > &  U )
    {}

    template < typename approx_t >
    void
    solve_lower_tri ( const eval_side_t                  side,
                      const diag_type_t                  diag,
                      const Hpro::TMatrix< value_t > &   L,
                      Hpro::TMatrix< value_t > &         M,
                      accumulator< value_t > &           accu,
                      const Hpro::TTruncAcc &            acc,
                      const approx_t &                   approx,
                      shared_cluster_basis< value_t > &  rowcb,   // new cluster bases for M
                      shared_cluster_basis< value_t > &  colcb )
    {
        //
        // evaluate all computable updates to M
        //

        trace::region_start( "eval" );

        accu.eval( value_t(1), M, acc, approx );

        trace::region_end( "eval" );
    
        if ( is_blocked_all( L, M ) )
        {
            auto  BL = cptrcast( &L, Hpro::TBlockMatrix< value_t > );
            auto  BM =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
            //
            // first, split accumulated updates U and recursive updates upd_rec
            // into subblock updates
            // - to release U before recursion and by that avoid memory
            //   consumption dependent on hierarchy depth
            //

            auto  sub_accu = accu.restrict( *BM );

            accu.clear_matrix();

            if ( side == from_left )
            {
                for ( uint i = 0; i < BM->nblock_rows(); ++i )
                {
                    const auto  L_ii = BL->block( i, i );
            
                    HLR_ASSERT( ! is_null( rowcb.son(i) ) );

                    for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                    // ::tbb::parallel_for< uint >(
                    //     0, BM->nblock_cols(),
                    //     [&,side,diag,L_ii,BM,i] ( const uint  j )
                    {
                        HLR_ASSERT( ! is_null( colcb.son(j) ) );
                    
                        solve_lower_tri< value_t >( side, diag, *L_ii, *BM->block(i,j),
                                                    sub_accu(i,j), acc, approx,
                                                    *rowcb.son(i), *colcb.son(j) );
                    }// );

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
            // std::cout << M.id() << "  " << M.block_is().to_string() << std::endl;
            
            //
            // update and solve local matrix
            //

            auto  UM = ptrcast( &M, hlr::matrix::uniform_lrmatrix< value_t > );
            auto  R  = matrix::lrmatrix< value_t >( M.row_is(), M.col_is() );

            {
                std::scoped_lock  lock( M.mutex(), rowcb.mutex(), colcb.mutex() );

                R.set_lrmat( std::move( blas::prod( UM->row_basis(), UM->coeff() ) ),
                             std::move( blas::copy( UM->col_basis() ) ) );
            }
            
            trace::region_start( "apply" );
            
            // no recursive updates left, apply accumulated updates and solve
            accu.apply( value_t(-1), R, acc, approx );
        
            trace::region_end( "apply" );
            
            hlr::solve_lower_tri< value_t >( side, diag, L, R, acc, approx );

            //
            // now replace M by R and update row/column bases
            //

            trace::region_start( "basis" );
        
            auto  W  = R.U(); // std::move( blas::mat_U< value_t >( R ) );
            auto  X  = R.V(); // std::move( blas::mat_V< value_t >( R ) );
            auto  RW = blas::matrix< value_t >();
            auto  RX = blas::matrix< value_t >();

            ::tbb::parallel_invoke( [&] () { blas::qr( W, RW ); },
                                    [&] () { blas::qr( X, RX ); } );

            auto  T  = blas::prod( RW, blas::adjoint( RX ) );

            ::tbb::this_task_arena::isolate( [&] ()
            {
                auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
                
                ::tbb::parallel_invoke(
                    [&] ()
                    {
                        auto  Un = compute_extended_basis( rowcb, W, T, acc, approx, rowmap_U, rowmap_U_mtx, apply_adjoint );

                        update_coupling( rowcb, Un, rowmap_U, rowmap_U_mtx, false );
                        rowcb.set_basis( std::move( Un ) );
                    },
                        
                    [&] ()
                    {
                        auto  Vn = compute_extended_basis( colcb, X, T, acc, approx, colmap_U, colmap_U_mtx, apply_normal );
            
                        update_coupling( colcb, Vn, colmap_U, colmap_U_mtx, true );
                        colcb.set_basis( std::move( Vn ) );
                    }
                );

                //
                // update new basis and replace data in M
                //
        
                auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), W );
                auto  TV = blas::prod( blas::adjoint( colcb.basis() ), X );
                auto  TS = blas::prod( TU, T );
                auto  Sn = blas::prod( TS, blas::adjoint( TV ) );

                UM->set_matrix_data( rowcb, std::move( Sn ), colcb );

                // M now also part of matrices sharing rowcb/colcb
                {
                    auto  lock_is = std::scoped_lock( rowmap_U_mtx, colmap_U_mtx );
                    
                    rowmap_U[ rowcb.is() ].push_back( UM );
                    colmap_U[ colcb.is() ].push_back( UM );
                }
                
                trace::region_end( "basis" );
            } );
        }// if
        else
        {
            accu.apply( value_t(-1), M, acc, approx );

            hlr::solve_lower_tri< value_t >( side, diag, L, M, acc, approx );
        }// else
    }

    template < typename approx_t >
    void
    solve_upper_tri ( const eval_side_t                  side,
                      const diag_type_t                  diag,
                      const Hpro::TMatrix< value_t > &   U,
                      Hpro::TMatrix< value_t > &         M,
                      accumulator< value_t > &           accu,
                      const Hpro::TTruncAcc &            acc,
                      const approx_t &                   approx,
                      shared_cluster_basis< value_t > &  rowcb,   // new cluster bases for M
                      shared_cluster_basis< value_t > &  colcb )
    {
        //
        // evaluate all computable updates to M
        //

        trace::region_start( "eval" );
    
        accu.eval( value_t(1), M, acc, approx );
    
        trace::region_end( "eval" );
        
        if ( is_blocked_all( U, M ) )
        {
            auto  BU = cptrcast( &U, Hpro::TBlockMatrix< value_t > );
            auto  BM =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
            //
            // first, split accumulated updates U and recursive updates upd_rec
            // into subblock updates
            // - to release U before recursion and by that avoid memory
            //   consumption dependent on hierarchy depth
            //

            auto  sub_accu = accu.restrict( *BM );

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
                    
                    HLR_ASSERT( ! is_null_any( U_jj, colcb.son(j) ) );

                    for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                    // ::tbb::parallel_for< uint >(
                    //     0, BM->nblock_rows(),
                    //     [&,side,diag,U_jj,BM,j] ( const uint  i )
                    {
                        HLR_ASSERT( ! is_null( rowcb.son(i) ) );
                    
                        solve_upper_tri< value_t >( side, diag, *U_jj, *BM->block( i, j ),
                                                    sub_accu(i,j), acc, approx, 
                                                    *rowcb.son(i), *colcb.son(j) );
                    }//  );
            
                    for ( uint  k = j+1; k < BM->nblock_cols(); ++k )
                        for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                            sub_accu(i,k).add_update( *BM->block(i,j), *BU->block(j,k) );
                }// for
            }// else
        }// if
        else if ( hlr::matrix::is_uniform_lowrank( M ) )
        {
            // std::cout << M.id() << "  " << M.block_is().to_string() << std::endl;
            
            //
            // update and solve local matrix
            //

            auto  UM = ptrcast( &M, uniform_lrmatrix< value_t > );
            auto  R  = matrix::lrmatrix< value_t >( M.row_is(), M.col_is() );
                                             
            {
                std::scoped_lock  lock( M.mutex(), rowcb.mutex(), colcb.mutex() );

                R.set_lrmat( std::move( blas::prod( UM->row_basis(), UM->coeff() ) ),
                             std::move( blas::copy( UM->col_basis() ) ) );
            }

            trace::region_start( "apply" );
        
            // no recursive updates left, apply accumulated updates and solve
            accu.apply( value_t(-1), R, acc, approx );

            trace::region_end( "apply" );

            hlr::solve_upper_tri< value_t >( side, diag, U, R, acc, approx );

            //
            // now replace M by R and update row/column bases
            //

            trace::region_start( "basis" );
        
            auto  W  = R.U(); // std::move( blas::mat_U< value_t >( R ) );
            auto  X  = R.V(); // std::move( blas::mat_V< value_t >( R ) );
            auto  RW = blas::matrix< value_t >();
            auto  RX = blas::matrix< value_t >();

            ::tbb::parallel_invoke( [&] () { blas::qr( W, RW ); },
                                    [&] () { blas::qr( X, RX ); } );

            auto  T = blas::prod( RW, blas::adjoint( RX ) );

            ::tbb::this_task_arena::isolate( [&] ()
            {
                auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
                
                ::tbb::parallel_invoke(
                    [&] ()
                    {
                        auto  Un = compute_extended_basis( rowcb, W, T, acc, approx, rowmap_L, rowmap_L_mtx, apply_adjoint );

                        update_coupling( rowcb, Un, rowmap_L, rowmap_L_mtx, false );
                        rowcb.set_basis( std::move( Un ) );
                    },
                        
                    [&] ()
                    {
                        auto  Vn = compute_extended_basis( colcb, X, T, acc, approx, colmap_L, colmap_L_mtx, apply_normal );
            
                        update_coupling( colcb, Vn, colmap_L, colmap_L_mtx, true );
                        colcb.set_basis( std::move( Vn ) );
                    }
                );

                //
                // update new basis and replace data in M
                //
                
                auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), W );
                auto  TV = blas::prod( blas::adjoint( colcb.basis() ), X );
                auto  TS = blas::prod( TU, T );
                auto  Sn = blas::prod( TS, blas::adjoint( TV ) );
                
                UM->set_matrix_data( rowcb, std::move( Sn ), colcb );

                // M now also part of matrices sharing rowcb/colcb
                {
                    auto  lock_is = std::scoped_lock( rowmap_L_mtx, colmap_L_mtx );
                    
                    rowmap_L[ rowcb.is() ].push_back( UM );
                    colmap_L[ colcb.is() ].push_back( UM );
                }

                trace::region_end( "basis" );
            } );
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
    template < typename approx_t >
    void
    lu ( Hpro::TMatrix< value_t > &         A,
         Hpro::TMatrix< value_t > &         L,
         Hpro::TMatrix< value_t > &         U,
         accumulator< value_t > &           accu,
         const Hpro::TTruncAcc &            acc,
         const approx_t &                   approx,
         shared_cluster_basis< value_t > &  rowcb_L, // new cluster bases for L
         shared_cluster_basis< value_t > &  colcb_L,
         shared_cluster_basis< value_t > &  rowcb_U, // new cluster bases for U
         shared_cluster_basis< value_t > &  colcb_U )
    {
        //
        // evaluate all computable updates to M
        //

        trace::region_start( "eval" );
    
        accu.eval( value_t(1), A, acc, approx );
    
        trace::region_end( "eval" );

        //
        // (recursive) LU factorization
        //

        if ( is_blocked( A ) )
        {
            auto  BA = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
            auto  BL = ptrcast( &L, Hpro::TBlockMatrix< value_t > );
            auto  BU = ptrcast( &U, Hpro::TBlockMatrix< value_t > );

            //
            // first, split accumulated updates U and recursive updates upd_rec
            // into subblock updates
            // - to release U before recursion and by that avoid memory
            //   consumption dependent on hierarchy depth
            //

            auto  sub_accu = accu.restrict( *BA );

            accu.clear_matrix();

            //
            // recursive LU factorization but add updates to accumulator
            // instead of applying them
            //
        
            for ( uint  i = 0; i < std::min( BA->nblock_rows(), BA->nblock_cols() ); ++i )
            {
                HLR_ASSERT( ! is_null( BA->block( i, i ) ) );
            
                lu< value_t >( * BA->block( i, i ), * BL->block( i, i ), * BU->block( i, i ),
                               sub_accu(i,i), acc, approx,
                               *rowcb_L.son(i), *colcb_L.son(i),
                               *rowcb_U.son(i), *colcb_U.son(i) );

                // ::tbb::parallel_invoke( 
                //     [&,i,BA,BU,BL] ()
                //     {
                //         ::tbb::parallel_for< uint >(
                //             i+1, BA->nblock_rows(),
                //             [&,i,BA,BU,BL] ( const uint  j )
                            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
                            {
                                if ( ! is_null( BA->block( j, i ) ) )
                                    solve_upper_tri< value_t >( from_right, general_diag,
                                                                *BU->block( i, i ), *BL->block( j, i ),
                                                                sub_accu(j,i), acc, approx,
                                                                *rowcb_L.son(j), *colcb_L.son(i) );
                            }// );
                    // },
                    
                    // [&,i,BA,BU,BL] ()
                    // {
                    //     ::tbb::parallel_for< uint >(
                    //         i+1, BA->nblock_cols(),
                    //         [&,i,BA,BU,BL] ( const uint  j )
                            for ( uint  j = i+1; j < BA->nblock_cols(); ++j )
                            {
                                if ( ! is_null( BA->block( i, j ) ) )
                                    solve_lower_tri< value_t >( from_left, unit_diag,
                                                                *BL->block( i, i ), *BU->block( i, j ),
                                                                sub_accu(i,j), acc, approx,
                                                                *rowcb_U.son(i), *colcb_U.son(j) );
                            }// );
                //     }
                // );

                for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
                    for ( uint  l = i+1; l < BA->nblock_cols(); ++l )
                        if ( ! is_null_any( BA->block( j, i ), BA->block( i, l ) ) )
                            sub_accu(j,l).add_update( *BL->block( j, i ), *BU->block( i, l ) );
            }// for
        }// if
        else if ( hlr::matrix::is_dense( A ) )
        {
            auto  DA = ptrcast( &A, matrix::dense_matrix< value_t > );
            auto  DU = ptrcast( &U, matrix::dense_matrix< value_t > );

            accu.apply( value_t(-1), A, acc, approx );

            auto  Am = DA->mat();
            auto  Um = DU->mat();
        
            blas::copy( Am, Um );
            blas::invert( Um );

            if ( DU->is_compressed() )
                DU->set_matrix( std::move( Um ), acc );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + A.typestr() );
    }
};

}}}}}// namespace hlr::tbb::uniform::accu::detail

#endif // __HLR_TBB_DETAIL_UNIFORM_ACCU_HH

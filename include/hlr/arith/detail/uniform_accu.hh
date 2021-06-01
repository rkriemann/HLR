#ifndef __HLR_ARITH_DETAIL_UNIFORM_ACCU_HH
#define __HLR_ARITH_DETAIL_UNIFORM_ACCU_HH
//
// Project     : HLib
// Module      : arith/uniform
// Description : arithmetic functions for uniform matrices using accumulators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

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

namespace hlr { namespace uniform { namespace accu {

namespace timer = HLIB::Time::Wall;

// #define HLR_UNIFORM_ACCU_TIC( t )    auto  __tic ## t = timer::now()
// #define HLR_UNIFORM_ACCU_TOC( t )  { auto  __toc ## t = timer::since( __tic ## t ); t_ ## t += __toc ## t.seconds(); }
#define HLR_UNIFORM_ACCU_TIC( t )  {}
#define HLR_UNIFORM_ACCU_TOC( t )  {}

extern double  t_basis;
extern double  t_apply;
extern double  t_apply_uni;
extern double  t_apply_dense;
extern double  t_eval;
extern double  t_eval_uni;
extern double  t_eval_rest;
extern double  t_eval_rec;
extern double  t_add_accu;
extern size_t  n_inner;
extern size_t  n_prodA;
extern size_t  n_prodB;

namespace detail
{

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
    
    //
    // ctors
    //

    accumulator ( inner_map_t *                        aprod_inner = nullptr )
            : prod_inner( aprod_inner )
    {}
    
    accumulator ( std::unique_ptr< hpro::TMatrix > &&  amatrix,
                  update_list &&                       apending,
                  inner_map_t *                        aprod_inner = nullptr )
            : matrix( std::move( amatrix ) )
            , pending( std::move( apending ) )
            , prod_inner( aprod_inner )
    {}
    
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

        return accumulator{ std::move( U_ij ), std::move( P_ij ), prod_inner };
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
            HLR_UNIFORM_ACCU_TIC( eval_uni );
            
            //
            // filter out different variants of uniform factors
            //
            
            update_list  pending_uniAB;
            update_list  pending_uniA;
            update_list  pending_uniB;
        
            for ( auto  it = pending.begin(); it != pending.end(); )
            {
                if ( hlr::matrix::is_uniform_lowrank_all( (*it).A, (*it).B ) )
                {
                    pending_uniAB.push_back( *it );
                    it = pending.erase( it );
                }// if
                else
                    ++it;
            }// for
            
            for ( auto  it = pending.begin(); it != pending.end(); )
            {
                if ( hlr::matrix::is_uniform_lowrank( (*it).A ) )
                {
                    pending_uniA.push_back( *it );
                    it = pending.erase( it );
                }// if
                else
                    ++it;
            }// for

            for ( auto  it = pending.begin(); it != pending.end(); )
            {
                if ( hlr::matrix::is_uniform_lowrank( (*it).B ) )
                {
                    pending_uniB.push_back( *it );
                    it = pending.erase( it );
                }// if
                else
                    ++it;
            }// for
        
            //
            // first uniform x uniform only as only coefficients need to be added without
            // truncation due to shared bases:
            //
            //   U ( Σ_i S_i X_i' W_i T_i ) V' = U R V'  with R = Σ_i S_i X_i' W_i T_i
            //

            auto  U = blas::matrix< value_t >();
            auto  V = blas::matrix< value_t >();
            auto  R = blas::matrix< value_t >();

            for ( auto  [ op_A, A, op_B, B ] : pending_uniAB )
            {
                auto  RA  = cptrcast( A, matrix::uniform_lrmatrix< value_t > );
                auto  RB  = cptrcast( B, matrix::uniform_lrmatrix< value_t > );

                auto  S   = RA->coeff();
                auto  X   = RA->col_basis( op_A );
                auto  W   = RB->row_basis( op_B );
                auto  T   = RB->coeff();
                auto  XW  = blas::matrix< value_t >();

                if ( ! is_null( prod_inner ) )
                {
                    // store product if not present
                    if ( prod_inner->find( RA->col_is( op_A ) ) == prod_inner->end() )
                        prod_inner->emplace( RA->col_is( op_A ), std::move( blas::prod( blas::adjoint( X ), W ) ) );
                    else
                        n_inner++;

                    XW = prod_inner->at( RA->col_is( op_A ) );
                }// if
                else
                    XW = std::move( blas::prod( blas::adjoint( X ), W ) );
                        
                auto  SXW = blas::prod( blas::mat_view( op_A, S ), XW );

                if ( U.ncols() == 0 )
                    U = RA->row_basis( op_A );

                if ( V.ncols() == 0 )
                    V = RB->col_basis( op_B );
                
                if ( R.nrows() == 0 )
                    R = std::move( blas::prod( SXW, blas::mat_view( op_B, T ) ) );
                else 
                    blas::prod( value_t(1), SXW, blas::mat_view( op_B, T ), value_t(1), R );
            }// for
        
            //
            // now handle all uniform x non-uniform
            //
            //   U ( Σ_i S_i X_i' ) × B_i = U ( Σ_i S_i ( X_i' × B_i ) )
            //                            = U ( Σ_i S_i Z_i' )   with Z_i = B_i' × X_i
            //                            = U ( Σ_i (Z_i B_i'))'
            //                            = U Z'                 with Z   = Σ_i (Z_i S_i')
            //

            auto  Z = blas::matrix< value_t >();

            for ( auto  [ op_A, A, op_B, B ] : pending_uniA )
            {
                auto  RA = cptrcast( A, matrix::uniform_lrmatrix< value_t > );
                auto  S  = RA->coeff();
                auto  X  = RA->col_basis( op_A );

                if ( U.ncols() == 0 )
                    U = RA->row_basis( op_A );
                
                if ( Z.ncols() == 0 )
                    Z = std::move( blas::matrix< value_t >( M.ncols(), U.ncols() ) );

                auto  Z_i = std::move( blas::matrix< value_t >( B->ncols(), X.ncols() ) );

                hlr::multiply( alpha, blas::adjoint( op_B ), *B, X, Z_i );
                blas::prod( value_t(1), Z_i, blas::adjoint( S ), value_t(1), Z );
            }// for
        
            //
            // now handle all non-uniform x uniform
            //
            //   A_i × ( Σ_i W_i S_i ) V' = ( Σ_i ( A_i × W_i ) S_i ) V'
            //                            = ( Σ_i Y_i S_i ) V' with Y_i = A_i × W_i
            //                            = ( Σ_i (Y_i S_i) ) V'
            //                            = Y V'
            //

            auto  Y = blas::matrix< value_t >();

            for ( auto  [ op_A, A, op_B, B ] : pending_uniB )
            {
                auto  RB = cptrcast( B, matrix::uniform_lrmatrix< value_t > );
                auto  W  = RB->row_basis( op_B );
                auto  S  = RB->coeff();

                if ( V.ncols() == 0 )
                    V = RB->col_basis( op_B );
                    
                if ( Y.ncols() == 0 )
                    Y = std::move( blas::matrix< value_t >( M.nrows(), V.ncols() ) );

                auto  Y_i = std::move( blas::matrix< value_t >( A->nrows(), W.ncols() ) );

                hlr::multiply( alpha, op_A, *A, W, Y_i );
                blas::prod( value_t(1), Y_i, S, value_t(1), Y );
            }// for

            HLR_UNIFORM_ACCU_TOC( eval_uni );
            
            //
            // sum up individual updates
            //

            HLR_UNIFORM_ACCU_TIC( add_accu );
                    
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
                        auto  US = blas::prod( alpha, U, R );
                        auto  T  = hpro::TRkMatrix( M.row_is(), M.col_is(), US, V );
                
                        hlr::add( alpha, T, *matrix, acc, approx );

                    }// else
                }// else
            }// if

            // if (( Z.ncols() > 0 ) && ( Y.ncols() > 0 ))
            // {
            //     auto  UY = blas::join_row< value_t >( { U, Y } );
            //     auto  ZV = blas::join_row< value_t >( { Z, V } );
                
            //     if ( is_null( matrix ) )
            //         matrix = std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( UY ), std::move( ZV ) );
            //     else
            //     {
            //         auto  T = hpro::TRkMatrix( M.row_is(), M.col_is(), UY, ZV );
                    
            //         hlr::add( value_t(1), T, *matrix, acc, approx );
            //     }// else
            // }// if
            if ( Z.ncols() > 0 )
            {
                if ( is_null( matrix ) )
                    matrix = std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( blas::copy( U ) ), std::move( Z ) );
                else
                {
                    auto  T = hpro::TRkMatrix( M.row_is(), M.col_is(), U, Z );
                    
                    hlr::add( value_t(1), T, *matrix, acc, approx );
                }// else
            }// if
            
            if ( Y.ncols() > 0 )
            {
                if ( is_null( matrix ) )
                    matrix = std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( Y ), std::move( blas::copy( V ) ) );
                else
                {
                    auto  T = hpro::TRkMatrix( M.row_is(), M.col_is(), Y, V );
                    
                    hlr::add( value_t(1), T, *matrix, acc, approx );
                }// else
            }// if

            HLR_UNIFORM_ACCU_TOC( add_accu );
        }// if

        //
        // handle remaining computable updates, i.e., one factor is a leaf block
        //

        auto  BC = std::unique_ptr< hpro::TBlockMatrix >(); // for recursive handling

        HLR_UNIFORM_ACCU_TIC( eval_rest );
            
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
                if ( ! ( is_lowrank( M ) || matrix::is_lowrankS( M ) || matrix::is_uniform_lowrank( M ) ) )
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
            }// if
            else
            {
                //
                // compute update (either A or B is a leaf)
                //

                auto  T = std::unique_ptr< hpro::TMatrix >();

                if ( handle_dense ||
                     is_dense_all( A, B ) ||
                     ( is_blocked( A ) && is_dense(   B ) ) ||
                     ( is_dense(   A ) && is_blocked( B ) ))
                    T = std::make_unique< hpro::TDenseMatrix >( A->row_is( op_A ), B->col_is( op_B ), hpro::value_type_v< value_t > );
                else
                {
                    std::cout << "!!! : " << M.id() << " : " << A->typestr() << " x " << B->typestr() << std::endl;
                    T = std::make_unique< hpro::TRkMatrix >( A->row_is( op_A ), B->col_is( op_B ), hpro::value_type_v< value_t > );
                }// else

                hlr::multiply< value_t >( alpha, op_A, *A, op_B, *B, *T, acc, approx );

                // if ( false & ( is_lowrank( *T ) && matrix::is_uniform_lowrank( M ) ))
                // {
                //     HLR_ASSERT( ( row_basis.nrows() > 0 ) && ( col_basis.nrows() > 0 ) );
                        
                //     // check error when representing in local basis
                //     auto  R  = ptrcast( T.get(), hpro::TRkMatrix );
                //     auto  U  = blas::copy( blas::mat_U< value_t >( R ) );
                //     auto  V  = blas::copy( blas::mat_V< value_t >( R ) );

                //     io::matlab::write( U, "U" );
                //     io::matlab::write( V, "V" );

                //     io::matlab::write( row_basis, "W" );
                //     io::matlab::write( col_basis, "X" );
                    
                //     auto  RU = blas::matrix< value_t >();
                //     auto  RV = blas::matrix< value_t >();

                //     blas::qr( U, RU );
                //     blas::qr( V, RV );

                //     // T = W W' U (X X' V)' = W W' U V' X X', with W being the row basis and X the column basis
                //     auto  TU  = blas::prod( blas::adjoint( row_basis ), U );
                //     auto  TV  = blas::prod( blas::adjoint( col_basis ), V );
                //     auto  S1  = blas::prod( RU, blas::adjoint( RV ) );
                //     auto  S2  = blas::prod( TU, S1 );
                //     auto  S   = blas::prod( S2, blas::adjoint( TV ) );

                //     auto  M1  = blas::prod( blas::mat_U< value_t >( R ), blas::adjoint( blas::mat_V< value_t >( R ) ) );
                //     auto  T2  = blas::prod( row_basis, S );
                //     auto  M2  = blas::prod( T2, blas::adjoint( col_basis ) );

                //     io::matlab::write( M1, "M1" );
                //     io::matlab::write( M2, "M2" );
                    
                //     blas::add( value_t(-1), M1, M2 );
                //     std::cout << M.id() << " : " << boost::format( "%.4e" ) % blas::norm_F( M2 )
                //               << " / " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
                // }// if
                
                //
                // apply update to accumulator
                //
            
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
            }// else
        }// for

        HLR_UNIFORM_ACCU_TOC( eval_rest );
        
        //
        // now handle recursive updates if M is a leaf block
        //
    
        if ( ! is_null( BC ) )
        {
            HLR_UNIFORM_ACCU_TIC( eval_rec );
            
            //
            // TODO: try with empty sub_mat, don't release U and add sub results later
            //
        
            //
            // first, split update matrix into subblock updates
            // (to release matrix before recursion)
            //

            auto  sub_accu = restrict< value_t >( *BC );

            matrix.reset( nullptr );
        
            //
            // apply recursive updates
            //
        
            for ( uint  i = 0; i < BC->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                {
                    sub_accu(i,j).eval( alpha, *BC->block(i,j), acc, approx );

                    // replace block in BC by accumulator matrix for agglomeration below
                    BC->delete_block( i, j );
                    BC->set_block( i, j, sub_accu(i,j).release_matrix() );
                }// for
            }// for

            //
            // finally convert subblocks to single low-rank matrix for new accumulated updates
            //

            if ( handle_dense )
                matrix = seq::matrix::convert_to_dense< value_t >( *BC );
            else
                matrix = seq::matrix::convert_to_lowrank( *BC, acc, approx );

            HLR_UNIFORM_ACCU_TOC( eval_rec );
        }// if
    }

    //
    // return true if given matrix is dense
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
           const uniform_map_t &    rowmap,
           const uniform_map_t &    colmap )
           // hpro::TMatrix &          REF )
{
    //
    // evaluate all computable updates to M
    //

    HLR_UNIFORM_ACCU_TIC( eval );
    
    accu.eval( value_t(1), M, acc, approx );
    
    HLR_UNIFORM_ACCU_TOC( eval );

    //
    // recurse
    //

    if ( is_blocked( M ) )
    {
        auto  BM       = ptrcast( &M, hpro::TBlockMatrix );
        auto  sub_accu = accu.restrict< value_t >( *BM );

        accu.clear_matrix();

        for ( uint  i = 0; i < BM->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BM->nblock_cols(); ++j )
            {
                if ( is_null( BM->block( i, j ) ) )
                    continue;

                multiply( alpha, * BM->block( i, j ), sub_accu( i, j ),
                          acc, approx, rowmap, colmap );
            }// for
        }// for
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        //
        // update local matrix as standard low-rank matrix
        //

        auto  UM = ptrcast( &M, matrix::uniform_lrmatrix< value_t > );
        auto  R  = hpro::TRkMatrix( UM->row_is(), UM->col_is(),
                                    std::move( blas::prod( UM->row_basis(), UM->coeff() ) ),
                                    std::move( blas::copy( UM->col_basis() ) ) );

        HLR_UNIFORM_ACCU_TIC( apply_uni );
        
        // no recursive updates left, apply accumulated updates and solve
        accu.apply( alpha, R, acc, approx );

        HLR_UNIFORM_ACCU_TOC( apply_uni );
        
        //
        // now replace M by R and update row/column bases
        //

        HLR_UNIFORM_ACCU_TIC( basis );
        
        auto  W  = blas::mat_U< value_t >( R );
        auto  X  = blas::mat_V< value_t >( R );
        auto  RW = blas::matrix< value_t >();
        auto  RX = blas::matrix< value_t >();

        blas::qr( W, RW );
        blas::qr( X, RX );

        auto  T = blas::prod( RW, blas::adjoint( RX ) );
                    
        hlr::uniform::detail::update_row_col_basis( *UM, W, T, X, acc, approx, rowmap, colmap );

        HLR_UNIFORM_ACCU_TOC( basis );
        
        // // DEBUG {
        // {
        //     auto  D1 = matrix::convert_to_dense< value_t >( M );
        //     auto  D2 = matrix::convert_to_dense< value_t >( REF );
            
        //     hlr::add( value_t(-1), *D2, *D1 );
        //     std::cout << "ref error " << M.id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *D1 ) / norm::frobenius( *D2 ) ) << std::endl;
        // }
        // // DEBUG }
    }// if
    else
    {
        HLR_UNIFORM_ACCU_TIC( apply_dense );
        
        accu.apply( alpha, M, acc, approx );

        HLR_UNIFORM_ACCU_TOC( apply_dense );
        
        // // DEBUG {
        // {
        //     auto  D1 = matrix::convert_to_dense< value_t >( M );
        //     auto  D2 = matrix::convert_to_dense< value_t >( REF );
            
        //     hlr::add( value_t(-1), *D2, *D1 );
        //     std::cout << "ref error " << M.id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *D1 ) / norm::frobenius( *D2 ) ) << std::endl;
        // }
        // // DEBUG }
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
    HLR_UNIFORM_ACCU_TIC( eval );

    accu.eval( value_t(1), M, acc, approx );

    HLR_UNIFORM_ACCU_TOC( eval );
    
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
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        //
        // update and solve local matrix
        //

        auto  UM = ptrcast( &M, matrix::uniform_lrmatrix< value_t > );
        auto  R  = hpro::TRkMatrix( UM->row_is(), UM->col_is(), std::move( blas::prod( UM->row_basis(), UM->coeff() ) ), std::move( blas::copy( UM->col_basis() ) ) );

        HLR_UNIFORM_ACCU_TIC( apply );
        
        // no recursive updates left, apply accumulated updates and solve
        accu.apply( value_t(-1), R, acc, approx );

        HLR_UNIFORM_ACCU_TOC( apply );
        
        hlr::solve_lower_tri< value_t >( side, diag, L, R, acc, approx );

        // // DEBUG {
        // {
        //     auto  D1 = matrix::convert_to_dense< value_t >( R );
        //     auto  D2 = matrix::convert_to_dense< value_t >( REF );
            
        //     hlr::add( value_t(-1), *D2, *D1 );
        //     std::cout << "ref error " << M.id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *D1 ) / norm::frobenius( *D2 ) ) << std::endl;
        // }
        // // DEBUG }

        //
        // now replace M by R and update row/column bases
        //

        HLR_UNIFORM_ACCU_TIC( basis );
        
        auto  W  = blas::mat_U< value_t >( R );
        auto  X  = blas::mat_V< value_t >( R );
        auto  RW = blas::matrix< value_t >();
        auto  RX = blas::matrix< value_t >();

        blas::qr( W, RW );
        blas::qr( X, RX );

        auto  T = blas::prod( RW, blas::adjoint( RX ) );
                    
        hlr::uniform::detail::update_row_col_basis( *UM, W, T, X, acc, approx, rowmap, colmap );

        HLR_UNIFORM_ACCU_TOC( basis );
        
        // // DEBUG {
        // {
        //     auto  D1 = matrix::convert_to_dense< value_t >( M );
        //     auto  D2 = matrix::convert_to_dense< value_t >( REF );
            
        //     hlr::add( value_t(-1), *D2, *D1 );
        //     std::cout << "ref error " << M.id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *D1 ) / norm::frobenius( *D2 ) ) << std::endl;
        // }
        // // DEBUG }
    }// if
    else
    {
        accu.apply( value_t(-1), M, acc, approx );

        hlr::solve_lower_tri< value_t >( side, diag, L, M, acc, approx );
        
        // // DEBUG {
        // {
        //     auto  D1 = matrix::convert_to_dense< value_t >( M );
        //     auto  D2 = matrix::convert_to_dense< value_t >( REF );
            
        //     hlr::add( value_t(-1), *D2, *D1 );
        //     std::cout << "ref error " << M.id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *D1 ) / norm::frobenius( *D2 ) ) << std::endl;
        // }
        // // DEBUG }
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

    HLR_UNIFORM_ACCU_TIC( eval );
    
    // apply computable updates
    accu.eval( value_t(1), M, acc, approx );

    HLR_UNIFORM_ACCU_TOC( eval );
    
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
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        //
        // update and solve local matrix
        //

        auto  UM = ptrcast( &M, matrix::uniform_lrmatrix< value_t > );
        auto  R  = hlr::matrix::convert_to_lowrank< value_t >( M );

        HLR_UNIFORM_ACCU_TIC( apply );
        
        // no recursive updates left, apply accumulated updates and solve
        accu.apply( value_t(-1), *R, acc, approx );

        HLR_UNIFORM_ACCU_TOC( apply );

        hlr::solve_upper_tri< value_t >( side, diag, U, *R, acc, approx );

        //
        // now replace M by R and update row/column bases
        //

        HLR_UNIFORM_ACCU_TIC( basis );
        
        auto  W  = blas::mat_U< value_t >( *R );
        auto  X  = blas::mat_V< value_t >( *R );
        auto  RW = blas::matrix< value_t >();
        auto  RX = blas::matrix< value_t >();

        blas::qr( W, RW );
        blas::qr( X, RX );

        auto  T = blas::prod( RW, blas::adjoint( RX ) );
        
        hlr::uniform::detail::update_row_col_basis( *UM, W, T, X, acc, approx, rowmap, colmap );
        
        HLR_UNIFORM_ACCU_TOC( basis );
    }// if
    else
    {
        accu.apply( value_t(-1), M, acc, approx );

        hlr::solve_upper_tri< value_t >( side, diag, U, M, acc, approx );
        
        // // DEBUG {
        // {
        //     auto  D1 = matrix::convert_to_dense< value_t >( M );
        //     auto  D2 = matrix::convert_to_dense< value_t >( REF );
            
        //     hlr::add( value_t(-1), *D2, *D1 );
        //     std::cout << "ref error " << M.id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *D1 ) / norm::frobenius( *D2 ) ) << std::endl;
        // }
        // // DEBUG }
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

    HLR_UNIFORM_ACCU_TIC( eval );
    
    accu.eval( value_t(1), A, acc, approx );

    HLR_UNIFORM_ACCU_TOC( eval );

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

        // // DEBUG {
        // {
        //     auto  D1 = matrix::convert_to_dense< value_t >( A );
        //     auto  D2 = matrix::convert_to_dense< value_t >( REF );

        //     hlr::add( value_t(-1), *D2, *D1 );
        //     std::cout << "ref error " << A.id() << " : " << boost::format( "%.4e" ) % ( norm::frobenius( *D1 ) / norm::frobenius( *D2 ) ) << std::endl;
        // }
        // // DEBUG }
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

}// namespace detail

}}}// namespace hlr::uniform::accu

#endif // __HLR_ARITH_DETAIL_UNIFORM_ACCU_HH

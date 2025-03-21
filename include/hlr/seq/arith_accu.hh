#ifndef __HLR_SEQ_ARITH_ACCU_HH
#define __HLR_SEQ_ARITH_ACCU_HH
//
// Project     : HLR
// Module      : seq/arith_accu.hh
// Description : sequential arithmetic functions using accumulators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/arith/multiply.hh"
#include "hlr/arith/add.hh"
#include "hlr/arith/solve.hh"
#include "hlr/utils/tensor.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/utils/trace.hh"

namespace hlr { namespace seq { namespace accu {

////////////////////////////////////////////////////////////////////////////////
//
// accumulator based matrix multiplication
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

template < typename value_t >
struct accumulator
{
    using  matrix_t = Hpro::TMatrix< value_t >;
    
    //
    // represents an update, i.e., matrix product
    //
    struct update
    {
        const matop_t     op_A;
        const matrix_t *  A;
        const matop_t     op_B;
        const matrix_t *  B;
        const matop_t     op_D;
        const matrix_t *  D;
    };
    
    // represents set of updates
    using  update_list = std::list< update >;

    // accumulated computed updates
    std::unique_ptr< matrix_t >   matrix;

    // accumulated pending (recursive) updates
    update_list                   pending;

    //
    // ctors
    //

    accumulator ()
    {}
    
    accumulator ( std::unique_ptr< matrix_t > &&  amatrix,
                  update_list &&                  apending )
            : matrix(  std::move( amatrix  ) )
            , pending( std::move( apending ) )
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
    matrix_t *
    release_matrix ()
    {
        return matrix.release();
    }

    //
    // add update A×B
    //
    void
    add_update ( const matrix_t &  A,
                 const matrix_t &  B )
    {
        pending.push_back( { apply_normal, &A, apply_normal, &B, apply_normal, nullptr } );
    }

    void
    add_update ( const matop_t     op_A,
                 const matrix_t &  A,
                 const matop_t     op_B,
                 const matrix_t &  B )
    {
        pending.push_back( { op_A, &A, op_B, &B, apply_normal, nullptr } );
    }
    
    void
    add_update ( const matrix_t &  A,
                 const matrix_t &  D,
                 const matrix_t &  B )
    {
        pending.push_back( { apply_normal, &A, apply_normal, &B, apply_normal, &D } );
    }

    //
    // apply accumulated updates and free accumulator matrix
    //
    template < typename approx_t >
    void
    apply ( const value_t     alpha,
            matrix_t &        M,
            const accuracy &  acc,
            const approx_t &  approx )
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
        auto  U_ij = std::unique_ptr< matrix_t >();
        auto  P_ij = update_list();
        
        if ( ! is_null( matrix ) )
        {
            HLR_ASSERT( ! is_null( M.block( i, j ) ) );
            
            U_ij = hlr::matrix::restrict( *matrix, M.block( i, j )->block_is() );

            // prefer dense if destination is dense
            if ( check_dense( *M.block( i, j ) ) && ! matrix::is_dense( *U_ij ) )
                U_ij = std::move( hlr::matrix::convert_to_dense< value_t >( *U_ij ) );
        }// if

        for ( auto  [ op_A, A, op_B, B, op_D, D ] : pending )
        {
            // filter out non-recursive updates
            if ( ! hlr::is_blocked_all( A, B ) )
                continue;

            if ( ! ( is_null( D ) || is_blocked( D ) ))
                continue;
                                            
            auto  BA = cptrcast( A, Hpro::TBlockMatrix< value_t > );
            auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );
            auto  BD = cptrcast( D, Hpro::TBlockMatrix< value_t > );
                                            
            for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
            {
                auto  A_il = BA->block( i, l, op_A );
                auto  B_lj = BB->block( l, j, op_B );
                auto  D_ll = ( is_null( D ) ? nullptr : BD->block( l, l, op_D ) );
                                                
                if ( is_null_any( A_il, B_lj ) )
                    continue;
                                                
                P_ij.push_back( { op_A, A_il, op_B, B_lj, op_D, D_ll } );
            }// for
        }// for

        return accumulator{ std::move( U_ij ), std::move( P_ij ) };
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
    eval ( const value_t     alpha,
           const matrix_t &  M,
           const accuracy &  acc,
           const approx_t &  approx )
    {
        auto  BC = std::unique_ptr< Hpro::TBlockMatrix< value_t > >(); // for recursive handling

        //
        // handle all, actually computable updates, i.e., one factor is a leaf block
        //

        bool  handle_dense = check_dense( M );

        for ( auto  [ op_A, A, op_B, B, op_D, D ] : pending )
        {
            if ( is_null( D ) )
            {
                if ( ! hlr::is_blocked_all( A, B ) && ! ( matrix::is_lowrank_any( A, B ) || matrix::is_lowrank_sv_any( A, B ) ) )
                {
                    handle_dense = true;
                    break;
                }// if
            }// if
            else
            {
                if ( ! hlr::is_blocked_all( A, B, D ) && ! ( matrix::is_lowrank_any( A, B ) || matrix::is_lowrank_sv_any( A, B ) ) )
                {
                    handle_dense = true;
                    break;
                }// if
            }// else
        }// for
        
        for ( auto  [ op_A, A, op_B, B, op_D, D ] : pending )
        {
            if (( is_null( D ) && hlr::is_blocked_all( A, B, &M ) ) || hlr::is_blocked_all( A, B, D, &M ) )
                continue;
        
            if (( is_null( D ) && hlr::is_blocked_all( A, B ) ) || hlr::is_blocked_all( A, B, D ) )
            {
                //
                // if M is a leaf and A _and_ B are blocked, a temporary matrix
                // is created for further recursive update handling
                //

                if ( ! is_null( BC ) )
                    continue;
                
                // TODO: non low-rank M
                HLR_ASSERT( matrix::is_lowrank( M ) || matrix::is_lowrank_sv( M ) );
                
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
            }// if
            else
            {
                //
                // compute update (either A or B is a leaf)
                //

                auto  T = std::unique_ptr< matrix_t >();

                if ( is_null( D ) )
                    T = hlr::multiply< value_t >( alpha, op_A, *A, op_B, *B );
                else
                    T = hlr::multiply_diag< value_t >( alpha, op_A, *A, op_D, *D, op_B, *B );

                if ( handle_dense && ! matrix::is_dense( *T ) )
                    T = matrix::convert_to_dense< value_t >( *T );
                
                //
                // apply update to accumulator
                //
            
                if ( is_null( matrix ) )
                {
                    matrix = std::move( T );
                }// if
                else if ( ! matrix::is_dense( *matrix ) && matrix::is_dense( *T ) )
                {
                    // prefer dense format to avoid unnecessary truncations
                    hlr::add( value_t(1), *matrix, *T );
                    matrix = std::move( T );
                }// if
                else
                {
                    hlr::add( value_t(1), *T, *matrix, acc, approx );
                }// else
            }// else
        }// for

        //
        // now handle recursive updates if M is a leaf block
        //
    
        if ( ! is_null( BC ) )
        {
            //
            // TODO: try with empty sub_mat, don't release U and add sub results later
            //
        
            //
            // first, split update matrix into subblock updates
            // (to release matrix before recursion)
            //

            auto  sub_accu = restrict( *BC );

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
        }// if
    }

    //
    // return true if given matrix is dense
    //
    bool
    check_dense ( const matrix_t &  M ) const
    {
        // return false;
        if ( matrix::is_dense( M ) )
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
                    if ( ! is_null( B->block( i, j ) ) && ! matrix::is_dense( B->block( i, j ) ) )
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
// compute C = C + α op( A ) op( B ) where A and B are provided as accumulated updates
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t              alpha,
           Hpro::TMatrix< value_t > & C,
           accumulator< value_t > &   accu,
           const accuracy &           acc,
           const approx_t &           approx )
{
    //
    // first handle all computable updates to C, including if C is non-blocked
    //

    trace::region_start( "eval" );
    
    accu.eval( alpha, C, acc, approx );
    
    trace::region_end( "eval" );
    
    //
    // now handle recursive updates
    //
    
    if ( is_blocked( C ) )
    {
        auto  BC = ptrcast(  &C, Hpro::TBlockMatrix< value_t > );

        //
        // first, split update U into subblock updates
        // (to release U before recursion and by that avoid
        //  memory consumption dependent on hierarchy depth)
        //

        auto  sub_accu = accu.restrict( *BC );

        accu.clear_matrix();

        //
        // now apply recursive multiplications, e.g.,
        // collect all sub-products and recurse
        //
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                multiply< value_t >( alpha, *BC->block(i,j), sub_accu(i,j), acc, approx );
    }// if
    else 
    {
        trace::region_start( "apply" );
        
        // apply accumulated updates
        accu.apply( alpha, C, acc, approx );

        trace::region_end( "apply" );
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B,
           Hpro::TMatrix< value_t > &        C,
           const accuracy &                  acc,
           const approx_t &                  approx )
{
    auto  accu = detail::accumulator< value_t >();

    accu.add_update( op_A, A, op_B, B );
    detail::multiply< value_t >( alpha, C, accu, acc, approx );
}

////////////////////////////////////////////////////////////////////////////////
//
// accumulator based LU factorization
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

template < typename value_t,
           typename approx_t >
void
solve_diag ( const eval_side_t                side,
             const diag_type_t                diag,
             const matop_t                    op_D,
             const Hpro::TMatrix< value_t > & D,
             Hpro::TMatrix< value_t > &       M,
             accumulator< value_t > &         accu,
             const accuracy &                 acc,
             const approx_t &                 approx )
{
    //
    // apply computable updates
    //
    
    trace::region_start( "eval" );

    accu.eval( value_t(1), M, acc, approx );

    trace::region_end( "eval" );
    
    if ( hlr::is_blocked_all( D, M ) )
    {
        auto  BD = cptrcast( &D, Hpro::TBlockMatrix< value_t > );
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
                const auto  D_ii = BD->block( i, i, op_D );
            
                for ( uint j = 0; j < BM->nblock_cols(); ++j )
                    solve_diag< value_t >( side, diag, op_D, *D_ii, *BM->block(i,j), sub_accu(i,j), acc, approx );
            }// for
        }// if
        else
        {
            for ( uint j = 0; j < BM->nblock_cols(); ++j )
            {
                const auto  D_jj = BD->block( j, j, op_D );
            
                for ( uint i = 0; i < BM->nblock_rows(); ++i )
                    solve_diag< value_t >( side, diag, op_D, *D_jj, *BM->block(i,j), sub_accu(i,j), acc, approx );
            }// for
        }// else
    }// if
    else
    {
        // no recursive updates left, apply accumulated updates and solve
        trace::region_start( "apply" );
        
        accu.apply( value_t(-1), M, acc, approx );

        trace::region_end( "apply" );

        hlr::solve_diag< value_t >( side, diag, op_D, D, M, acc, approx );
    }// if
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  L,
                  Hpro::TMatrix< value_t > &        M,
                  accumulator< value_t > &          accu,
                  const accuracy &                  acc,
                  const approx_t &                  approx )
{
    //
    // apply computable updates
    //
    
    trace::region_start( "eval" );

    accu.eval( value_t(1), M, acc, approx );

    trace::region_end( "eval" );
    
    if ( hlr::is_blocked_all( L, M ) )
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
            
                for ( uint j = 0; j < BM->nblock_cols(); ++j )
                    solve_lower_tri< value_t >( side, diag, *L_ii, *BM->block(i,j), sub_accu(i,j), acc, approx );

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
    else
    {
        // no recursive updates left, apply accumulated updates and solve
        trace::region_start( "apply" );

        // if ( ! is_null( accu.matrix ) )
        //     std::cout << "apply " << M.id() << " : " << norm::frobenius( M ) << ", " << norm::frobenius( * (accu.matrix) ) << std::endl;
        
        accu.apply( value_t(-1), M, acc, approx );

        trace::region_end( "apply" );

        // std::cout << "apply " << M.id() << " : " << norm::frobenius( M ) << std::endl;
        
        hlr::solve_lower_tri< value_t >( side, diag, L, M, acc, approx );

        // std::cout << "lower " << M.id() << " : " << norm::frobenius( M ) << std::endl;
    }// if
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  U,
                  Hpro::TMatrix< value_t > &        M,
                  accumulator< value_t > &          accu,
                  const accuracy &                  acc,
                  const approx_t &                  approx )
{
    // apply computable updates
    trace::region_start( "eval" );

    accu.eval( value_t(1), M, acc, approx );

    trace::region_end( "eval" );
    
    if ( hlr::is_blocked_all( U, M ) )
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
            
                for ( uint i = 0; i < BM->nblock_rows(); ++i )
                    solve_upper_tri< value_t >( side, diag, *U_jj, *BM->block( i, j ), sub_accu(i,j), acc, approx );
            
                for ( uint  k = j+1; k < BM->nblock_cols(); ++k )
                    for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                        sub_accu(i,k).add_update( *BM->block(i,j), *BU->block(j,k) );
            }// for
        }// else
    }// if
    else
    {
        // no recursive updates left, apply accumulated updates and solve
        trace::region_start( "apply" );
        
        accu.apply( value_t(-1), M, acc, approx );

        trace::region_end( "apply" );
        
        // std::cout << "apply " << M.id() << " : " << norm::frobenius( M ) << std::endl;

        hlr::solve_upper_tri< value_t >( side, diag, U, M, acc, approx );

        // std::cout << "upper " << M.id() << " : " << norm::frobenius( M ) << std::endl;
    }// if
}

template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &  M,
     accumulator< value_t > &    accu,
     const accuracy &            acc,
     const approx_t &            approx )
{
    //
    // evaluate all computable updates to M
    //

    trace::region_start( "eval" );

    accu.eval( value_t(1), M, acc, approx );

    trace::region_end( "eval" );
    
    //
    // (recursive) LU factorization
    //
    
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( &M, Hpro::TBlockMatrix< value_t > );

        //
        // first, split accumulated updates U and recursive updates upd_rec
        // into subblock updates
        // - to release U before recursion and by that avoid memory
        //   consumption dependent on hierarchy depth
        //

        auto  sub_accu = accu.restrict( *BM );

        accu.clear_matrix();

        //
        // recursive LU factorization but add updates to accumulator
        // instead of applying them
        //
        
        for ( uint  i = 0; i < std::min( BM->nblock_rows(), BM->nblock_cols() ); ++i )
        {
            auto  B_ii = BM->block( i, i );

            lu< value_t >( *B_ii, sub_accu(i,i), acc, approx );

            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                solve_upper_tri< value_t >( from_right, general_diag, *B_ii, *BM->block( j, i ), sub_accu(j,i), acc, approx );

            for ( uint  j = i+1; j < BM->nblock_cols(); ++j )
                solve_lower_tri< value_t >( from_left, unit_diag, *B_ii, *BM->block( i, j ), sub_accu(i,j), acc, approx );

            // add updates to sub lists
            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                for ( uint  l = i+1; l < BM->nblock_cols(); ++l )
                    sub_accu(j,l).add_update( *BM->block( j, i ), *BM->block( i, l ) );
        }// for
    }// if
    else
    {
        //
        // no recursive updates left, apply accumulated updates
        // and factorize
        //

        trace::region_start( "apply" );
        
        accu.apply( value_t(-1), M, acc, approx );

        trace::region_end( "apply" );

        // std::cout << "apply " << M.id() << " : " << norm::frobenius( M ) << std::endl;
        
        if ( matrix::is_dense( M ) )
        {
            auto  D              = ptrcast( &M, matrix::dense_matrix< value_t > );
            auto  DD             = D->mat();
            auto  was_compressed = D->is_compressed();
            
            blas::invert( DD );

            if ( was_compressed )
                D->set_matrix( std::move( DD ), acc );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );

        // std::cout << "lu " << M.id() << " : " << norm::frobenius( M ) << std::endl;
    }// else
}

template < typename value_t,
           typename approx_t >
void
ldu ( Hpro::TMatrix< value_t > & M,
      accumulator< value_t > &   accu,
      const accuracy &           acc,
      const approx_t &           approx )
{
    //
    // evaluate all computable updates to M
    //

    trace::region_start( "eval" );

    accu.eval( value_t(1), M, acc, approx );

    trace::region_end( "eval" );
    
    //
    // (recursive) LU factorization
    //
    
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( &M, Hpro::TBlockMatrix< value_t > );

        //
        // first, split accumulated updates U and recursive updates upd_rec
        // into subblock updates
        // - to release U before recursion and by that avoid memory
        //   consumption dependent on hierarchy depth
        //

        auto  sub_accu = accu.restrict( *BM );

        accu.clear_matrix();

        //
        // recursive LU factorization but add updates to accumulator
        // instead of applying them
        //
        
        for ( uint  i = 0; i < std::min( BM->nblock_rows(), BM->nblock_cols() ); ++i )
        {
            auto  B_ii = BM->block( i, i );

            lu< value_t >( *B_ii, sub_accu(i,i), acc, approx );

            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
            {
                solve_upper_tri< value_t >( from_right, unit_diag, *B_ii, *BM->block( j, i ), sub_accu(j,i), acc, approx );
                solve_diag< value_t >(      from_right, general_diag, apply_normal, *B_ii, *BM->block( j, i ), sub_accu(j,i), acc, approx );
            }// for

            for ( uint  j = i+1; j < BM->nblock_cols(); ++j )
            {
                solve_lower_tri< value_t >( from_left, unit_diag, *B_ii, *BM->block( i, j ), sub_accu(i,j), acc, approx );
                solve_diag< value_t >(      from_left, general_diag, apply_normal, *B_ii, *BM->block( i, j ), sub_accu(i,j), acc, approx );
            }// for

            // add updates to sub lists
            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                for ( uint  l = i+1; l < BM->nblock_cols(); ++l )
                    sub_accu(j,l).add_update( *BM->block( j, i ), *B_ii, *BM->block( i, l ) );
        }// for
    }// if
    else
    {
        //
        // no recursive updates left, apply accumulated updates
        // and factorize
        //

        trace::region_start( "apply" );
        
        accu.apply( value_t(-1), M, acc, approx );

        trace::region_end( "apply" );
        
        if ( matrix::is_dense( M ) )
        {
            auto  D  = ptrcast( &M, matrix::dense_matrix< value_t > );
            auto  DD = D->mat();

            blas::invert( DD );
        
            if ( D->is_compressed() )
                D->set_matrix( std::move( DD ), acc );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &  M,
     const accuracy &            acc,
     const approx_t &            approx )
{
    auto  accu = detail::accumulator< value_t >();
    
    detail::lu< value_t >( M, accu, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
ldu ( Hpro::TMatrix< value_t > &  M,
      const accuracy &            acc,
      const approx_t &            approx )
{
    auto  accu = detail::accumulator< value_t >();
    
    detail::ldu< value_t >( M, accu, acc, approx );
}

}}}// namespace hlr::seq::accu

#endif // __HLR_SEQ_ARITH_ACCU_HH

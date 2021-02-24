#ifndef __HLR_TBB_ACCUMULATOR_HH
#define __HLR_TBB_ACCUMULATOR_HH
//
// Project     : HLR
// File        : accumulator.hh
// Description : implements update accumulator for H-arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>

#include <hlr/arith/multiply.hh>
#include <hlr/arith/add.hh>
#include "hlr/matrix/accumulator_base.hh"
#include "hlr/matrix/restrict.hh"
#include <hlr/utils/checks.hh>
#include "hlr/utils/tensor.hh"
#include "hlr/tbb/matrix.hh"

namespace hlr { namespace tbb { namespace matrix {

struct accumulator : public hlr::matrix::accumulator_base
{
    //
    // ctors
    //

    accumulator ()
    {}
    
    accumulator ( std::unique_ptr< hpro::TMatrix > &&  amatrix,
                  update_list &&                       apending )
            : hlr::matrix::accumulator_base( std::move( amatrix ), std::move( apending ) )
    {}
    
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

        return accumulator{ std::move( U_ij ), std::move( P_ij ) };
    }

    //
    // return restriction of updates to all sub blocks of given block matrix
    //
    tensor2< accumulator >
    restrict ( const hpro::TBlockMatrix &  M ) const
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
    template < typename value_t,
               typename approx_t >
    void
    eval ( const value_t            alpha,
           const hpro::TMatrix &    M,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
    {
        std::unique_ptr< hpro::TBlockMatrix >  BC; // for recursive handling

        //
        // handle all, actually computable updates, i.e., one factor is a leaf block
        //
    
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
                HLR_ASSERT( is_lowrank( M ) );
                
                auto  BA = cptrcast( A, hpro::TBlockMatrix );
                auto  BB = cptrcast( B, hpro::TBlockMatrix );
                
                BC = std::make_unique< hpro::TBlockMatrix >( A->row_is( op_A ), B->col_is( op_B ) );

                BC->set_block_struct( BA->nblock_rows( op_A ), BB->nblock_cols( op_B ) );

                for ( uint  i = 0; i < BC->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                    {
                        HLR_ASSERT( ! is_null_any( BA->block( i, 0, op_A ), BB->block( 0, j, op_B ) ) );
                        
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

                auto  T = multiply< value_t >( alpha, op_A, A, op_B, B );

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
        
            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                                0, BC->nblock_cols() ),
                [&,alpha] ( const auto & r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                        for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                        {
                            sub_accu(i,j).eval( alpha, *BC->block(i,j), acc, approx );

                            // replace block in BC by accumulator matrix for agglomeration below
                            BC->delete_block( i, j );
                            BC->set_block( i, j, sub_accu(i,j).release_matrix() );
                        }// for
                } );

            //
            // finally convert subblocks to single low-rank matrix for new accumulated updates
            //

            matrix = tbb::matrix::convert_to_lowrank( *BC, acc, approx );
        }// if
    }
};

}}} // namespace hlr::tbb::matrix

#endif // __HLR_TBB_ACCUMULATOR_HH

#ifndef __HLR_ARITH_MULVEC_HH
#define __HLR_ARITH_MULVEC_HH
//
// Project     : HLR
// Module      : mul_vec
// Description : matrix-vector multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/h2.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/utils/log.hh>
#include <hlr/utils/hash.hh>

namespace hlr
{

//
// compute y = y + α op( M ) x for blas::vector
//
template < typename value_t >
void
mul_vec ( const value_t                     alpha,
          const Hpro::matop_t               op_M,
          const Hpro::TMatrix< value_t > &  M,
          const blas::vector< value_t > &   x,
          blas::vector< value_t > &         y )
{
    using namespace hlr::matrix;
    
    // assert( M.ncols( op_M ) == x.length() );
    // assert( M.nrows( op_M ) == y.length() );

    if ( is_blocked( M ) )
    {
        auto        B       = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        const auto  row_ofs = M.row_is( op_M ).first();
        const auto  col_ofs = M.col_is( op_M ).first();
    
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
            
                if ( ! is_null( B_ij ) )
                {
                    auto  x_j = x( B_ij->col_is( op_M ) - col_ofs );
                    auto  y_i = y( B_ij->row_is( op_M ) - row_ofs );
                
                    mul_vec( alpha, op_M, *B_ij, x_j, y_i );
                }// if
            }// for
        }// for
    }// if
    else
        M.apply_add( alpha, x, y, op_M );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// compute y = y + α op( M ) x for scalar vectors
//
template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const Hpro::matop_t                       op_M,
          const Hpro::TMatrix< value_t > &          M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y )
{
    mul_vec( alpha, op_M, M, blas::vec( x ), blas::vec( y ) );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// just dummy implementation
//

template < typename value_t > using  matrix_list_t       = std::deque< const Hpro::TMatrix< value_t > * >;
template < typename value_t > using  cluster_block_map_t = std::unordered_map< indexset, matrix_list_t< value_t >, indexset_hash >;

template < typename value_t >
void
mul_vec_cl ( const value_t                             alpha,
             const matop_t                             op_M,
             const Hpro::TMatrix< value_t > &          M,
             const cluster_block_map_t< value_t > &    blocks,
             const vector::scalar_vector< value_t > &  x,
             vector::scalar_vector< value_t > &        y )
{
    if ( alpha == value_t(0) )
        return;

    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            // WARNING: assuming that by following diagonal blocks all clusters are reached
            auto  B_ii = B->block( i, i );
            
            if ( ! is_null( B_ii ) )
                mul_vec_cl( alpha, op_M, *B_ii, blocks, x, y );
        }// for
    }// if

    if ( ! blocks.contains( M.row_is( op_M ) ) )
        return;

    //
    // compute update with all block in current block row
    //
    
    auto &  mat_list = blocks.at( M.row_is( op_M ) );
    auto    y_j      = blas::vector< value_t >( blas::vec( y ), M.row_is( op_M ) - y.ofs() );
    auto    yt       = blas::vector< value_t >( y_j.length() );

    for ( auto  A : mat_list )
    {
        auto  x_i = blas::vector< value_t >( blas::vec( x ), A->col_is( op_M ) - x.ofs() );
        
        A->apply_add( 1, x_i, yt, op_M );
    }// for

    blas::add( alpha, yt, y_j );
}

template < typename value_t >
void
setup_cluster_block_map ( const matop_t                     op_M,
                          const Hpro::TMatrix< value_t > &  M,
                          cluster_block_map_t< value_t > &  blocks )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( & M, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
                if ( B->block( i, j ) != nullptr )
                    setup_cluster_block_map( op_M, * B->block( i, j ), blocks );
    }// if
    else
    {
        blocks[ M.row_is( op_M ) ].push_back( & M );
    }// else
}

}// namespace hlr

#endif // __HLR_ARITH_MULVEC_HH

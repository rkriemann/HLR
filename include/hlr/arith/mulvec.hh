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
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/utils/log.hh>
#include <hlr/utils/hash.hh>
#include <hlr/utils/flops.hh>

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
// cluster tree oriented version
// (for just dummy seq. implementation)
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// build data structure with joined lowrank/dense blocks per row/column cluster
//
template < typename value_t >
struct cluster_matrix_t
{
    // corresponding index set of cluster
    indexset                           is;
    
    // joined low rank factors
    blas::matrix< value_t >            U;

    // joined dense matrices
    blas::matrix< value_t >            D;

    // son matrices (following cluster tree)
    std::vector< cluster_matrix_t * >  sub_blocks;

    //
    // ctor
    //
    cluster_matrix_t ( const indexset &  ais )
            : is( ais )
    {}
};

//
// 
//
template < typename value_t >
std::pair< std::unique_ptr< cluster_matrix_t< value_t > >,
           std::unique_ptr< cluster_matrix_t< value_t > > >
build_cluster_matrix ( const matop_t                     op_M,
                       const Hpro::TMatrix< value_t > &  M )
{
    //
    // first collect blocks per cluster
    //
    
    auto  row_map = cluster_block_map_t< value_t >();
    auto  col_map = cluster_block_map_t< value_t >();

    setup_cluster_block_map( apply_normal,  M, row_map );
    setup_cluster_block_map( apply_adjoint, M, col_map );

    //
    // convert matrix into cluster matrix structur
    //

    auto  row_CM = std::make_unique< cluster_matrix_t< value_t > >( M.row_is( op_M ) );
    auto  col_CM = std::make_unique< cluster_matrix_t< value_t > >( M.col_is( op_M ) );
        
    convert_to_cluster_matrix( row_map, row_CM );
    convert_to_cluster_matrix( col_map, col_CM );

    return { std::move( row_CM ), std::move( col_CM ) };
}

//
// return FLOPs needed for computing y = y + α op( M ) x
// (implicit vectors)
//
template < typename value_t >
flops_t
mul_vec_flops ( const Hpro::matop_t               op_M,
                const Hpro::TMatrix< value_t > &  M )
{
    using namespace hlr::matrix;
    
    if ( is_blocked( M ) )
    {
        auto        B       = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        const auto  row_ofs = M.row_is( op_M ).first();
        const auto  col_ofs = M.col_is( op_M ).first();
        flops_t     flops   = 0;
    
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
            
                if ( ! is_null( B_ij ) )
                    flops += mul_vec_flops( op_M, *B_ij );
            }// for
        }// for

        return flops;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        const auto  nrows = M.nrows( op_M );
        const auto  ncols = M.ncols( op_M );
        const auto  rank  = cptrcast( &M, matrix::lrmatrix< value_t > )->rank();
        
        // t :=     V^H x
        // y := y + α·U·t
        return FMULS_GEMV( ncols, rank ) + FMULS_GEMV( nrows, rank );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        const auto  nrows = M.nrows( op_M );
        const auto  ncols = M.ncols( op_M );
        
        return FMULS_GEMV( nrows, ncols );
    }// if
    else
        HLR_ERROR( "unsupported matrix type: " + M.typestr() );

    return 0;
}

//
// return size of data involved in computing y = y + α op( M ) x
//
template < typename value_t >
size_t
mul_vec_datasize ( const Hpro::matop_t               op_M,
                   const Hpro::TMatrix< value_t > &  M )
{
    // matrix and vector data
    return M.data_byte_size() + 2 * sizeof( value_t ) * ( M.nrows() + M.ncols() );
}

}// namespace hlr

#endif // __HLR_ARITH_MULVEC_HH

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
#include <hlr/matrix/lrsvmatrix.hh>
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

template < typename value_t >
struct cluster_blocks_t
{
    using matrix = Hpro::TMatrix< value_t >;
        
    // corresponding index set of cluster
    indexset                           is;
    
    // list of associated matrices
    std::list< const matrix * >        M;

    // son matrices (following cluster tree)
    std::vector< cluster_blocks_t * >  sub_blocks;

    // ctor
    cluster_blocks_t ( const indexset &  ais )
            : is( ais )
    {}

    // dtor
    ~cluster_blocks_t ()
    {
        for ( auto  cb : sub_blocks )
            delete cb;
    }
};

namespace detail
{ 

template < typename value_t >
void
build_cluster_blocks ( const matop_t                     op_M,
                       const Hpro::TMatrix< value_t > &  M,
                       cluster_blocks_t< value_t > &     cb )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( & M, Hpro::TBlockMatrix< value_t > );

        if ( cb.sub_blocks.size() == 0 )
            cb.sub_blocks.resize( B->nblock_rows( op_M ) );
        
        for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
        {
            HLR_ASSERT( ! is_null( B->block( i, 0, op_M ) ) );

            if ( is_null( cb.sub_blocks[i] ) )
                cb.sub_blocks[i] = new cluster_blocks_t< value_t >( B->block( i, 0, op_M )->row_is( op_M ) );
        }// for
                
        for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols( op_M ); ++j )
            {
                if ( B->block( i, j, op_M ) != nullptr )
                    build_cluster_blocks( op_M, * B->block( i, j, op_M ), * cb.sub_blocks[i] );
            }// for
        }// for
    }// if
    else
    {
        cb.M.push_back( &M );
    }// else
}

}// namespace detail

template < typename value_t >
std::unique_ptr< cluster_blocks_t< value_t > >
build_cluster_blocks ( const matop_t                     op_M,
                       const Hpro::TMatrix< value_t > &  M )
{
    auto  cb = std::make_unique< cluster_blocks_t< value_t > >( M.row_is( op_M ) );

    detail::build_cluster_blocks( op_M, M, *cb );

    return cb;
}

template < typename value_t >
void
mul_vec_cl ( const value_t                             alpha,
             const matop_t                             op_M,
             const cluster_blocks_t< value_t > &       cb,
             const vector::scalar_vector< value_t > &  x,
             vector::scalar_vector< value_t > &        y )
{
    if ( alpha == value_t(0) )
        return;

    //
    // compute update with all block in current block row
    //

    if ( ! cb.M.empty() )
    {
        auto  y_j = blas::vector< value_t >( blas::vec( y ), cb.is - y.ofs() );
        auto  yt  = blas::vector< value_t >( y_j.length() );
    
        for ( auto  M : cb.M )
        {
            auto  x_i = blas::vector< value_t >( blas::vec( x ), M->col_is( op_M ) - x.ofs() );
            
            M->apply_add( 1, x_i, yt, op_M );
        }// for

        blas::add( alpha, yt, y_j );
    }// if

    //
    // recurse
    //
    
    for ( auto  sub : cb.sub_blocks )
        mul_vec_cl( alpha, op_M, *sub, x, y );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// build data structure with joined lowrank/dense blocks per row/column cluster
//
template < typename value_t >
struct cluster_matrix_t
{
    using matrix = Hpro::TMatrix< value_t >;

    // corresponding index set of cluster
    indexset                           is;
    
    // joined low rank factors
    blas::matrix< value_t >            U;

    // list of dense matrices
    std::list< const matrix * >        D;

    // list of lowrank matrices
    std::list< const matrix * >        R;

    // son matrices (following cluster tree)
    std::vector< cluster_matrix_t * >  sub_blocks;

    // ctor
    cluster_matrix_t ( const indexset &  ais )
            : is( ais )
    {}

    // dtor
    ~cluster_matrix_t ()
    {
        for ( auto  cm : sub_blocks )
            delete cm;
    }
};

namespace detail
{ 

template < typename value_t >
void
build_cluster_matrix ( const matop_t                     op_M,
                       const Hpro::TMatrix< value_t > &  M,
                       cluster_matrix_t< value_t > &     cb )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( & M, Hpro::TBlockMatrix< value_t > );

        if ( cb.sub_blocks.size() == 0 )
            cb.sub_blocks.resize( B->nblock_rows( op_M ) );
        
        for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
        {
            HLR_ASSERT( ! is_null( B->block( i, 0, op_M ) ) );

            if ( is_null( cb.sub_blocks[i] ) )
                cb.sub_blocks[i] = new cluster_matrix_t< value_t >( B->block( i, 0, op_M )->row_is( op_M ) );
        }// for
                
        for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols( op_M ); ++j )
            {
                if ( B->block( i, j, op_M ) != nullptr )
                    build_cluster_matrix( op_M, * B->block( i, j, op_M ), * cb.sub_blocks[i] );
            }// for
        }// for
    }// if
    else if ( matrix::is_lowrank_sv( M ) )
    {
        cb.R.push_back( &M );
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        cb.R.push_back( &M );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        cb.D.push_back( &M );
    }// if
    else
        HLR_ERROR( "unsupported matrix type: " + M.typestr() );
}

template < typename value_t >
void
build_joined_matrix ( const matop_t                  op_M,
                      cluster_matrix_t< value_t > &  cm )
{
    if ( ! cm.R.empty() )
    {
        //
        // determine joined rank
        //

        uint  k = 0;
        
        for ( auto  M : cm.R )
            k += cptrcast( M, matrix::lrmatrix< value_t > )->rank();

        //
        // build joined U factor
        //

        auto  nrows = cm.is.size();
        auto  U     = blas::matrix< value_t >( nrows, k );
        uint  pos   = 0;

        for ( auto  M : cm.R )
        {
            auto  R   = cptrcast( M, matrix::lrmatrix< value_t > );
            auto  UR  = R->U( op_M );
            auto  k_i = R->rank();
            auto  U_i = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k_i - 1 ) );

            blas::copy( UR, U_i );
            pos += k_i;
        }// for

        cm.U = std::move( U );
    }// if

    for ( auto  sub : cm.sub_blocks )
        build_joined_matrix( op_M, *sub );
}

}// namespace detail

template < typename value_t >
std::unique_ptr< cluster_matrix_t< value_t > >
build_cluster_matrix ( const matop_t                     op_M,
                       const Hpro::TMatrix< value_t > &  M )
{
    auto  cm = std::make_unique< cluster_matrix_t< value_t > >( M.row_is( op_M ) );

    detail::build_cluster_matrix( op_M, M, *cm );
    detail::build_joined_matrix( op_M, *cm );
    
    return cm;
}

template < typename value_t >
void
mul_vec_cl ( const value_t                             alpha,
             const matop_t                             op_M,
             const cluster_matrix_t< value_t > &       cm,
             const vector::scalar_vector< value_t > &  x,
             vector::scalar_vector< value_t > &        y )
{
    if ( alpha == value_t(0) )
        return;

    //
    // compute update with all block in current block row
    //

    if ( ! cm.D.empty() || ! cm.R.empty() )
    {
        auto  y_j = blas::vector< value_t >( blas::vec( y ), cm.is - y.ofs() );
        auto  yt  = blas::vector< value_t >( y_j.length() );
    
        if ( ! cm.D.empty() )
        {
            for ( auto  M : cm.D )
            {
                auto  x_i = blas::vector< value_t >( blas::vec( x ), M->col_is( op_M ) - x.ofs() );
                
                M->apply_add( 1, x_i, yt, op_M );
            }// for
        }// if

        if ( ! cm.R.empty() )
        {
            auto  t   = blas::vector< value_t >( cm.U.ncols() );
            uint  pos = 0;
            
            for ( auto  M : cm.R )
            {
                auto  R   = cptrcast( M, matrix::lrmatrix< value_t > );
                auto  VR  = R->V( op_M );
                auto  k_i = R->rank();
                auto  x_i = blas::vector< value_t >( blas::vec( x ), M->col_is( op_M ) - x.ofs() );
                auto  t_i = blas::vector< value_t >( t, blas::range( pos, pos + k_i - 1 )  );

                blas::mulvec( blas::adjoint( VR ), x_i, t_i );
                pos += k_i;
            }// for

            blas::mulvec( value_t(1), cm.U, t, value_t(1), yt );
        }// if

        blas::add( alpha, yt, y_j );
    }// if

    //
    // recurse
    //
    
    for ( auto  sub : cm.sub_blocks )
        mul_vec_cl( alpha, op_M, *sub, x, y );
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

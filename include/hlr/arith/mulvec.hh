#ifndef __HLR_ARITH_MULVEC_HH
#define __HLR_ARITH_MULVEC_HH
//
// Project     : HLR
// Module      : mul_vec
// Description : matrix-vector multiplication functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/h2.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/lrsvmatrix.hh>
#include <hlr/matrix/level_hierarchy.hh>
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

template < typename value_t >
void
mul_vec_cl2 ( const value_t                             alpha,
              const matop_t                             op_M,
              const cluster_blocks_t< value_t > &       cb,
              const vector::scalar_vector< value_t > &  x,
              vector::scalar_vector< value_t > &        y )
{
    mul_vec_cl( alpha, op_M, cb, x, y );
}

template < typename value_t >
void
mul_vec_hier ( const value_t                               alpha,
               const hpro::matop_t                         op_M,
               const matrix::level_hierarchy< value_t > &  M,
               const vector::scalar_vector< value_t > &    x,
               vector::scalar_vector< value_t > &          y )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( op_M == apply_normal );
    
    const auto  nlvl = M.nlevel();

    for ( uint  lvl = 0; lvl < nlvl; ++lvl )
    {
        for ( uint  row = 0; row < M.row_ptr[lvl].size()-1; ++row )
        {
            const auto  lb = M.row_ptr[lvl][row];
            const auto  ub = M.row_ptr[lvl][row+1];

            if ( lb == ub )
                continue;
            
            auto  y_j = blas::vector< value_t >( blas::vec( y ), M.row_mat[lvl][lb]->row_is( op_M ) - y.ofs() );

            for ( uint  j = lb; j < ub; ++j )
            {
                auto  col_idx = M.col_idx[lvl][j];
                auto  mat     = M.row_mat[lvl][j];
                auto  x_i     = blas::vector< value_t >( blas::vec( x ), mat->col_is( op_M ) - x.ofs() );

                mat->apply_add( 1, x_i, y_j, op_M );
            }// for
        }// for
    }// for
}

template < typename value_t >
void
realloc ( cluster_blocks_t< value_t > & )
{
    // nothing to do
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// build data structure with joined lowrank/dense blocks per row/column cluster
//
template < typename T_value >
struct cluster_matrix_t
{
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    using  matrix  = Hpro::TMatrix< value_t >;

    // corresponding index set of cluster
    indexset                           is;

    // signal APLR compression
    bool                               compressed;
    
    // joined low rank factors
    blas::matrix< value_t >            U;

    // compressed storage
    compress::valr::zarray             zU;
    blas::vector< real_t >             S;

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
    using  real_t = Hpro::real_type_t< value_t >;
    
    if ( ! cm.R.empty() )
    {
        //
        // check if compression is to be used
        //

        bool  has_lr   = false;
        bool  has_lrsv = false;
        
        for ( auto  M : cm.R )
        {
            if      ( matrix::is_lowrank(    M ) ) has_lr   = true;
            else if ( matrix::is_lowrank_sv( M ) ) has_lrsv = true;
        }// for

        if ( has_lr && has_lrsv )
            HLR_ERROR( "only either lrmatrix or lrsvmatrix supported" );

        if ( has_lr )
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

            cm.compressed = false;
        }// if
        else
        {
            HLR_ASSERT( has_lrsv );
            
            if ( ! cm.R.empty() )
            {
                bool  all_compressed   = true;
                bool  all_uncompressed = true;
            
                for ( auto  M : cm.R )
                {
                    auto  R = cptrcast( M, matrix::lrsvmatrix< value_t > );
                    
                    if ( R->is_compressed() ) all_uncompressed = false;
                    else                      all_compressed   = false;
                }// for
                
                HLR_ASSERT( all_compressed != all_uncompressed );

                if ( all_compressed )
                {
                    //
                    // determine joined rank and compressed size
                    //
                
                    uint    k     = 0;
                    size_t  zsize = 0;
                
                    for ( auto  M : cm.R )
                    {
                        auto  R = cptrcast( M, matrix::lrsvmatrix< value_t > );
                    
                        k += R->rank();
                    
                        if ( op_M == apply_normal ) zsize += R->zU().size();
                        else                        zsize += R->zV().size();
                    }// for
                
                    //
                    // build joined U factor
                    //
                
                    uint    kpos = 0;
                    size_t  zpos = 0;
                
                    cm.S = blas::vector< real_t >( k );
                    cm.zU.resize( zsize );
                
                    for ( auto  M : cm.R )
                    {
                        auto  R   = cptrcast( M, matrix::lrsvmatrix< value_t > );
                        auto  UR  = R->U( op_M );
                        auto  k_i = R->rank();
                        auto  S_i = blas::vector< value_t >( cm.S, blas::range( kpos, kpos + k_i - 1 ) );

                        blas::copy( R->S(), S_i );
                        kpos += k_i;

                        auto  zsize_i = ( op_M == apply_normal ? R->zU().size() : R->zV().size() );
                
                        if ( op_M == apply_normal ) memcpy( cm.zU.data() + zpos, R->zU().data(), zsize_i );
                        else                        memcpy( cm.zU.data() + zpos, R->zV().data(), zsize_i );

                        zpos += zsize_i;
                    }// for

                    HLR_ASSERT( zpos == zsize );
            
                    cm.compressed = true;
                }// if
                else
                {
                    //
                    // determine joined rank
                    //

                    uint  k = 0;
        
                    for ( auto  M : cm.R )
                        k += cptrcast( M, matrix::lrsvmatrix< value_t > )->rank();

                    //
                    // build joined U factor
                    //

                    auto  nrows = cm.is.size();
                    auto  U     = blas::matrix< value_t >( nrows, k );
                    uint  pos   = 0;

                    for ( auto  M : cm.R )
                    {
                        auto  R   = cptrcast( M, matrix::lrsvmatrix< value_t > );
                        auto  UR  = blas::prod_diag( R->U( op_M ), R->S() );
                        auto  k_i = R->rank();
                        auto  U_i = blas::matrix< value_t >( U, blas::range::all, blas::range( pos, pos + k_i - 1 ) );

                        blas::copy( UR, U_i );
                        pos += k_i;
                    }// for

                    cm.U = std::move( U );

                    cm.compressed = false;
                }// else
            }// if
        }// else
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
            if ( cm.compressed )
            {
                const auto  k   = cm.S.length();
                auto        t   = blas::vector< value_t >( k );
                uint        pos = 0;
            
                for ( auto  M : cm.R )
                {
                    auto  R   = cptrcast( M, matrix::lrsvmatrix< value_t > );
                    auto  k_i = R->rank();
                    auto  x_i = blas::vector< value_t >( blas::vec( x ), M->col_is( op_M ) - x.ofs() );
                    auto  t_i = blas::vector< value_t >( t, blas::range( pos, pos + k_i - 1 )  );
                    
                    #if defined(HLR_HAS_ZBLAS_APLR)
                    if ( op_M == apply_normal )
                        compress::valr::zblas::mulvec( R->ncols(), k_i, apply_adjoint, value_t(1), R->zV(), x_i.data(), t_i.data() );
                    else
                        compress::valr::zblas::mulvec( R->nrows(), k_i, apply_adjoint, value_t(1), R->zU(), x_i.data(), t_i.data() );
                    #else
                    HLR_ERROR( "TODO" );
                    #endif

                    pos += k_i;
                }// for

                for ( uint  i = 0; i < k; ++i )
                    t(i) *= cm.S(i);

                #if defined(HLR_HAS_ZBLAS_APLR)
                compress::valr::zblas::mulvec( yt.length(), k, apply_normal, value_t(1), cm.zU, t.data(), yt.data() );
                #else
                HLR_ERROR( "TODO" );
                #endif
            }// if
            else
            {
                auto  t   = blas::vector< value_t >( cm.U.ncols() );
                uint  pos = 0;
            
                for ( auto  M : cm.R )
                {
                    HLR_ASSERT( matrix::is_lowrank_sv( M ) );
                    
                    auto  R   = cptrcast( M, matrix::lrsvmatrix< value_t > );
                    auto  VR  = R->V( op_M );
                    auto  k_i = R->rank();
                    auto  x_i = blas::vector< value_t >( blas::vec( x ), M->col_is( op_M ) - x.ofs() );
                    auto  t_i = blas::vector< value_t >( t, blas::range( pos, pos + k_i - 1 )  );

                    blas::mulvec( blas::adjoint( VR ), x_i, t_i );
                    pos += k_i;
                }// for

                blas::mulvec( value_t(1), cm.U, t, value_t(1), yt );
            }// else
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
        if constexpr ( Hpro::is_complex_type_v< value_t > )
            return FLOPS_ZGEMV( ncols, rank ) + FLOPS_ZGEMV( nrows, rank );
        else
            return FLOPS_DGEMV( ncols, rank ) + FLOPS_DGEMV( nrows, rank );
    }// if
    else if ( matrix::is_lowrank_sv( M ) )
    {
        const auto  nrows = M.nrows( op_M );
        const auto  ncols = M.ncols( op_M );
        const auto  rank  = cptrcast( &M, matrix::lrsvmatrix< value_t > )->rank();
        
        // t :=     V^H x
        // y := y + α·U·t
        if constexpr ( Hpro::is_complex_type_v< value_t > )
            return FLOPS_ZGEMV( ncols, rank ) + FLOPS_ZGEMV( nrows, rank );
        else
            return FLOPS_DGEMV( ncols, rank ) + FLOPS_DGEMV( nrows, rank );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        const auto  nrows = M.nrows( op_M );
        const auto  ncols = M.ncols( op_M );
        
        if constexpr ( Hpro::is_complex_type_v< value_t > )
            return FLOPS_ZGEMV( nrows, ncols );
        else
            return FLOPS_DGEMV( nrows, ncols );
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

namespace tlr
{

//
// special version for BLR format
//
template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const Hpro::matop_t                       op_M,
          const Hpro::TMatrix< value_t > &          M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y )
{
    using namespace hlr::matrix;

    HLR_ASSERT( is_blocked( M ) );

    auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

    for ( uint  i = 0; i < B->nblock_rows(); ++i )
    {
        bool  first = true;
        auto  y_i   = blas::vector< value_t >();
        
        for ( uint  j = 0; j < B->nblock_cols(); ++j )
        {
            auto  B_ij = B->block( i, j );
            
            if ( ! is_null( B_ij ) )
            {
                if ( first )
                {
                    y_i   = blas::vector< value_t >( blas::vec( y ), B_ij->row_is( op_M ) - y.ofs() );
                    first = false;
                }// if
                
                auto  x_j = blas::vector< value_t >( blas::vec( x ), B_ij->col_is( op_M ) - x.ofs() );
        
                B_ij->apply_add( alpha, x_j, y_i, op_M );
            }// if
        }// for
    }// for
}

}// namespace tlr

}// namespace hlr

#endif // __HLR_ARITH_MULVEC_HH

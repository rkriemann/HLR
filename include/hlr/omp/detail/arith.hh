#ifndef __HLR_OMP_ARITH_IMPL_HH
#define __HLR_OMP_ARITH_IMPL_HH
//
// Project     : HLR
// Module      : arith.hh
// Description : implementation of arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2025. All Rights Reserved.
//

#include <boost/thread.hpp>

namespace hlr { namespace omp { namespace detail {

using indexset = Hpro::TIndexSet;

///////////////////////////////////////////////////////////////////////
//
// matrix-vector with chunk updates
//
///////////////////////////////////////////////////////////////////////

using  mutex_map_t = std::vector< std::mutex >;

constexpr size_t  CHUNK_SIZE = 64;

//
// compute y = y + α op( M ) x
// - very basic algorithm for educational purposes
//
template < typename value_t >
void
mul_vec_simple ( const value_t                    alpha,
                 const Hpro::matop_t              op_M,
                 const Hpro::TMatrix< value_t > & M,
                 const blas::vector< value_t > &  x,
                 blas::vector< value_t > &        y,
                 const size_t                     ofs_rows,
                 const size_t                     ofs_cols,
                 std::mutex &                     mtx )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        #pragma omp taskloop collapse(2) default(shared) firstprivate(alpha,op_M,ofs_rows,ofs_cols)
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                        
                if ( ! is_null( B_ij ) )
                    mul_vec_simple( alpha, op_M, *B_ij, x, y, ofs_rows, ofs_cols, mtx );
            }// for
        }// for
    }// if
    else
    {
        if constexpr ( false )
        {
            auto  x_is = x( M.col_is( op_M ) - ofs_cols );
            auto  y_is = y( M.row_is( op_M ) - ofs_rows );

            {
                auto  lock = std::scoped_lock( mtx );

                M.apply_add( alpha, x_is, y_is, op_M );
            }
        }// if
        else
        {
            auto  x_is = x( M.col_is( op_M ) - ofs_cols );
            auto  y_is = y( M.row_is( op_M ) - ofs_rows );
            auto  t    = blas::vector< value_t >( y_is.length() );
            
            M.apply_add( alpha, x_is, t, op_M );
            
            {
                auto  lock = std::scoped_lock( mtx );
                
                blas::add( value_t(1), t, y_is );
            }
        }// else
    }// else
}

//
// apply t to y in chunks of size CHUNK_SIZE
// while only locking currently updated chunk
//
template < typename value_t >
void
update ( const indexset &                 is,
         const blas::vector< value_t > &  t,
         blas::vector< value_t > &        y,
         mutex_map_t &                    mtx_map )
{
    const idx_t  ofs         = is.first();
    idx_t        start_idx   = is.first();
    const idx_t  last_idx    = is.last();
    idx_t        chunk       = start_idx / CHUNK_SIZE;
    idx_t        end_idx     = std::min< idx_t >( (chunk+1) * CHUNK_SIZE - 1, last_idx );

    while ( start_idx <= end_idx )
    {
        auto  chunk_is = indexset( start_idx, end_idx );
        auto  t_i      = blas::vector< value_t >( t, chunk_is - ofs );
        auto  y_i      = blas::vector< value_t >( y, chunk_is - ofs );

        {
            std::scoped_lock  lock( mtx_map[ chunk ] );
                
            blas::add( value_t(1), t_i, y_i );
        }

        ++chunk;
        start_idx = end_idx + 1;
        end_idx   = std::min< idx_t >( end_idx + CHUNK_SIZE, last_idx );
    }// while
}

//
// compute y = y + α op( M ) x
// - update vector by locked chunks of fixed size
//
template < typename value_t >
void
mul_vec_chunk ( const value_t                    alpha,
                const Hpro::matop_t              op_M,
                const Hpro::TMatrix< value_t > & M,
                const blas::vector< value_t > &  x,
                blas::vector< value_t > &        y,
                const size_t                     ofs_rows,
                const size_t                     ofs_cols,
                mutex_map_t &                    mtx_map )
{
    // assert( M->ncols( op_M ) == x.length() );
    // assert( M->nrows( op_M ) == y.length() );

    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        #pragma omp taskloop collapse(2) default(shared) firstprivate(alpha,op_M,ofs_rows,ofs_cols)
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                        
                if ( ! is_null( B_ij ) )
                    mul_vec_chunk( alpha, op_M, *B_ij, x, y, ofs_rows, ofs_cols, mtx_map );
            }// for
        }// for
    }// if
    else
    {
        const auto  row_is = M.row_is( op_M );
        auto        x_is   = x( M.col_is( op_M ) - ofs_cols );
        auto        y_is   = y( row_is - ofs_rows );
        auto        yt     = blas::vector< value_t >( y_is.length() );
        
        M.apply_add( alpha, x_is, yt, op_M );
        update( row_is, yt, y_is, mtx_map );
    }// else
}

///////////////////////////////////////////////////////////////////////
//
// mat-vec with parallel execution only along row clusters
//
///////////////////////////////////////////////////////////////////////

using hlr::vector::scalar_vector;

template < typename value_t >
void
mul_vec_row ( const value_t                     alpha,
              const Hpro::matop_t               op_M,
              const Hpro::TMatrix< value_t > &  M,
              const blas::vector< value_t > &   x,
              blas::vector< value_t > &         y,
              const size_t                      x_ofs,
              const size_t                      y_ofs )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        #pragma omp taskloop default(shared) firstprivate(alpha,op_M,x_ofs,y_ofs)
        for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
        {
            for ( size_t  j = 0; j < B->nblock_cols( op_M ); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( ! is_null( B_ij ) )
                    mul_vec_row( alpha, op_M, *B_ij, x, y, x_ofs, y_ofs );
            }// for
        }
    }// if
    else
    {
        auto  x_i = x( M.col_is( op_M ) - x_ofs );
        auto  y_j = y( M.row_is( op_M ) - y_ofs );
        auto  yt  = blas::vector< value_t >( y_j.length() );
        
        M.apply_add( alpha, x_i, yt, op_M );
        blas::add( 1, yt, y_j );
    }// else
}

///////////////////////////////////////////////////////////////////////
//
// matrix-vector executing all blocks per block row in one task
//
///////////////////////////////////////////////////////////////////////

using hlr::cluster_block_map_t;

template < typename value_t >
blas::vector< value_t >
mul_vec_reduce ( const matop_t                             op_M,
                 const matrix_list_t< value_t > &          matrices,
                 const vector::scalar_vector< value_t > &  x,
                 const blas::vector< value_t > &           y_j,
                 const uint                                lb,
                 const uint                                ub )
{
    if ( ub - lb > 16 )
    {
        const uint  mid = (ub + lb) / 2;
        auto        y1  = blas::vector< value_t >();
        auto        y2  = blas::vector< value_t >();

        #pragma omp taskgroup
        {
            #pragma omp task default(shared) firstprivate(op_M,lb,mid)
            { y1 = std::move( mul_vec_reduce( op_M, matrices, x, y_j, lb, mid ) ); }
            
            #pragma omp task default(shared) firstprivate(op_M,mid,ub)
            { y2 = std::move( mul_vec_reduce( op_M, matrices, x, y_j, mid, ub ) ); }
        }

        blas::add( value_t(1), y1, y2 );

        return y2;
    }// if
    else
    {
        auto  yt = blas::vector< value_t >( y_j.length() );
    
        for ( uint  i = lb; i < ub; ++i )
        {
            const auto  A   = matrices[i];
            auto        x_i = blas::vector< value_t >( blas::vec( x ), A->col_is( op_M ) - x.ofs() );
        
            A->apply_add( 1, x_i, yt, op_M );
        }// for

        return yt;
    }// else
}
    
template < typename value_t >
void
mul_vec_cl ( const value_t                           alpha,
             const matop_t                           op_M,
             const Hpro::TMatrix< value_t > &        M,
             const cluster_block_map_t< value_t > &  blocks,
             const blas::vector< value_t > &         x,
             blas::vector< value_t > &               y,
             const size_t                            x_ofs,
             const size_t                            y_ofs )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        #pragma omp taskloop default(shared) firstprivate(alpha,op_M,x_ofs,y_ofs)
        for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
        {
            // WARNING: assuming that by following diagonal blocks all clusters are reached
            auto  B_ii = B->block( i, i );
                    
            if ( ! is_null( B_ii ) )
                hlr::omp::detail::mul_vec_cl( alpha, op_M, *B_ii, blocks, x, y, x_ofs, y_ofs );
        }// for
    }// if

    if ( ! blocks.contains( M.row_is( op_M ) ) )
        return;

    //
    // compute update with all blocks in current block row
    //
    
    auto &  mat_list = blocks.at( M.row_is( op_M ) );
    auto    y_j      = y( M.row_is( op_M ) - y_ofs );

    #if 0
    auto    yt       = mul_vec_reduce( op_M, mat_list, x, y_j, 0, mat_list.size() );
    #else
    auto    yt       = blas::vector< value_t >( y_j.length() );
    
    for ( auto  A : mat_list )
    {
        auto  x_i = x( A->col_is( op_M ) - x_ofs );
        
        A->apply_add( 1, x_i, yt, op_M );
    }// for
    #endif

    blas::add( alpha, yt, y_j );
}

template < typename value_t >
void
mul_vec_cl ( const value_t                        alpha,
             const matop_t                        op_M,
             const cluster_blocks_t< value_t > &  cb,
             const blas::vector< value_t > &      x,
             blas::vector< value_t > &            y,
             const size_t                         x_ofs,
             const size_t                         y_ofs )
{
    //
    // compute update with all block in current block row
    //

    if ( ! cb.M.empty() )
    {
        auto  y_j = y( cb.is - y_ofs );
        auto  yt  = blas::vector< value_t >( y_j.length() );
    
        for ( auto  M : cb.M )
        {
            auto  x_i = x( M->col_is( op_M ) - x_ofs );
            
            M->apply_add( 1, x_i, yt, op_M );
        }// for
        
        blas::add( alpha, yt, y_j );
    }// if

    //
    // recurse
    //

    if ( cb.sub_blocks.size() > 0 )
    {
        #pragma omp taskloop default(shared) firstprivate(alpha,op_M,x_ofs,y_ofs)
        for ( uint  i = 0; i < cb.sub_blocks.size(); ++i )
        {
            hlr::omp::detail::mul_vec_cl( alpha, op_M, *cb.sub_blocks[i], x, y, x_ofs, y_ofs );
        }// for
    }// if
}

template < typename value_t >
void
mul_vec_cl2 ( const value_t                        alpha,
              const matop_t                        op_M,
              const cluster_blocks_t< value_t > &  cb,
              const blas::vector< value_t > &      x,
              blas::vector< value_t > &            y,
              const size_t                         x_ofs,
              const size_t                         y_ofs )
{

    //
    // recurse
    //

    if ( cb.sub_blocks.size() > 0 )
    {
        #pragma omp taskloop default(shared) firstprivate(alpha,op_M,x_ofs,y_ofs)
        for ( uint  i = 0; i < cb.sub_blocks.size(); ++i )
        {
            hlr::omp::detail::mul_vec_cl( alpha, op_M, *cb.sub_blocks[i], x, y, x_ofs, y_ofs );
        }// for
    }// if

    //
    // compute update with all block in current block row
    //

    if ( ! cb.M.empty() )
    {
        auto  y_j = y( cb.is - y_ofs );
        auto  yt  = blas::vector< value_t >( y_j.length() );
    
        for ( auto  M : cb.M )
        {
            auto  x_i = x( M->col_is( op_M ) - x_ofs );
            
            M->apply_add( 1, x_i, yt, op_M );
        }// for
        
        blas::add( alpha, yt, y_j );
    }// if
}

template < typename value_t >
void
build_joined_matrix ( const matop_t                  op_M,
                      cluster_matrix_t< value_t > &  cm,
                      const accuracy &               acc )
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
                
                    uint    k = 0;
                
                    for ( auto  M : cm.R )
                        k += cptrcast( M, matrix::lrsvmatrix< value_t > )->rank();
                
                    //
                    // build joined U factor
                    //

                    const auto  nrows = cm.is.size();
                    uint        kpos  = 0;
                    auto        U     = blas::matrix< value_t >( nrows, k );

                    cm.S = blas::vector< real_t >( k );

                    for ( auto  M : cm.R )
                    {
                        const auto  R   = cptrcast( M, matrix::lrsvmatrix< value_t > );
                        const auto  k_i = R->rank();
                        auto        U_i = blas::matrix< value_t >( U, blas::range::all, blas::range( kpos, kpos + k_i - 1 ) );
                        auto        S_i = blas::vector< value_t >( cm.S, blas::range( kpos, kpos + k_i - 1 ) );

                        blas::copy( R->S(), S_i );
                        
                        if ( op_M == apply_normal )
                        {
                            auto  RU = R->U();

                            blas::copy( RU, U_i );
                        }// if
                        else
                        {
                            auto  RV = R->V();

                            blas::copy( RV, U_i );
                        }// else

                        kpos += k_i;
                    }// for
                    
                    auto  S_tol = compress::valr::get_tolerances( acc, cm.S );
                    
                    cm.zU         = compress::valr::compress_lr( U, S_tol );
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

    //
    // recurse
    //
    
    if ( cm.sub_blocks.size() > 0 )
    {
        #pragma omp taskloop default(shared) firstprivate(op_M)
        for ( uint  i = 0; i < cm.sub_blocks.size(); ++i )
        {
            hlr::omp::detail::build_joined_matrix( op_M, *cm.sub_blocks[i], acc );
        }// for
    }// if
}

template < typename value_t >
void
mul_vec_cl ( const value_t                        alpha,
             const matop_t                        op_M,
             const cluster_matrix_t< value_t > &  cm,
             const blas::vector< value_t > &      x,
             blas::vector< value_t > &            y,
             const size_t                         x_ofs,
             const size_t                         y_ofs )
{
    if ( alpha == value_t(0) )
        return;

    //
    // compute update with all block in current block row
    //

    if ( ! cm.D.empty() || ! cm.R.empty() )
    {
        auto  y_j = y( cm.is - y_ofs );
        auto  yt  = blas::vector< value_t >( y_j.length() );
    
        if ( ! cm.D.empty() )
        {
            for ( auto  M : cm.D )
            {
                auto  x_i = x( M->col_is( op_M ) - x_ofs );
                
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
                    auto  x_i = x( M->col_is( op_M ) - x_ofs );
                    auto  t_i = blas::vector< value_t >( t, blas::range( pos, pos + k_i - 1 )  );
                    
                    #if defined(HLR_HAS_ZBLAS_VALR)
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

                #if defined(HLR_HAS_ZBLAS_VALR)
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
                    if ( matrix::is_lowrank( M ) )
                    {
                        auto  R   = cptrcast( M, matrix::lrmatrix< value_t > );
                        auto  VR  = R->V( op_M );
                        auto  k_i = R->rank();
                        auto  x_i = x( M->col_is( op_M ) - x_ofs );
                        auto  t_i = blas::vector< value_t >( t, blas::range( pos, pos + k_i - 1 )  );
                        
                        blas::mulvec( blas::adjoint( VR ), x_i, t_i );
                        pos += k_i;
                    }// if
                    else if ( matrix::is_lowrank_sv( M ) )
                    {
                        auto  R   = cptrcast( M, matrix::lrsvmatrix< value_t > );
                        auto  VR  = R->V( op_M );
                        auto  k_i = R->rank();
                        auto  x_i = x( M->col_is( op_M ) - x_ofs );
                        auto  t_i = blas::vector< value_t >( t, blas::range( pos, pos + k_i - 1 )  );
                        
                        blas::mulvec( blas::adjoint( VR ), x_i, t_i );
                        pos += k_i;
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type: " + M->typestr() );
                }// for

                blas::mulvec( value_t(1), cm.U, t, value_t(1), yt );
            }// else
        }// if

        blas::add( alpha, yt, y_j );
    }// if

    //
    // recurse
    //
    
    if ( cm.sub_blocks.size() > 0 )
    {
        #pragma omp taskloop default(shared) firstprivate(alpha,op_M,x_ofs,y_ofs)
        for ( uint  i = 0; i < cm.sub_blocks.size(); ++i )
        {
            hlr::omp::detail::mul_vec_cl( alpha, op_M, *cm.sub_blocks[i], x, y, x_ofs, y_ofs );
        }// for
    }// if
}

template < typename value_t >
void
mul_vec_hier ( const value_t                               alpha,
               const Hpro::matop_t                         op_M,
               const matrix::level_hierarchy< value_t > &  M,
               const blas::vector< value_t > &             x,
               blas::vector< value_t > &                   y,
               const size_t                                x_ofs,
               const size_t                                y_ofs )
{
    HLR_ASSERT( op_M == apply_normal );
    
    for ( uint  lvl = 0; lvl < M.nlevel(); ++lvl )
    {
        #pragma omp taskloop default(shared) firstprivate(alpha,op_M,lvl)
        for ( uint  row = 0; row < M.row_ptr[lvl].size()-1; ++row )
        {
            const auto  lb = M.row_ptr[lvl][row];
            const auto  ub = M.row_ptr[lvl][row+1];

            if ( lb == ub )
                continue;
            
            const auto  row_is = M.row_mat[lvl][lb]->row_is( op_M );
            auto        y_j    = y( row_is - y_ofs );
            auto        t_j    = blas::vector< value_t >( y_j.length() );
                
            for ( uint  j = lb; j < ub; ++j )
            {
                auto  col_idx = M.col_idx[lvl][j];
                auto  mat     = M.row_mat[lvl][j];
                auto  x_i     = x( mat->col_is( op_M ) - x_ofs );
                        
                mat->apply_add( alpha, x_i, t_j, op_M );
            }// for

            blas::add( 1, t_j, y_j );
        }// for
    }// for
}

//
// compute y = y + α op( M ) x
// - using thread local storage as destination vector
//
template < typename value_t >
void
mul_vec_ts ( const value_t                                            alpha,
             const Hpro::matop_t                                      op_M,
             const Hpro::TMatrix< value_t > &                         M,
             const blas::vector< value_t > &                          x,
             boost::thread_specific_ptr< blas::vector< value_t > > &  y,
             const size_t                                             x_ofs,
             const size_t                                             y_ofs,
             const size_t                                             y_size )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        #pragma omp taskloop collapse(2) default(shared) firstprivate(alpha,op_M,x_ofs,y_ofs,y_size)
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                        
                if ( ! is_null( B_ij ) )
                    mul_vec_ts( alpha, op_M, *B_ij, x, y, x_ofs, y_ofs, y_size );
            }// for
        }// for
    }// if
    else
    {
        if ( ! y.get() )
            y.reset( blas::vector< value_t >( y_size ) );
        
        auto  x_is  = x( M.col_is( op_M ) - x_ofs );
        auto  y_loc = y.get();
        auto  y_is  = (*y_loc)( M.row_is( op_M ) - y_ofs );
            
        M.apply_add( alpha, x_is, y_is, op_M );
    }// else
}

}}} // namespace hlr::omp::detail

#endif // __HLR_OMP_ARITH_IMPL_HH

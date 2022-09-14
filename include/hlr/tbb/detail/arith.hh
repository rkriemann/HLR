#ifndef __HLR_TBB_ARITH_IMPL_HH
#define __HLR_TBB_ARITH_IMPL_HH
//
// Project     : HLib
// File        : arith.hh
// Description : implementation of arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

namespace hlr { namespace tbb { namespace detail {

using indexset = Hpro::TIndexSet;

///////////////////////////////////////////////////////////////////////
//
// matrix-vector with chunk updates
//
///////////////////////////////////////////////////////////////////////

using  mutex_map_t = std::map< idx_t, std::unique_ptr< std::mutex > >;

constexpr size_t  CHUNK_SIZE = 64;

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
        const indexset  chunk_is( start_idx, end_idx );
        auto            t_i = blas::vector< value_t >( t, chunk_is - ofs );
        auto            y_i = blas::vector< value_t >( y, chunk_is - ofs );

        {
            std::scoped_lock  lock( * mtx_map[ chunk ] );
                
            blas::add( value_t(1), t_i, y_i );
        }

        ++chunk;
        start_idx = end_idx + 1;
        end_idx   = std::min< idx_t >( end_idx + CHUNK_SIZE, last_idx );
    }// while
}

//
// compute y = y + α op( M ) x
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

    if ( alpha == value_t(0) )
        return;

    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< size_t >( 0, B->nblock_rows(),
                                              0, B->nblock_cols() ),
            [=,&x,&y,&mtx_map] ( const auto &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  B_ij = B->block( i, j );
                        
                        if ( ! is_null( B_ij ) )
                            mul_vec_chunk( alpha, op_M, *B_ij, x, y, ofs_rows, ofs_cols, mtx_map );
                    }// for
                }// for
            } );
    }// if
    else
    {
        const auto  row_is = M.row_is( op_M );
        const auto  col_is = M.col_is( op_M );
        auto        x_is   = x( col_is - ofs_cols );
        auto        y_is   = y( row_is - ofs_rows );
        auto        yt     = blas::vector< value_t >( y_is.length() );
        
        if ( matrix::is_compressible_dense( M ) )
        {
            M.apply_add( alpha, x_is, yt, op_M );
        }// if
        else if ( is_dense( M ) )
        {
            auto  D  = cptrcast( &M, Hpro::TDenseMatrix< value_t > );
        
            blas::mulvec( alpha, blas::mat_view( op_M, blas::mat( D ) ), x_is, value_t(1), yt );
        }// if
        else if ( matrix::is_compressible_lowrank( M ) )
        {
            M.apply_add( alpha, x_is, yt, op_M );
        }// if
        else if ( matrix::is_compressible_lowrankS( M ) )
        {
            M.apply_add( alpha, x_is, yt, op_M );
        }// if
        else if ( is_lowrank( M ) )
        {
            auto  R = cptrcast( &M, Hpro::TRkMatrix< value_t > );

            if ( op_M == Hpro::apply_normal )
            {
                auto  t = blas::mulvec( value_t(1), blas::adjoint( blas::mat_V( R ) ), x_is );

                blas::mulvec( alpha, blas::mat_U( R ), t, value_t(1), yt );
            }// if
            else if ( op_M == Hpro::apply_transposed )
            {
                HLR_ASSERT( Hpro::is_complex_type< value_t >::value == false );
            
                auto  t = blas::mulvec( value_t(1), blas::transposed( blas::mat_U( R ) ), x_is );

                blas::mulvec( alpha, blas::mat_V( R ), t, value_t(1), yt );
            }// if
            else if ( op_M == Hpro::apply_adjoint )
            {
                auto  t = blas::mulvec( value_t(1), blas::adjoint( blas::mat_U( R ) ), x_is );

                blas::mulvec( alpha, blas::mat_V( R ), t, value_t(1), yt );
            }// if
        }// if
        else if ( hlr::matrix::is_uniform_lowrank( M ) )
        {
            auto  R = cptrcast( &M, hlr::matrix::uniform_lrmatrix< value_t > );
        
            if ( op_M == Hpro::apply_normal )
            {
                //
                // y = y + U·S·V^H x
                //
        
                auto  t = R->col_cb().transform_forward( x );
                auto  s = blas::mulvec( alpha, R->coeff(), t );

                yt = std::move( R->row_cb().transform_backward( s ) );
            }// if
            else if ( op_M == Hpro::apply_transposed )
            {
                //
                // y = y + (U·S·V^H)^T x
                //   = y + conj(V)·S^T·U^T x
                //
        
                auto  cx = blas::copy( x );

                blas::conj( cx );
        
                auto  t  = R->row_cb().transform_forward( cx );

                blas::conj( t );
        
                auto  s = blas::mulvec( alpha, blas::transposed(R->coeff()), t );

                yt = std::move( R->col_cb().transform_backward( s ) );
            }// if
            else if ( op_M == Hpro::apply_adjoint )
            {
                //
                // y = y + (U·S·V^H)^H x
                //   = y + V·S^H·U^H x
                //
        
                auto  t = R->row_cb().transform_forward( x );
                auto  s = blas::mulvec( alpha, blas::adjoint(R->coeff()), t );

                yt = std::move( R->col_cb().transform_backward( s ) );
            }// if
        }// if
        else
            HLR_ASSERT( "unsupported matrix type" );

        update( M.row_is( op_M ), yt, y_is, mtx_map );
    }// else
}

///////////////////////////////////////////////////////////////////////
//
// matrix-vector with reductions
//
///////////////////////////////////////////////////////////////////////

template < typename value_t >
void
mul_vec_reduce ( const value_t                    alpha,
                 const matop_t                    op_M,
                 const Hpro::TMatrix< value_t > & M,
                 const blas::vector< value_t > &  x,
                 blas::vector< value_t > &        y )
{
    
}

}}}// namespace hlr::tbb::detail

#endif // __HLR_TBB_ARITH_IMPL_HH

#ifndef __HLR_TBB_ARITH_IMPL_HH
#define __HLR_TBB_ARITH_IMPL_HH
//
// Project     : HLR
// Module      : arith.hh
// Description : implementation of arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
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
        
        M.apply_add( alpha, x_is, yt, op_M );
        update( M.row_is( op_M ), yt, y_is, mtx_map );
    }// else
}

//
// mat-vec with parallel execution only along row clusters
//
using hlr::vector::scalar_vector;

template < typename value_t >
void
mul_vec_row ( const value_t                     alpha,
              const Hpro::matop_t               op_M,
              const Hpro::TMatrix< value_t > &  M,
              const scalar_vector< value_t > &  sx,
              scalar_vector< value_t > &        sy )
{
    if ( alpha == value_t(0) )
        return;

    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        ::tbb::parallel_for< size_t >( 0, B->nblock_rows( op_M ) ),
              [&,alpha,op_M] ( const auto &  i )
              {
                  for ( size_t  j = 0; j < B->nblock_cols( op_M ); ++j )
                  {
                      auto  B_ij = B->block( i, j );
                      
                      if ( ! is_null( B_ij ) )
                          mul_vec_row( alpha, op_M, *B_ij, sx, sy );
                  }// for
              }
        );
    }// if
    else
    {
        auto  x_i = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );
        auto  yt  = blas::vector< value_t >( y_is.length() );
        
        M.apply_add( alpha, x_i, y_j, op_M );
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

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with lower/upper triangular matrix
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
solve_lower_tri ( const Hpro::matop_t               op_L,
                  const Hpro::TMatrix< value_t > &  L,
                  Hpro::TScalarVector< value_t > &  v,
                  const Hpro::diag_type_t           diag_mode )
{
    HLR_LOG( 4, Hpro::to_string( "solve_lower_tri( %d )", L.id() ) );
        
    if ( is_blocked( L ) )
    {
        auto        BL  = cptrcast( & L, Hpro::TBlockMatrix< value_t > );
        const auto  nbr = BL->nblock_rows();
        const auto  nbc = BL->nblock_cols();
            
        if ( op_L == Hpro::apply_normal )
        {
            //
            // solve from top to bottom
            // - first diagonal block L_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
            {
                auto  L_ii = BL->block( i, i );
            
                if ( ! is_null( L_ii ) )
                {
                    auto  v_i = v.sub_vector( L_ii->col_is() );
                
                    solve_lower_tri( op_L, *L_ii, v_i, diag_mode );
                }// if
                
                ::tbb::parallel_for< uint >(
                    i+1, nbr,
                    [BL,i,op_L,&v] ( const uint  j )
                    {
                        auto  L_ji = BL->block( j, i );
                        
                        if ( ! is_null( L_ji ) )
                        {
                            auto  v_j = v.sub_vector( L_ji->row_is() );
                            auto  v_i = v.sub_vector( L_ji->col_is() );
                            
                            mul_vec< value_t >( value_t(-1), op_L, *L_ji, v_i, v_j );
                        }// if
                    } );
            }// for
        }// if
        else
        {
            //
            // solve bottom to top
            // - first diagonal block L_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( int  i = std::min( nbr, nbc )-1; i >= 0; --i )
            {
                auto  L_ii = BL->block( i, i );
            
                if ( ! is_null( L_ii ) )
                {
                    auto  v_i = v.sub_vector( L_ii->row_is() );
                    
                    solve_lower_tri( op_L, *L_ii, v_i, diag_mode );
                }// if

                ::tbb::parallel_for(
                    0, i,
                    [BL,i,op_L,&v] ( const uint  j )
                    {
                        auto  L_ij = BL->block( i, j );
                    
                        if ( ! is_null( L_ij ) )
                        {
                            auto  v_i = v.sub_vector( L_ij->col_is() );
                            auto  v_j = v.sub_vector( L_ij->row_is() );
                            
                            mul_vec( value_t(-1), op_L, *L_ij, v_j, v_i );
                        }// if
                    } );
            }// for
        }// else
    }// if
    else if ( matrix::is_dense( L ) )
    {
        if ( diag_mode == Hpro::general_diag )
        {
            //
            // assuming L contains inverse (store_inverse!)
            //

            auto  vc = Hpro::TScalarVector< value_t >( v );

            v.scale( 0 );
            mul_vec( value_t(1), op_L, L, vc, v );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );
}

template < typename value_t >
void
solve_upper_tri ( const Hpro::matop_t               op_U,
                  const Hpro::TMatrix< value_t > &  U,
                  Hpro::TScalarVector< value_t > &  v,
                  const Hpro::diag_type_t           diag_mode )
{
    HLR_LOG( 4, Hpro::to_string( "solve_upper_tri( %d )", U.id() ) );
        
    if ( is_blocked( U ) )
    {
        auto        BU  = cptrcast( & U, Hpro::TBlockMatrix< value_t > );
        const auto  nbr = BU->nblock_rows();
        const auto  nbc = BU->nblock_cols();
            
        if ( op_U == Hpro::apply_normal )
        {
            //
            // solve from top to bottom
            // - first diagonal block U_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( int  i = std::min<int>( nbr, nbc )-1; i >= 0; --i )
            {
                auto  U_ii = BU->block( i, i );
            
                if ( ! is_null( U_ii ) )
                {
                    auto  v_i = v.sub_vector( U_ii->col_is() );
                
                    solve_upper_tri( op_U, *U_ii, v_i, diag_mode );
                }// if
            
                ::tbb::parallel_for(
                    0, i,
                    [BU,i,op_U,&v] ( const uint  j )
                    {
                        auto  U_ji = BU->block( j, i );
                        
                        if ( ! is_null( U_ji ) )
                        {
                            auto  v_j = v.sub_vector( U_ji->row_is() );
                            auto  v_i = v.sub_vector( U_ji->col_is() );
                            
                            mul_vec( value_t(-1), op_U, *U_ji, v_i, v_j );
                        }// if
                    } );
            }// for
        }// if
        else
        {
            //
            // solve bottom to top
            // - first diagonal block U_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
            {
                auto  U_ii = BU->block( i, i );
            
                if ( ! is_null( U_ii ) )
                {
                    auto  v_i = v.sub_vector( U_ii->row_is() );
                
                    solve_upper_tri( op_U, *U_ii, v_i, diag_mode );
                }// if

                ::tbb::parallel_for(
                    i+1, nbc,
                    [BU,i,op_U,&v] ( const uint  j )
                    {
                        auto  U_ij = BU->block( i, j );
                    
                        if ( ! is_null( U_ij ) )
                        {
                            auto  v_i = v.sub_vector( U_ij->col_is() );
                            auto  v_j = v.sub_vector( U_ij->row_is() );
                            
                            mul_vec( value_t(-1), op_U, *U_ij, v_j, v_i );
                        }// if
                    } );
            }// for
        }// else
    }// if
    else if ( matrix::is_dense( U ) )
    {
        if ( diag_mode == Hpro::general_diag )
        {
            //
            // assuming U contains inverse (store_inverse!)
            //

            auto  vc = Hpro::TScalarVector< value_t >( v );

            v.scale( 0 );
            mul_vec( value_t(1), op_U, U, vc, v );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );
}

}}}// namespace hlr::tbb::detail

#endif // __HLR_TBB_ARITH_IMPL_HH

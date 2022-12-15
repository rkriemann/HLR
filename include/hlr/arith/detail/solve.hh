#ifndef __HLR_ARITH_DETAIL_SOLVE_HH
#define __HLR_ARITH_DETAIL_SOLVE_HH
//
// Project     : HLib
// File        : solve.hh
// Description : matrix solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlr/arith/multiply.hh>

namespace hlr
{

#if defined(NDEBUG)
#  define HLR_SOLVE_PRINT( msg )
#else
#  define HLR_SOLVE_PRINT( msg )   HLR_LOG( 4, msg )
#endif

////////////////////////////////////////////////////////////////////////////////
//
// matrix solving with diagonal D
//
// solve D·X = M (from_left) or X·D = M (from_right)
// - on exit, M contains X
//
////////////////////////////////////////////////////////////////////////////////

//
// forward for general version
//
template < typename value_t,
           typename approx_t >
void
solve_diag ( const eval_side_t                   side,
             const diag_type_t                   diag,
             const matop_t                       op_D,
             const Hpro::TMatrix< value_t > &    D,
             Hpro::TMatrix< value_t > &          M,
             const Hpro::TTruncAcc &             acc,
             const approx_t &                    approx );

//
// forward for general version without (!) approximation
//
template < typename value_t >
void
solve_diag ( const eval_side_t                 side,
             const diag_type_t                 diag,
             const matop_t                     op_D,
             const Hpro::TMatrix< value_t > &  D,
             Hpro::TMatrix< value_t > &        M );

//
// specializations
//
template < typename value_t,
           typename approx_t >
void
solve_diag ( const eval_side_t                      side,
             const diag_type_t                      diag,
             const matop_t                          op_D,
             const Hpro::TBlockMatrix< value_t > &  D,
             Hpro::TBlockMatrix< value_t > &        M,
             const Hpro::TTruncAcc &                acc,
             const approx_t &                       approx )
{
    if ( side == from_left )
    {
        //
        // ⎛D₀₀       ⎞   ⎛X₀₀ X₀₁ X₀₂ …⎞   ⎛M₀₀ M₀₁ M₀₂ …⎞
        // ⎜   D₁₁    ⎟ × ⎜X₁₀ X₁₁ X₁₂ …⎟ = ⎜M₁₀ M₁₁ M₁₂ …⎟
        // ⎜      D₂₂ ⎟   ⎜X₂₀ X₂₁ X₂₂ …⎟   ⎜M₂₀ M₂₁ M₂₂ …⎟
        // ⎝         …⎠   ⎝ …   …   …  …⎠   ⎝ …   …   …  …⎠
        //
        //
        // ⇒ (D₀₀) × (X₀₀ X₀₁ X₀₂ …) = (M₀₀ M₀₁ M₀₂ …)
        //
        
        auto  nbr = M.nblock_rows();
        auto  nbc = M.nblock_cols();

        HLR_ASSERT( ( D.nblock_rows( op_D ) == nbr ) && ( D.nblock_cols( op_D ) == nbr ) );
            
        for ( auto  i = 0; i < int(nbr); ++i )
        {
            auto  D_ii = D.block( i, i, op_D );
            
            HLR_ASSERT( ! is_null( D_ii ) );
                
            for ( auto  j = 0; j < int(nbc); ++j )
            {
                auto  M_ij = M.block( i, j );
                
                if ( ! is_null( M_ij ) )
                    solve_diag< value_t >( side, diag, op_D, *D_ii, *M_ij, acc, approx );
            }// for
        }// for
    }// if
    else
    {
        //
        // ⎛X₀₀ X₀₁ X₀₂ …⎞   ⎛D₀₀       ⎞   ⎛M₀₀ M₀₁ M₀₂ …⎞
        // ⎜X₁₀ X₁₁ X₁₂ …⎟ × ⎜   D₁₁    ⎟ = ⎜M₁₀ M₁₁ M₁₂ …⎟
        // ⎜X₂₀ X₂₁ X₂₂ …⎟   ⎜      D₂₂ ⎟   ⎜M₂₀ M₂₁ M₂₂ …⎟
        // ⎝ …   …   …  …⎠   ⎝         …⎠   ⎝ …   …   …  …⎠
        //
        // ⇒
        //
        // ⎛X₀₀⎞           ⎛M₀₀⎞
        // ⎜X₁₀⎟ × (D₀₀) = ⎜M₁₀⎟
        // ⎜X₂₀⎟           ⎜M₂₀⎟
        // ⎝ … ⎠           ⎝ … ⎠
        //

        auto  nbr = M.nblock_rows();
        auto  nbc = M.nblock_cols();

        HLR_ASSERT( ( D.nblock_rows( op_D ) == nbc ) && ( D.nblock_cols( op_D ) == nbc ) );
            
        for ( auto  j = 0; j < int(nbc); ++j )
        {
            auto  D_jj = D.block( j, j, op_D );
            
            HLR_ASSERT( ! is_null( D_jj ) );
                
            for ( auto  i = 0; i < int(nbr); ++i )
            {
                auto  M_ij = M.block( i, j );
                
                if ( ! is_null( M_ij ) )
                    solve_diag< value_t >( side, diag, op_D, *D_jj, *M_ij, acc, approx );
            }// for
        }// for
    }// else
}

template < typename value_t >
void
solve_diag ( const eval_side_t                      side,
             const diag_type_t                      diag,
             const matop_t                          op_D,
             const Hpro::TBlockMatrix< value_t > &  D,
             Hpro::TDenseMatrix< value_t > &        M )
{
    if ( side == from_left )
    {
        //
        // ⎛D₀₀       ⎞   ⎛X₀₀ X₀₁ X₀₂ …⎞   ⎛M₀₀ M₀₁ M₀₂ …⎞
        // ⎜   D₁₁    ⎟ × ⎜X₁₀ X₁₁ X₁₂ …⎟ = ⎜M₁₀ M₁₁ M₁₂ …⎟
        // ⎜      D₂₂ ⎟   ⎜X₂₀ X₂₁ X₂₂ …⎟   ⎜M₂₀ M₂₁ M₂₂ …⎟
        // ⎝         …⎠   ⎝ …   …   …  …⎠   ⎝ …   …   …  …⎠
        //
        // ⇒ (D₀₀) × (X₀₀ X₀₁ X₀₂ …) = (M₀₀ M₀₁ M₀₂ …)
        //
        
        for ( uint j = 0; j < D.nblock_cols( op_D ); ++j )
        {
            const auto  D_jj = D.block( j, j, op_D );

            HLR_ASSERT( ! is_null( D_jj ) );

            auto  M_j = blas::matrix< value_t >( blas::mat( M ),
                                                 D_jj->col_is( op_D ) - D.col_ofs( op_D ),
                                                 blas::range::all );
            auto  D_j = Hpro::TDenseMatrix< value_t >( D_jj->col_is( op_D ), M.col_is(), M_j );

            solve_diag< value_t >( side, diag, op_D, *D_jj, D_j );
        }// for
    }// if
    else
    {
        //
        // ⎛X₀₀ X₀₁ X₀₂ …⎞   ⎛D₀₀       ⎞   ⎛M₀₀ M₀₁ M₀₂ …⎞
        // ⎜X₁₀ X₁₁ X₁₂ …⎟ × ⎜   D₁₁    ⎟ = ⎜M₁₀ M₁₁ M₁₂ …⎟
        // ⎜X₂₀ X₂₁ X₂₂ …⎟   ⎜      D₂₂ ⎟   ⎜M₂₀ M₂₁ M₂₂ …⎟
        // ⎝ …   …   …  …⎠   ⎝         …⎠   ⎝ …   …   …  …⎠
        //
        // ⇒
        //
        // ⎛X₀₀⎞           ⎛M₀₀⎞
        // ⎜X₁₀⎟ × (D₀₀) = ⎜M₁₀⎟
        // ⎜X₂₀⎟           ⎜M₂₀⎟
        // ⎝ … ⎠           ⎝ … ⎠
        //
        
        for ( uint i = 0; i < D.nblock_rows( op_D ); ++i )
        {
            const auto  D_ii = D.block( i, i, op_D );

            HLR_ASSERT( ! is_null( D_ii ) );

            auto  M_i = blas::matrix< value_t >( blas::mat( M ),
                                                 blas::range::all,
                                                 D_ii->row_is( op_D ) - D.row_ofs( op_D ) );
            auto  D_i = Hpro::TDenseMatrix< value_t >( M.row_is(), D_ii->row_is( op_D ), M_i );

            solve_diag< value_t >( side, diag, op_D, *D_ii, D_i );
        }// for
    }// else
}

template < typename value_t >
void
solve_diag ( const eval_side_t                      side,
             const diag_type_t                      diag,
             const matop_t                          op_D,
             const Hpro::TBlockMatrix< value_t > &  D,
             Hpro::TRkMatrix< value_t > &           M )
{
    if ( side == from_left )
    {
        //
        // solve D M = D W X' = U V'
        // as D W = U
        //
        
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_U( M ) );

        solve_diag< value_t >( side, diag, op_D, D, U );
    }// if
    else
    {
        //
        // solve M D = W X' D = U V'
        // as X' D = V'
        // as D' X = V
        //
        
        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_V( M ) );

        solve_diag< value_t >( from_left, diag, blas::adjoint( op_D ), D, V );
    }// else
}

template < typename value_t >
void
solve_diag ( const eval_side_t                      side,
             const diag_type_t                      diag,
             const matop_t                          op_D,
             const Hpro::TBlockMatrix< value_t > &  D,
             matrix::lrsmatrix< value_t > &         M )
{
    if ( side == from_left )
    {
        //
        // solve D M = D W T X' = U S V'
        // as D W = U
        //
        
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), M.U() );

        solve_diag< value_t >( side, diag, op_D, D, U );
    }// if
    else
    {
        //
        // solve M D = W T X' D = U S V'
        // as X' D = V'
        // as L' X = V
        //
        
        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), M.V() );

        solve_diag< value_t >( side, diag, blas::adjoint( op_D ), D, V );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_diag ( const eval_side_t                      side,
             const diag_type_t                      diag,
             const matop_t                          /* op_D */,
             const Hpro::TDenseMatrix< value_t > &  /* D */,
             Hpro::TBlockMatrix< value_t > &        /* M */,
             const Hpro::TTruncAcc &                /* acc */,
             const approx_t &                       /* approx */ )
{
    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        HLR_ERROR( "todo" );
    }// if
    else
    {
        HLR_ERROR( "todo" );
    }// else
}

template < typename value_t >
void
solve_diag ( const eval_side_t                      side,
             const diag_type_t                      diag,
             const matop_t                          op_D,
             const Hpro::TDenseMatrix< value_t > &  D,
             Hpro::TDenseMatrix< value_t > &        M )
{
    if ( diag == unit_diag )
        return;
    
    //
    // blockwise evaluation and D is assumed to hold D^-1
    // TODO: option argument?
    //

    auto  Mc = blas::copy( blas::mat( M ) );

    if ( side == from_left )
    {
        blas::prod( value_t(1), blas::mat_view( op_D, blas::mat( D ) ), Mc, value_t(0), blas::mat( M ) );
    }// if
    else
    {
        blas::prod( value_t(1), Mc, blas::mat_view( op_D, blas::mat( D ) ), value_t(0), blas::mat( M ) );
    }// else
}

template < typename value_t >
void
solve_diag ( const eval_side_t                      side,
             const diag_type_t                      diag,
             const matop_t                          op_D,
             const Hpro::TDenseMatrix< value_t > &  D,
             Hpro::TRkMatrix< value_t > &           M )
{
    if ( diag == unit_diag )
        return;

    //
    // see blocked D version above
    //
    
    if ( side == from_left )
    {
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_U( M ) );

        solve_diag< value_t >( side, diag, op_D, D, U );
    }// if
    else
    {
        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_V( M ) );

        solve_diag< value_t >( from_left, diag, blas::adjoint( op_D ), D, V );
    }// else
}

template < typename value_t >
void
solve_diag ( const eval_side_t                      side,
             const diag_type_t                      diag,
             const matop_t                          op_D,
             const Hpro::TDenseMatrix< value_t > &  D,
             matrix::lrsmatrix< value_t > &         M )
{
    if ( diag == unit_diag )
        return;
    
    //
    // see blocked D version above
    //
    
    if ( side == from_left )
    {
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), M.U() );

        solve_diag< value_t >( side, diag, op_D, D, U );
    }// if
    else
    {
        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), M.V() );

        solve_diag< value_t >( from_left, diag, blas::adjoint( op_D ), D, V );
    }// else
}

////////////////////////////////////////////////////////////////////////////////
//
// matrix solving with lower triangular matrix L
//
// solve L·X = M (from_left) or X·L = M (from_right)
// - on exit, M contains X
//
////////////////////////////////////////////////////////////////////////////////

//
// forward for general version
//
template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const Hpro::TMatrix< value_t > &    L,
                  Hpro::TMatrix< value_t > &          M,
                  const Hpro::TTruncAcc &             acc,
                  const approx_t &                    approx );

//
// forward for general version without (!) approximation
//
template < typename value_t >
void
solve_lower_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const Hpro::TMatrix< value_t > &    L,
                  Hpro::TMatrix< value_t > &          M );

//
// specializations
//
template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  L,
                  Hpro::TBlockMatrix< value_t > &        M,
                  const Hpro::TTruncAcc &                acc,
                  const approx_t &                       approx )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( B %d, B %d )", L.id(), M.id() ) );

    if ( side == from_left )
    {
        auto  nbr = M.nblock_rows();
        auto  nbc = M.nblock_cols();

        HLR_ASSERT( ( L.nblock_rows() == nbr ) && ( L.nblock_cols() == nbr ) );
            
        if ( is_nd( L ) )
        {
            //
            // exploiting nested disection structure:
            //
            //   no updates for solve only with diagonal blocks and update
            //
            
            for ( auto  i = 0; i < int(nbr-1); ++i )
            {
                for ( auto  j = 0; j < int(nbc); ++j )
                {
                    auto  L_ii = L.block( i, i );
                    auto  M_ij = M.block( i, j );

                    HLR_ASSERT( ! is_null( L_ii ) );
                            
                    if ( ! is_null( M_ij ) )
                    {
                        solve_lower_tri< value_t >( side, diag, *L_ii, *M_ij, acc, approx );

                        if ( ! is_null( L.block( nbr-1, i ) ) )
                        {
                            HLR_ASSERT( ! is_null( M.block( nbr-1, j ) ) );
                                    
                            multiply< value_t >( value_t(-1),
                                                 apply_normal, *L.block( nbr-1, i ),
                                                 apply_normal, *M_ij,
                                                 *M.block( nbr-1, j ), acc, approx );
                        }// if
                    }// if
                }// for
            }// for

            HLR_ASSERT( ! is_null( L.block( nbr-1, nbr-1 ) ) );
            
            for ( auto  j = 0; j < int(nbc); ++j )
            {
                auto  M_ij = M.block( nbr-1, j );
                
                if ( ! is_null( M_ij ) )
                    solve_lower_tri< value_t >( side, diag, *L.block( nbr-1, nbr-1 ), *M_ij, acc, approx );
            }// for
        }// if
        else
        {
            //
            // from top to bottom in L
            // - solve in current block row
            // - update matrices in remaining block rows
            //
        
            for ( uint i = 0; i < nbr; ++i )
            {
                const auto  L_ii = L.block( i, i );

                HLR_ASSERT( ! is_null( L_ii ) );
            
                for ( uint j = 0; j < nbc; ++j )
                {
                    auto  M_ij = M.block( i, j );
                
                    if ( ! is_null( M_ij ) )
                        solve_lower_tri< value_t >( side, diag, *L_ii, *M_ij, acc, approx );
                }// for

                for ( uint  k = i+1; k < nbr; ++k )
                {
                    for ( uint  j = 0; j < nbc; ++j )
                    {
                        if ( ! is_null_any( L.block(k,i), M.block(i,j) ) )
                        {
                            HLR_ASSERT( ! is_null( M.block(k,j) ) );
                        
                            multiply< value_t >( value_t(-1),
                                                 apply_normal, *L.block(k,i),
                                                 apply_normal, *M.block(i,j),
                                                 *M.block(k,j), acc, approx );
                        }// if
                    }// for
                }// for
            }// for
        }// else
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  L,
                  Hpro::TDenseMatrix< value_t > &        M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( B %d, D %d )", L.id(), M.id() ) );

    if ( side == from_left )
    {
        //
        // from top to bottom in L
        // - solve in current block row
        // - update matrices in remaining block rows
        //
        
        for ( uint j = 0; j < L.nblock_cols(); ++j )
        {
            const auto  L_jj = L.block( j, j );

            HLR_ASSERT( ! is_null( L_jj ) );

            auto  M_j = blas::matrix< value_t >( blas::mat( M ),
                                                 L_jj->col_is() - L.col_ofs(),
                                                 blas::range::all );
            auto  D_j = Hpro::TDenseMatrix< value_t >( L_jj->col_is(), M.col_is(), M_j );

            solve_lower_tri< value_t >( side, diag, *L_jj, D_j );

            for ( uint  k = j+1; k < L.nblock_rows(); ++k )
            {
                auto  L_kj = L.block( k, j );
                
                if ( ! is_null( L_kj ) )
                {
                    auto  M_k = blas::matrix< value_t >( blas::mat( M ),
                                                         L_kj->row_is() - L.row_ofs(),
                                                         blas::range::all );
                    auto  D_k = Hpro::TDenseMatrix< value_t >( L_kj->row_is(), M.col_is(), M_k );
                    
                    multiply< value_t >( value_t(-1),
                                         apply_normal, *L_kj,
                                         apply_normal, D_j,
                                         D_k );
                }// for
            }// for
        }// for
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  L,
                  matrix::dense_matrix< value_t > &      M,
                  const Hpro::TTruncAcc &                acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( B %d, D %d )", L.id(), M.id() ) );

    if ( side == from_left )
    {
        auto  lock = std::scoped_lock( M.mutex() );

        M.decompress();
        
        //
        // from top to bottom in L
        // - solve in current block row
        // - update matrices in remaining block rows
        //
        
        for ( uint j = 0; j < L.nblock_cols(); ++j )
        {
            const auto  L_jj = L.block( j, j );

            HLR_ASSERT( ! is_null( L_jj ) );

            auto  M_j = blas::matrix< value_t >( blas::mat( M ),
                                                 L_jj->col_is() - L.col_ofs(),
                                                 blas::range::all );
            auto  D_j = Hpro::TDenseMatrix< value_t >( L_jj->col_is(), M.col_is(), M_j );

            solve_lower_tri< value_t >( side, diag, *L_jj, D_j );

            for ( uint  k = j+1; k < L.nblock_rows(); ++k )
            {
                auto  L_kj = L.block( k, j );
                
                if ( ! is_null( L_kj ) )
                {
                    auto  M_k = blas::matrix< value_t >( blas::mat( M ),
                                                         L_kj->row_is() - L.row_ofs(),
                                                         blas::range::all );
                    auto  D_k = Hpro::TDenseMatrix< value_t >( L_kj->row_is(), M.col_is(), M_k );
                    
                    multiply< value_t >( value_t(-1),
                                         apply_normal, *L_kj,
                                         apply_normal, D_j,
                                         D_k );
                }// for
            }// for
        }// for

        M.compress( acc );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  L,
                  Hpro::TRkMatrix< value_t > &           M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( B %d, R %d )", L.id(), M.id() ) );

    if ( side == from_left )
    {
        //
        // solve L M = L U(X) V(X)' = U(M) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_U( M ) );

        solve_lower_tri< value_t >( side, diag, L, U );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  L,
                  matrix::lrmatrix< value_t > &          M,
                  const Hpro::TTruncAcc &                acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( B %d, R %d )", L.id(), M.id() ) );

    if ( side == from_left )
    {
        auto  lock = std::scoped_lock( M.mutex() );

        // TODO: just decompress U(M)
        
        M.decompress();
        
        //
        // solve L M = L U(X) V(X)' = U(M) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_U( M ) );

        solve_lower_tri< value_t >( side, diag, L, U );

        M.compress( acc );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  L,
                  matrix::lrsmatrix< value_t > &         M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( B %d, R %d )", L.id(), M.id() ) );

    if ( side == from_left )
    {
        //
        // solve L M = L U(X) S(X) V(X)' = U(M) S(M) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), M.U() );

        solve_lower_tri< value_t >( side, diag, L, U );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  /* L */,
                  Hpro::TBlockMatrix< value_t > &        /* M */,
                  const Hpro::TTruncAcc &                /* acc */,
                  const approx_t &                       /* approx */ )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( D %d, B %d )", L.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        HLR_ERROR( "todo" );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  L,
                  Hpro::TDenseMatrix< value_t > &        M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( D %d, D %d )", L.id(), M.id() ) );
    
    // Hpro::DBG::write( L, "L.mat", "L" );
    // Hpro::DBG::write( M, "M.mat", "M" );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        //
        // blockwise evaluation and L is assumed to hold L^-1
        // TODO: option argument?
        //

        auto  Mc = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), blas::mat( L ), Mc, value_t(0), blas::mat( M ) );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else

    // Hpro::DBG::write( M, "X.mat", "X" );
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  L,
                  matrix::dense_matrix< value_t > &      M,
                  const Hpro::TTruncAcc &                acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( D %d, D %d )", L.id(), M.id() ) );
    
    // Hpro::DBG::write( L, "L.mat", "L" );
    // Hpro::DBG::write( M, "M.mat", "M" );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        auto  lock = std::scoped_lock( M.mutex() );

        M.decompress();
        
        //
        // blockwise evaluation and L is assumed to hold L^-1
        // TODO: option argument?
        //

        auto  Mc = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), blas::mat( L ), Mc, value_t(0), blas::mat( M ) );

        M.compress( acc );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else

    // Hpro::DBG::write( M, "X.mat", "X" );
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                        side,
                  const diag_type_t                        diag,
                  const matrix::dense_matrix< value_t > &  L,
                  Hpro::TDenseMatrix< value_t > &          M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( D %d, D %d )", L.id(), M.id() ) );
    
    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        //
        // blockwise evaluation and L is assumed to hold L^-1
        // TODO: option argument?
        //

        const auto  DL = L.mat_decompressed();
        auto        Mc = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), DL, Mc, value_t(0), blas::mat( M ) );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else

    // Hpro::DBG::write( M, "X.mat", "X" );
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                        side,
                  const diag_type_t                        diag,
                  const matrix::dense_matrix< value_t > &  L,
                  matrix::dense_matrix< value_t > &        M,
                  const Hpro::TTruncAcc &                  acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( D %d, D %d )", L.id(), M.id() ) );
    
    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        auto  lock = std::scoped_lock( M.mutex() );

        M.decompress();
        
        //
        // blockwise evaluation and L is assumed to hold L^-1
        // TODO: option argument?
        //

        const auto  DL = L.mat_decompressed();
        auto        Mc = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), DL, Mc, value_t(0), blas::mat( M ) );

        M.compress( acc );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else

    // Hpro::DBG::write( M, "X.mat", "X" );
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  L,
                  Hpro::TRkMatrix< value_t > &           M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( D %d, R %d )", L.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        //
        // solve L M = L U(X) V(X)' = U(M) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_U( M ) );

        solve_lower_tri< value_t >( side, diag, L, U );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  L,
                  matrix::lrmatrix< value_t > &          M,
                  const Hpro::TTruncAcc &                acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( D %d, R %d )", L.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        auto  lock = std::scoped_lock( M.mutex() );

        // TODO: just decompress U(M)
        M.decompress();
        
        //
        // solve L M = L U(X) V(X)' = U(M) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_U( M ) );

        solve_lower_tri< value_t >( side, diag, L, U );

        M.compress( acc );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                        side,
                  const diag_type_t                        diag,
                  const matrix::dense_matrix< value_t > &  L,
                  Hpro::TRkMatrix< value_t > &             M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( D %d, R %d )", L.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        //
        // solve L M = L U(X) V(X)' = U(M) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_U( M ) );

        solve_lower_tri< value_t >( side, diag, L, U );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                        side,
                  const diag_type_t                        diag,
                  const matrix::dense_matrix< value_t > &  L,
                  matrix::lrmatrix< value_t > &            M,
                  const Hpro::TTruncAcc &                  acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( D %d, R %d )", L.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        auto  lock = std::scoped_lock( M.mutex() );

        // TODO: just decompress U(M)
        M.decompress();
        
        //
        // solve L M = L U(X) V(X)' = U(M) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_U( M ) );

        solve_lower_tri< value_t >( side, diag, L, U );

        M.compress( acc );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

template < typename value_t >
void
solve_lower_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  L,
                  matrix::lrsmatrix< value_t > &         M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_lower_tri( D %d, R %d )", L.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        //
        // solve L M = L U(X) S(X) V(X)' = U(M) S(X) V(M)'
        // as L U(X) = U(M)
        //
        
        auto  U = Hpro::TDenseMatrix< value_t >( M.row_is(), Hpro::is( 0, M.rank()-1 ), M.U() );

        solve_lower_tri< value_t >( side, diag, L, U );
    }// if
    else
    {
        HLR_ASSERT( false );
    }// else
}

////////////////////////////////////////////////////////////////////////////////
//
// matrix solving with upper triangular matrix U
//
// solve U·X = M or X·U = M 
// - on exit, M contains X
//
////////////////////////////////////////////////////////////////////////////////

//
// forward for general version
//
template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const Hpro::TMatrix< value_t > &    U,
                  Hpro::TMatrix< value_t > &          M,
                  const Hpro::TTruncAcc &             acc,
                  const approx_t &                    approx );

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const Hpro::TMatrix< value_t > &    U,
                  Hpro::TMatrix< value_t > &          M );

//
// specializations
//
template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  U,
                  Hpro::TBlockMatrix< value_t > &        M,
                  const Hpro::TTruncAcc &                acc,
                  const approx_t &                       approx )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( B %d, B %d )", U.id(), M.id() ) );
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        for ( uint j = 0; j < M.nblock_cols(); ++j )
        {
            const auto  U_jj = U.block( j, j );

            HLR_ASSERT( ! is_null( U_jj ) );
            
            for ( uint i = 0; i < M.nblock_rows(); ++i )
            {
                auto  M_ij = M.block( i, j );
                
                if ( ! is_null( M_ij ) )
                    solve_upper_tri< value_t >( side, diag, *U_jj, *M_ij, acc, approx );
            }// for
            
            for ( uint  k = j+1; k < M.nblock_cols(); ++k )
            {
                for ( uint  i = 0; i < M.nblock_rows(); ++i )
                {
                    if ( ! is_null_any( M.block(i,j), U.block(j,k) ) )
                    {
                        HLR_ASSERT( ! is_null( M.block(i,k) ) );
                        
                        multiply< value_t >( value_t(-1),
                                             apply_normal, *M.block(i,j),
                                             apply_normal, *U.block(j,k),
                                             *M.block(i,k), acc, approx );
                    }// if
                }// for
            }// for
        }// for
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  U,
                  Hpro::TDenseMatrix< value_t > &        M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( B %d, D %d )", U.id(), M.id() ) );

    if ( side == from_left )
    {
        //
        // assumption: U' is used
        //

        const matop_t  op_U = apply_adjoint;
        
        for ( uint j = 0; j < U.nblock_cols( op_U ); ++j )
        {
            const auto  U_jj = U.block( j, j, op_U );

            HLR_ASSERT( ! is_null( U_jj ) );

            auto  M_j = blas::matrix< value_t >( blas::mat( M ),
                                                 U_jj->col_is( op_U ) - U.col_ofs( op_U ),
                                                 blas::range::all );
            auto  D_j = Hpro::TDenseMatrix< value_t >( U_jj->col_is( op_U ), M.col_is(), M_j );
            
            solve_upper_tri< value_t >( side, diag, *U_jj, D_j );
            
            for ( uint  k = j+1; k < U.nblock_rows( op_U ); ++k )
            {
                auto  U_kj = U.block( k, j, op_U );
                    
                if ( ! is_null( U_kj ) )
                {
                    auto  M_k = blas::matrix< value_t >( blas::mat( M ),
                                                         U_kj->row_is( op_U ) - U.row_ofs( op_U ),
                                                         blas::range::all );
                    auto  D_k = Hpro::TDenseMatrix< value_t >( U_kj->row_is( op_U ), M.col_is(), M_k );
            
                    multiply< value_t >( value_t(-1), op_U, *U_kj, apply_normal, D_j, D_k );
                }// if
            }// for
        }// for
    }// if
    else
    {
        for ( uint i = 0; i < U.nblock_rows(); ++i )
        {
            const auto  U_ii = U.block( i, i );

            HLR_ASSERT( ! is_null( U_ii ) );

            auto  M_i = blas::matrix< value_t >( blas::mat( M ),
                                                 blas::range::all,
                                                 U_ii->row_is() - U.row_ofs() );
            auto  D_i = Hpro::TDenseMatrix< value_t >( M.row_is(), U_ii->row_is(), M_i );
            
            solve_upper_tri< value_t >( side, diag, *U_ii, D_i );
            
            for ( uint  k = i+1; k < U.nblock_cols(); ++k )
            {
                auto  U_ik = U.block(i,k);
                    
                if ( ! is_null( U_ik ) )
                {
                    auto  M_k = blas::matrix< value_t >( blas::mat( M ),
                                                         blas::range::all,
                                                         U_ik->col_is() - U.col_ofs() );
                    auto  D_k = Hpro::TDenseMatrix< value_t >( M.row_is(), U_ik->col_is(), M_k );
            
                    multiply< value_t >( value_t(-1), apply_normal, D_i, apply_normal, *U_ik, D_k );
                }// if
            }// for
        }// for
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  U,
                  matrix::dense_matrix< value_t > &      M,
                  const Hpro::TTruncAcc &                acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( B %d, D %d )", U.id(), M.id() ) );

    auto  lock = std::scoped_lock( M.mutex() );

    M.decompress();
        
    if ( side == from_left )
    {
        //
        // assumption: U' is used
        //

        const matop_t  op_U = apply_adjoint;
        
        for ( uint j = 0; j < U.nblock_cols( op_U ); ++j )
        {
            const auto  U_jj = U.block( j, j, op_U );

            HLR_ASSERT( ! is_null( U_jj ) );

            auto  M_j = blas::matrix< value_t >( blas::mat( M ),
                                                 U_jj->col_is( op_U ) - U.col_ofs( op_U ),
                                                 blas::range::all );
            auto  D_j = Hpro::TDenseMatrix< value_t >( U_jj->col_is( op_U ), M.col_is(), M_j );
            
            solve_upper_tri< value_t >( side, diag, *U_jj, D_j );
            
            for ( uint  k = j+1; k < U.nblock_rows( op_U ); ++k )
            {
                auto  U_kj = U.block( k, j, op_U );
                    
                if ( ! is_null( U_kj ) )
                {
                    auto  M_k = blas::matrix< value_t >( blas::mat( M ),
                                                         U_kj->row_is( op_U ) - U.row_ofs( op_U ),
                                                         blas::range::all );
                    auto  D_k = Hpro::TDenseMatrix< value_t >( U_kj->row_is( op_U ), M.col_is(), M_k );
            
                    multiply< value_t >( value_t(-1), op_U, *U_kj, apply_normal, D_j, D_k );
                }// if
            }// for
        }// for
    }// if
    else
    {
        for ( uint i = 0; i < U.nblock_rows(); ++i )
        {
            const auto  U_ii = U.block( i, i );

            HLR_ASSERT( ! is_null( U_ii ) );

            auto  M_i = blas::matrix< value_t >( blas::mat( M ),
                                                 blas::range::all,
                                                 U_ii->row_is() - U.row_ofs() );
            auto  D_i = Hpro::TDenseMatrix< value_t >( M.row_is(), U_ii->row_is(), M_i );
            
            solve_upper_tri< value_t >( side, diag, *U_ii, D_i );
            
            for ( uint  k = i+1; k < U.nblock_cols(); ++k )
            {
                auto  U_ik = U.block(i,k);
                    
                if ( ! is_null( U_ik ) )
                {
                    auto  M_k = blas::matrix< value_t >( blas::mat( M ),
                                                         blas::range::all,
                                                         U_ik->col_is() - U.col_ofs() );
                    auto  D_k = Hpro::TDenseMatrix< value_t >( M.row_is(), U_ik->col_is(), M_k );
            
                    multiply< value_t >( value_t(-1), apply_normal, D_i, apply_normal, *U_ik, D_k );
                }// if
            }// for
        }// for
    }// else

    M.compress( acc );
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  U,
                  Hpro::TRkMatrix< value_t > &           M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( B %d, R %d )", U.id(), M.id() ) );

    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve X U = U(X) V(X)' U = U(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_V( M ) );

        solve_upper_tri< value_t >( from_left, diag, U, V );
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  U,
                  matrix::lrmatrix< value_t > &          M,
                  const Hpro::TTruncAcc &                acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( B %d, R %d )", U.id(), M.id() ) );

    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        auto  lock = std::scoped_lock( M.mutex() );

        // TODO: just decompress V(M)
        M.decompress();
        
        //
        // solve X U = U(X) V(X)' U = U(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_V( M ) );

        solve_upper_tri< value_t >( from_left, diag, U, V );

        M.compress( acc );
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TBlockMatrix< value_t > &  U,
                  matrix::lrsmatrix< value_t > &         M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( B %d, R %d )", U.id(), M.id() ) );

    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve X U = U(X) S(X) V(X)' U = U(M) S(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), M.V() );

        solve_upper_tri< value_t >( from_left, diag, U, V );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  /* U */,
                  Hpro::TBlockMatrix< value_t > &        /* M */,
                  const Hpro::TTruncAcc &                /* acc */,
                  const approx_t &                       /* approx */ )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( D %d, B %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        HLR_ERROR( "todo" );
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  U,
                  Hpro::TDenseMatrix< value_t > &        M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( D %d, D %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    //
    // blockwise evaluation and U is assumed to hold L^-1
    // TODO: option argument?
    //

    if ( side == from_left )
    {
        // assuming U' X = M
        const matop_t  op_U = apply_adjoint;
        auto           Mc   = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), blas::mat_view( op_U, blas::mat( U ) ), Mc, value_t(0), blas::mat( M ) );
    }// if
    else
    {
        auto  Mc = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), Mc, blas::mat( U ), value_t(0), blas::mat( M ) );
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  U,
                  matrix::dense_matrix< value_t > &      M,
                  const Hpro::TTruncAcc &                acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( D %d, D %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    //
    // blockwise evaluation and U is assumed to hold L^-1
    // TODO: option argument?
    //

    auto  lock = std::scoped_lock( M.mutex() );

    M.decompress();
        
    if ( side == from_left )
    {
        // assuming U' X = M
        const matop_t  op_U = apply_adjoint;
        auto           Mc   = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), blas::mat_view( op_U, blas::mat( U ) ), Mc, value_t(0), blas::mat( M ) );
    }// if
    else
    {
        auto  Mc = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), Mc, blas::mat( U ), value_t(0), blas::mat( M ) );
    }// else

    M.compress( acc );
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                        side,
                  const diag_type_t                        diag,
                  const matrix::dense_matrix< value_t > &  U,
                  Hpro::TDenseMatrix< value_t > &          M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( D %d, D %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    //
    // blockwise evaluation and U is assumed to hold L^-1
    // TODO: option argument?
    //

    const auto  DU = U.mat_decompressed();
    
    if ( side == from_left )
    {
        // assuming U' X = M
        const matop_t  op_U = apply_adjoint;
        auto           Mc   = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), blas::mat_view( op_U, DU ), Mc, value_t(0), blas::mat( M ) );
    }// if
    else
    {
        auto  Mc = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), Mc, DU, value_t(0), blas::mat( M ) );
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                        side,
                  const diag_type_t                        diag,
                  const matrix::dense_matrix< value_t > &  U,
                  matrix::dense_matrix< value_t > &        M,
                  const Hpro::TTruncAcc &                  acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( D %d, D %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    auto  lock = std::scoped_lock( M.mutex() );

    M.decompress();
        
    //
    // blockwise evaluation and U is assumed to hold L^-1
    // TODO: option argument?
    //

    const auto  DU = U.mat_decompressed();
    
    if ( side == from_left )
    {
        // assuming U' X = M
        const matop_t  op_U = apply_adjoint;
        auto           Mc   = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), blas::mat_view( op_U, DU ), Mc, value_t(0), blas::mat( M ) );
    }// if
    else
    {
        auto  Mc = blas::copy( blas::mat( M ) );

        blas::prod( value_t(1), Mc, DU, value_t(0), blas::mat( M ) );
    }// else

    M.compress( acc );
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  U,
                  Hpro::TRkMatrix< value_t > &           M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( D %d, R %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve X U = U(X) V(X)' U = U(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_V( M ) );

        solve_upper_tri< value_t >( from_left, diag, U, V );
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  U,
                  matrix::lrmatrix< value_t > &          M,
                  const Hpro::TTruncAcc &                acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( D %d, R %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        auto  lock = std::scoped_lock( M.mutex() );

        // TODO: just decompress V(M)
        M.decompress();
        
        //
        // solve X U = U(X) V(X)' U = U(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_V( M ) );

        solve_upper_tri< value_t >( from_left, diag, U, V );

        M.compress( acc );
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                        side,
                  const diag_type_t                        diag,
                  const matrix::dense_matrix< value_t > &  U,
                  Hpro::TRkMatrix< value_t > &             M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( D %d, R %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve X U = U(X) V(X)' U = U(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_V( M ) );

        solve_upper_tri< value_t >( from_left, diag, U, V );
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                        side,
                  const diag_type_t                        diag,
                  const matrix::dense_matrix< value_t > &  U,
                  matrix::lrmatrix< value_t > &            M,
                  const Hpro::TTruncAcc &                  acc )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( D %d, R %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        auto  lock = std::scoped_lock( M.mutex() );

        // TODO: just decompress V(M)
        M.decompress();
        
        //
        // solve X U = U(X) V(X)' U = U(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), blas::mat_V( M ) );

        solve_upper_tri< value_t >( from_left, diag, U, V );

        M.compress( acc );
    }// else
}

template < typename value_t >
void
solve_upper_tri ( const eval_side_t                      side,
                  const diag_type_t                      diag,
                  const Hpro::TDenseMatrix< value_t > &  U,
                  matrix::lrsmatrix< value_t > &         M )
{
    HLR_SOLVE_PRINT( Hpro::to_string( "solve_upper_tri( D %d, R %d )", U.id(), M.id() ) );

    if ( diag == unit_diag )
        return;
    
    if ( side == from_left )
    {
        HLR_ASSERT( false );
    }// if
    else
    {
        //
        // solve X U = U(X) S(X) V(X)' U = U(M) S(M) V(M)' = M
        // as V(X)' U = V(M)' or
        //    U' V(X) = V(M), respectively
        //

        auto  V = Hpro::TDenseMatrix< value_t >( M.col_is(), Hpro::is( 0, M.rank()-1 ), M.V() );

        solve_upper_tri< value_t >( from_left, diag, U, V );
    }// else
}

}// namespace hlr

#endif // __HLR_ARITH_DETAIL_SOLVE_HH

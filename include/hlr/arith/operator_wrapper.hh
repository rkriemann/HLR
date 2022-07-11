#ifndef __HLR_ARITH_OPERATOR_WRAPPER_HH
#define __HLR_ARITH_OPERATOR_WRAPPER_HH
//
// Project     : HLR
// Module      : arith/operator_wrapper
// Description : wrapper functions for some standard operators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hpro/cluster/TIndexSet.hh>

#include <hlr/arith/blas.hh>
#include <hlr/arith/operator_wrapper.hh>
#include <hlr/utils/log.hh>

namespace hlr
{

//////////////////////////////////////////////////////////////////////
//
// wrapper functions for blas::matrix
//
//////////////////////////////////////////////////////////////////////

template < typename value_t > size_t nrows ( const blas::matrix< value_t > &  M ) { return M.nrows(); }
template < typename value_t > size_t ncols ( const blas::matrix< value_t > &  M ) { return M.ncols(); }

template < typename value_t >
blas::vector< value_t >
get_row ( const blas::matrix< value_t > &  M,
          const size_t                     i )
{
    return blas::copy( M.row( i ) );
}

template < typename value_t >
blas::vector< value_t >
get_column ( const blas::matrix< value_t > &  M,
             const size_t                     j )
{
    return blas::copy( M.column( j ) );
}

template < typename value_t >
void
prod ( const value_t                    alpha,
       const matop_t                    op_M,
       const blas::matrix< value_t > &  M,
       const blas::vector< value_t > &  x,
       blas::vector< value_t > &        y )
{
    blas::mulvec( alpha, blas::mat_view( op_M, M ), x, value_t(1), y );
}

template < typename value_t >
void
prod ( const value_t                    alpha,
       const matop_t                    op_M,
       const blas::matrix< value_t > &  M,
       const blas::matrix< value_t > &  X,
       blas::matrix< value_t > &        Y )
{
    blas::prod( alpha, blas::mat_view( op_M, M ), X, value_t(1), Y );
}

////////////////////////////////////////////////////////////////////////////////
//
// operator wrapper for sum of matrices
//
////////////////////////////////////////////////////////////////////////////////

template < typename T_value >
struct matrixsum_operator
{
    using  value_t = T_value;

    const std::list< blas::matrix< value_t > > &  M;

    matrixsum_operator ( const std::list< blas::matrix< value_t > > &  aM )
            : M( aM )
    {
        HLR_ASSERT( ! aM.empty() );
    }
    
    size_t  nrows () const { return M.front().nrows(); }
    size_t  ncols () const { return M.front().ncols(); }

    blas::vector< value_t >
    get_row ( const size_t  j ) const
    {
        auto  row = blas::vector< value_t >( ncols() );

        for ( auto &  M_i : M )
        {
            auto  M_i_j = M_i.row( j );

            blas::add( value_t(1), M_i_j, row );
        }// for
        
        return row;
    }

    blas::vector< value_t >
    get_column ( const size_t  j ) const
    {
        auto  col = blas::vector< value_t >( nrows() );

        for ( auto &  M_i : M )
        {
            auto  M_i_j = M_i.col( j );

            blas::add( value_t(1), M_i_j, col );
        }// for
        
        return col;
    }
};

template < typename value_t > size_t nrows ( const matrixsum_operator< value_t > &  op ) { return op.nrows(); }
template < typename value_t > size_t ncols ( const matrixsum_operator< value_t > &  op ) { return op.ncols(); }

template < typename value_t >
blas::vector< value_t >
get_row ( const matrixsum_operator< value_t > &  op,
          const size_t                           i )
{
    return op.get_row( i );
}

template < typename value_t >
blas::vector< value_t >
get_column ( const matrixsum_operator< value_t > &  op,
             const size_t                           j )
{
    return op.get_column( j );
}

template < typename value_t >
void
prod ( const value_t                          alpha,
       const matop_t                          op_M,
       const matrixsum_operator< value_t > &  op,
       const blas::vector< value_t > &        x,
       blas::vector< value_t > &              y )
{
    for ( auto &  M_i : op.M )
        blas::mulvec( alpha, matrix_view( op_M, M_i ), x, value_t(1), y );
}

template < typename value_t >
void
prod ( const value_t                          alpha,
       const matop_t                          op_M,
       const matrixsum_operator< value_t > &  op,
       const blas::matrix< value_t > &        X,
       blas::matrix< value_t > &              Y )
{
    for ( auto &  M_i : op.M )
        blas::prod( alpha, matrix_view( op_M, M_i ), X, value_t(1), Y );
}

template < typename value_t >
matrixsum_operator< value_t >
operator_wrapper ( const std::list< blas::matrix< value_t > > &  M )
{
    return matrixsum_operator< value_t > ( M );
}

////////////////////////////////////////////////////////////////////////////////
//
// operator wrapper for factorized lowrank matrix
//
////////////////////////////////////////////////////////////////////////////////

template < typename T_value >
struct lowrank_operator
{
    using  value_t = T_value;

    const blas::matrix< value_t >  U;
    const blas::matrix< value_t >  V;

    lowrank_operator ( const blas::matrix< value_t > &  aU,
                       const blas::matrix< value_t > &  aV )
            : U( aU )
            , V( aV )
    {
        HLR_ASSERT( U.ncols() == V.ncols() );
    }
    
    size_t  nrows () const { return U.nrows(); }
    size_t  ncols () const { return V.nrows(); }

    blas::vector< value_t >
    get_row ( const size_t  i ) const
    {
        // M(i,:) = U(i,:) · V^H = (U(i,:) · V^H)^H^H = (V · U(i,:)^H)^H
        auto  U_i = std::move( blas::copy( U.row( i ) ) );

        blas::conj( U_i );
        
        auto  row = blas::mulvec( value_t(1), V, U_i );

        blas::conj( row );

        return row;
    }

    blas::vector< value_t >
    get_column ( const size_t  j ) const
    {
        // M(:,j) = U · V(j,:)^H
        auto  V_j = std::move( blas::copy( V.row( j ) ) );

        blas::conj( V_j );
        
        return blas::mulvec( value_t(1), U, V_j );
    }
};

template < typename value_t > size_t nrows ( const lowrank_operator< value_t > &  op ) { return op.nrows(); }
template < typename value_t > size_t ncols ( const lowrank_operator< value_t > &  op ) { return op.ncols(); }

template < typename value_t >
blas::vector< value_t >
get_row ( const lowrank_operator< value_t > &  op,
          const size_t                         i )
{
    return op.get_row( i );
}

template < typename value_t >
blas::vector< value_t >
get_column ( const lowrank_operator< value_t > &  op,
             const size_t                         j )
{
    return op.get_column( j );
}

template < typename value_t >
void
prod ( const value_t                        alpha,
       const matop_t                        op_M,
       const lowrank_operator< value_t > &  op,
       const blas::vector< value_t > &      x,
       blas::vector< value_t > &            y )
{
    if ( op.U.ncols() == 0 )
        return;
    
    switch ( op_M )
    {
        case Hpro::apply_normal :
        {
            const auto  t = blas::mulvec( alpha, blas::adjoint( op.V ), x );

            blas::mulvec( value_t(1), op.U, t, value_t(1), y );
        }
        break;

        case Hpro::apply_conjugate :
            HLR_ERROR( "todo" );
            break;
        
        case Hpro::apply_transposed :
            HLR_ERROR( "todo" );
            break;
        
        case Hpro::apply_adjoint :
        {
            const auto  t = blas::mulvec( alpha, blas::adjoint( op.U ), x );

            blas::mulvec( value_t(1), op.V, t, value_t(1), y );
        }
        break;
    }// if
}

template < typename value_t >
void
prod ( const value_t                        alpha,
       const matop_t                        op_M,
       const lowrank_operator< value_t > &  op,
       const blas::matrix< value_t > &      X,
       blas::matrix< value_t > &            Y )
{
    if ( op.U.ncols() == 0 )
        return;
    
    switch ( op_M )
    {
        case Hpro::apply_normal :
        {
            const auto  T = blas::prod( alpha, blas::adjoint( op.V ), X );

            blas::prod( value_t(1), op.U, T, value_t(1), Y );
        }
        break;

        case Hpro::apply_conjugate :
            HLR_ERROR( "todo" );
            break;
        
        case Hpro::apply_transposed :
            HLR_ERROR( "todo" );
            break;
        
        case Hpro::apply_adjoint :
        {
            const auto  T = blas::prod( alpha, blas::adjoint( op.U ), X );

            blas::prod( value_t(1), op.V, T, value_t(1), Y );
        }
        break;
    }// if
}

template < typename value_t >
lowrank_operator< value_t >
operator_wrapper ( const blas::matrix< value_t > &  U,
                   const blas::matrix< value_t > &  V )
{
    return lowrank_operator< value_t > ( U, V );
}

////////////////////////////////////////////////////////////////////////////////
//
// operator wrapper for som of factorized lowrank matrices
//
////////////////////////////////////////////////////////////////////////////////

template < typename T_value >
struct lowranksum_operator
{
    using  value_t = T_value;

    const std::list< blas::matrix< value_t > > &  U;
    const std::list< blas::matrix< value_t > > &  V;

    lowranksum_operator ( const std::list< blas::matrix< value_t > > &  aU,
                          const std::list< blas::matrix< value_t > > &  aV )
            : U( aU )
            , V( aV )
    {
        HLR_ASSERT( U.size() == V.size() );
        HLR_ASSERT( ! U.empty() );
    }
    
    size_t  nrows () const { return U.front().nrows(); }
    size_t  ncols () const { return V.front().nrows(); }

    blas::vector< value_t >
    get_row ( const size_t  j ) const
    {
        // M(j,:) = Σ_i U_i(j,:) · V_i' = ( Σ_i (U_i(j,:) · V_i')' )' = ( Σ_i V_i · U_i(j,:)' )'
        auto  row = blas::vector< value_t >( ncols() );
        auto  U_i = U.cbegin();
        auto  V_i = V.cbegin();

        for ( ; ( U_i != U.cend() ) && ( V_i != V.cend() ); ++U_i, ++V_i )
        {
            if ( (*U_i).ncols() == 0 )
                continue;
            
            auto  U_i_j = std::move( blas::copy( (*U_i).row( j ) ) );

            blas::conj( U_i_j );
        
            blas::mulvec( value_t(1), *V_i, U_i_j, value_t(1), row );
        }// for
        
        blas::conj( row );

        return row;
    }

    blas::vector< value_t >
    get_column ( const size_t  j ) const
    {
        // M(:,j) = Σ_i U_i · V_i(j,:)^H
        auto  col = blas::vector< value_t >( nrows() );
        auto  U_i = U.cbegin();
        auto  V_i = V.cbegin();

        for ( ; ( U_i != U.cend() ) && ( V_i != V.cend() ); ++U_i, ++V_i )
        {
            if ( (*U_i).ncols() == 0 )
                continue;
            
            auto  V_i_j = std::move( blas::copy( (*V_i).row( j ) ) );

            blas::conj( V_i_j );
        
            blas::mulvec( value_t(1), *U_i, V_i_j, value_t(1), col );
        }// for

        return col;
    }
};

template < typename value_t > size_t nrows ( const lowranksum_operator< value_t > &  op ) { return op.nrows(); }
template < typename value_t > size_t ncols ( const lowranksum_operator< value_t > &  op ) { return op.ncols(); }

template < typename value_t >
blas::vector< value_t >
get_row ( const lowranksum_operator< value_t > &  op,
          const size_t                            i )
{
    return op.get_row( i );
}

template < typename value_t >
blas::vector< value_t >
get_column ( const lowranksum_operator< value_t > &  op,
             const size_t                            j )
{
    return op.get_column( j );
}

template < typename value_t >
void
prod ( const value_t                           alpha,
       const matop_t                           op_M,
       const lowranksum_operator< value_t > &  op,
       const blas::vector< value_t > &         x,
       blas::vector< value_t > &               y )
{
    switch ( op_M )
    {
        case Hpro::apply_normal :
        {
            auto  U_i = op.U.cbegin();
            auto  V_i = op.V.cbegin();

            for ( ; ( U_i != op.U.cend() ) && ( V_i != op.V.cend() ); ++U_i, ++V_i )
            {
                if ( (*U_i).ncols() == 0 )
                    continue;
            
                const auto  t = blas::mulvec( alpha, blas::adjoint( *V_i ), x );

                blas::mulvec( value_t(1), *U_i, t, value_t(1), y );
            }// for
        }
        break;

        case Hpro::apply_conjugate :
            HLR_ERROR( "todo" );
            break;
        
        case Hpro::apply_transposed :
            HLR_ERROR( "todo" );
            break;
        
        case Hpro::apply_adjoint :
        {
            auto  U_i = op.U.cbegin();
            auto  V_i = op.V.cbegin();

            for ( ; ( U_i != op.U.cend() ) && ( V_i != op.V.cend() ); ++U_i, ++V_i )
            {
                if ( (*U_i).ncols() == 0 )
                    continue;
            
                const auto  t = blas::mulvec( alpha, blas::adjoint( *U_i ), x );

                blas::mulvec( value_t(1), *V_i, t, value_t(1), y );
            }// for
        }
        break;
    }// if
}

template < typename value_t >
void
prod ( const value_t                           alpha,
       const matop_t                           op_M,
       const lowranksum_operator< value_t > &  op,
       const blas::matrix< value_t > &         X,
       blas::matrix< value_t > &               Y )
{
    switch ( op_M )
    {
        case Hpro::apply_normal :
        {
            auto  U_i = op.U.cbegin();
            auto  V_i = op.V.cbegin();

            for ( ; ( U_i != op.U.cend() ) && ( V_i != op.V.cend() ); ++U_i, ++V_i )
            {
                if ( (*U_i).ncols() == 0 )
                    continue;
            
                const auto  T = blas::prod( alpha, blas::adjoint( *V_i ), X );

                blas::prod( value_t(1), *U_i, T, value_t(1), Y );
            }// for
        }
        break;

        case Hpro::apply_conjugate :
            HLR_ERROR( "todo" );
            break;
        
        case Hpro::apply_transposed :
            HLR_ERROR( "todo" );
            break;
        
        case Hpro::apply_adjoint :
        {
            auto  U_i = op.U.cbegin();
            auto  V_i = op.V.cbegin();

            for ( ; ( U_i != op.U.cend() ) && ( V_i != op.V.cend() ); ++U_i, ++V_i )
            {
                if ( (*U_i).ncols() == 0 )
                    continue;
            
                const auto  T = blas::prod( alpha, blas::adjoint( *U_i ), X );

                blas::prod( value_t(1), *V_i, T, value_t(1), Y );
            }// for
        }
        break;
    }// if
}

template < typename value_t >
lowranksum_operator< value_t >
operator_wrapper ( const std::list< blas::matrix< value_t > > &  U,
                   const std::list< blas::matrix< value_t > > &  V )
{
    return lowranksum_operator< value_t > ( U, V );
}

////////////////////////////////////////////////////////////////////////////////
//
// operator wrapper for sum of factorized lowrank matrices
//
////////////////////////////////////////////////////////////////////////////////

template < typename T_value >
struct lowranksumT_operator
{
    using  value_t = T_value;

    const std::list< blas::matrix< value_t > > &  U;
    const std::list< blas::matrix< value_t > > &  T;
    const std::list< blas::matrix< value_t > > &  V;

    lowranksumT_operator ( const std::list< blas::matrix< value_t > > &  aU,
                           const std::list< blas::matrix< value_t > > &  aT,
                           const std::list< blas::matrix< value_t > > &  aV )
            : U( aU )
            , T( aT )
            , V( aV )
    {
        HLR_ASSERT( U.size() == V.size() );
        HLR_ASSERT( U.size() == T.size() );
        HLR_ASSERT( ! U.empty() );
    }
    
    size_t  nrows () const { return U.front().nrows(); }
    size_t  ncols () const { return V.front().nrows(); }

    blas::vector< value_t >
    get_row ( const size_t  /* i */ ) const
    {
        HLR_ERROR( "todo" );
    }

    blas::vector< value_t >
    get_column ( const size_t  /* j */ ) const
    {
        HLR_ERROR( "todo" );
    }
};

template < typename value_t > size_t nrows ( const lowranksumT_operator< value_t > &  op ) { return op.nrows(); }
template < typename value_t > size_t ncols ( const lowranksumT_operator< value_t > &  op ) { return op.ncols(); }

template < typename value_t >
blas::vector< value_t >
get_row ( const lowranksumT_operator< value_t > &  op,
          const size_t                             i )
{
    return op.get_row( i );
}

template < typename value_t >
blas::vector< value_t >
get_column ( const lowranksumT_operator< value_t > &  op,
             const size_t                            j )
{
    return op.get_column( j );
}

template < typename value_t >
void
prod ( const value_t                            /* alpha */,
       const matop_t                            /* op_M */,
       const lowranksumT_operator< value_t > &  /* op */,
       const blas::vector< value_t > &          /* x */,
       const blas::vector< value_t > &          /* y */ )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
prod ( const value_t                            /* alpha */,
       const matop_t                            /* op_M */,
       const lowranksumT_operator< value_t > &  /* op */,
       const blas::matrix< value_t > &          /* X */,
       const blas::matrix< value_t > &          /* Y */ )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
lowranksumT_operator< value_t >
operator_wrapper ( const std::list< blas::matrix< value_t > > &  U,
                   const std::list< blas::matrix< value_t > > &  T,
                   const std::list< blas::matrix< value_t > > &  V )
{
    return lowranksumT_operator< value_t > ( U, T, V );
}

//////////////////////////////////////////////////////////////////////
//
// wrapper functions for Hpro::TLinearOperator
//
//////////////////////////////////////////////////////////////////////

template < typename value_t > size_t  nrows ( const Hpro::TLinearOperator< value_t > &  M ) { return M.range_dim(); }
template < typename value_t > size_t  ncols ( const Hpro::TLinearOperator< value_t > &  M ) { return M.domain_dim(); }

template < typename value_t >
void
prod ( const value_t                             alpha,
       const matop_t                             op_M,
       const Hpro::TLinearOperator< value_t > &  M,
       const blas::vector< value_t > &           x,
       blas::vector< value_t > &                 y )
{
    M.apply_add( alpha, x, y, op_M );
}

//////////////////////////////////////////////////////////////////////
//
// operator wrapper for TCoeffFn
//
//////////////////////////////////////////////////////////////////////

template < typename coeff_fn_t >
struct coefffn_operator
{
    using  value_t = typename coeff_fn_t::value_t;

    // block index set to be evaluated at
    Hpro::TBlockIndexSet  bis;

    // coefficient function
    const coeff_fn_t &    func;

    coefffn_operator ( const Hpro::TBlockIndexSet &  abis,
                       const coeff_fn_t &            afunc )
            : bis( abis )
            , func( afunc )
    {}
};

template < typename coeff_fn_t >
size_t
nrows ( const coefffn_operator< coeff_fn_t > &  op )
{
    return op.bis.row_is().size();
}

template < typename coeff_fn_t >
size_t
ncols ( const coefffn_operator< coeff_fn_t > &  op )
{
    return op.bis.col_is().size();
}

template < typename coeff_fn_t >
blas::vector< typename coeff_fn_t::value_t >
get_row ( const coefffn_operator< coeff_fn_t > &  op,
          const size_t                            i )
{
    blas::vector< typename coeff_fn_t::value_t >  v( ncols( op ) );
    const auto                                    ofs = i + op.bis.row_is().first();

    op.func.eval( Hpro::is( ofs, ofs ), op.bis.col_is(), v.data() );
                      
    return v;
}

template < typename coeff_fn_t >
blas::vector< typename coeff_fn_t::value_t >
get_column ( const coefffn_operator< coeff_fn_t > &  op,
             const size_t                            i )
{
    blas::vector< typename coeff_fn_t::value_t >  v( nrows( op ) );
    const auto                                    ofs = i + op.bis.col_is().first();

    op.func.eval( op.bis.row_is(), Hpro::is( ofs, ofs ), v.data() );
                      
    return v;
}

template < typename coeff_fn_t >
coefffn_operator< coeff_fn_t >
operator_wrapper ( const Hpro::TBlockIndexSet &  bis,
                   const coeff_fn_t &            func )
{
    return coefffn_operator< coeff_fn_t > ( bis, func );
}

}// namespace hlr

#endif // __HLR_ARITH_OPERATOR_WRAPPER_HH

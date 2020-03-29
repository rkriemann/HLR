#ifndef __HLR_ARITH_OPERATOR_WRAPPER_HH
#define __HLR_ARITH_OPERATOR_WRAPPER_HH
//
// Project     : HLR
// File        : arith/operator_wrapper.hh
// Description : wrapper functions for some standard operators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/cluster/TIndexSet.hh>

#include <hlr/arith/blas.hh>
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
    hpro::TBlockIndexSet  bis;

    // coefficient function
    const coeff_fn_t &    func;

    coefffn_operator ( const hpro::TBlockIndexSet &  abis,
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

    op.func.eval( hpro::is( ofs, ofs ), op.bis.col_is(), v.data() );
                      
    return v;
}

template < typename coeff_fn_t >
blas::vector< typename coeff_fn_t::value_t >
get_column ( const coefffn_operator< coeff_fn_t > &  op,
             const size_t                            i )
{
    blas::vector< typename coeff_fn_t::value_t >  v( nrows( op ) );
    const auto                                    ofs = i + op.bis.col_is().first();

    op.func.eval( op.bis.row_is(), hpro::is( ofs, ofs ), v.data() );
                      
    return v;
}

}// namespace hlr

#endif // __HLR_ARITH_OPERATOR_WRAPPER_HH
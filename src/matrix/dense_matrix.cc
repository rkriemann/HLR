//
// Project     : HLR
// Module      : dense_matrix
// Description : low-rank matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/matrix/dense_matrix.hh>

namespace hlr
{ 

namespace matrix
{

namespace
{

template < typename dest_value_t >
blas::vector< dest_value_t >
convert ( const hpro::TScalarVector &  v )
{
    if ( v.is_complex() )
    {
        if constexpr( std::is_same_v< hpro::complex, dest_value_t > )
            return blas::vec< hpro::complex >( v );
        if constexpr( hpro::is_complex_type_v< dest_value_t > )
            return blas::copy< dest_value_t, hpro::complex >( blas::vec< hpro::complex >( v ) );
        else
            HLR_ERROR( "real <- complex" );
    }// if
    else
    {
        if constexpr( std::is_same_v< hpro::real, dest_value_t > )
            return blas::vec< hpro::real >( v );
        else
            return blas::copy< dest_value_t >( blas::vec< hpro::real >( v ) );
    }// else
}

template < typename value_t >
void
vec_add ( const blas::vector< value_t > &  t,
          hpro::TScalarVector &            v )
{
    if ( v.is_complex() )
    {
        if constexpr( std::is_same_v< hpro::complex, value_t > )
            blas::add( value_t(1), t, blas::vec< hpro::complex >( v ) );
        else
        {
            auto  bv = blas::vec< hpro::complex >( v );
            
            for ( size_t  i = 0; i < t.length(); ++i )
                bv( i ) += t( i );
        }// else
    }// if
    else
    {
        if constexpr( std::is_same_v< value_t, hpro::real > )
        {
            blas::add( value_t(1), t, blas::vec< hpro::real >( v ) );
        }// if
        else if constexpr( ! hpro::is_complex_type_v< value_t > )
        {
            auto  bv = blas::vec< hpro::real >( v );
            
            for ( size_t  i = 0; i < t.length(); ++i )
                bv( i ) += t( i );
        }// if
        else
            HLR_ERROR( "real <- complex" );
    }// else
}

}// namespace anonymous

//
// matrix vector multiplication
//
void
dense_matrix::mul_vec ( const hpro::real       alpha,
                        const hpro::TVector *  vx,
                        const hpro::real       beta,
                        hpro::TVector *        vy,
                        const hpro::matop_t    op ) const
{
    HLR_ASSERT( vx->is() == this->col_is( op ) );
    HLR_ASSERT( vy->is() == this->row_is( op ) );
    HLR_ASSERT( is_scalar_all( vx, vy ) );

    const auto  sx = cptrcast( vx, hpro::TScalarVector );
    const auto  sy = ptrcast(  vy, hpro::TScalarVector );

    // y := β·y
    if ( beta != hpro::real(1) )
        vy->scale( beta );

    std::visit(
        [=] ( auto &&  M )
        {
            using  value_t  = typename std::decay_t< decltype(M) >::value_t;
            
            auto  x = convert< value_t >( *sx );
            auto  y = blas::vector< value_t >( sy->size() );

            blas::mulvec( value_t(alpha), blas::mat_view( op, M ), x, value_t(0), y );
            vec_add< value_t >( y, *sy );
        },
        _M );
    
}

void
dense_matrix::cmul_vec ( const hpro::complex    alpha,
                         const hpro::TVector *  vx,
                         const hpro::complex    beta,
                         hpro::TVector *        vy,
                         const hpro::matop_t    op ) const
{
    // HLR_ASSERT( vx->is_complex() == this->is_complex() );
    // HLR_ASSERT( vy->is_complex() == this->is_complex() );
    // HLR_ASSERT( vx->is() == this->col_is( op ) );
    // HLR_ASSERT( vy->is() == this->row_is( op ) );
    // HLR_ASSERT( is_scalar_all( vx, vy ) );

    // if constexpr( std::is_same_v< value_t, complex > )
    // {
    //     const auto  x = cptrcast( vx, hpro::TScalarVector );
    //     const auto  y = ptrcast(  vy, hpro::TScalarVector );
        
    //     // y := β·y
    //     if ( beta != complex(1) )
    //         blas::scale( value_t(beta), hpro::blas_vec< value_t >( y ) );
                     
    //     if ( op == hpro::apply_normal )
    //     {
    //         //
    //         // y = y + U·S·V^H x
    //         //
            
    //         // t := V^H x
    //         auto  t = blas::mulvec( blas::adjoint( col_basis() ), hpro::blas_vec< value_t >( x ) );

    //         // s := S t
    //         auto  s = blas::mulvec( _S, t );
        
    //         // r := U s
    //         auto  r = blas::mulvec( row_basis(), s );

    //         // y = y + r
    //         blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    //     }// if
    //     else if ( op == hpro::apply_transposed )
    //     {
    //         //
    //         // y = y + (U·S·V^H)^T x
    //         //   = y + conj(V)·S^T·U^T x
    //         //
        
    //         // t := U^T x
    //         auto  t = blas::mulvec( blas::transposed( row_basis() ), hpro::blas_vec< value_t >( x ) );
        
    //         // s := S^T t
    //         auto  s = blas::mulvec( blas::transposed(_S), t );

    //         // r := conj(V) s
    //         blas::conj( s );
            
    //         auto  r = blas::mulvec( col_basis(), s );

    //         blas::conj( r );

    //         // y = y + r
    //         blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    //     }// if
    //     else if ( op == hpro::apply_adjoint )
    //     {
    //         //
    //         // y = y + (U·S·V^H)^H x
    //         //   = y + V·S^H·U^H x
    //         //
        
    //         // t := U^H x
    //         auto  t = blas::mulvec( blas::adjoint( row_basis() ), hpro::blas_vec< value_t >( x ) );

    //         // s := S t
    //         auto  s = blas::mulvec( blas::adjoint(_S), t );
        
    //         // r := V s
    //         auto  r = blas::mulvec( col_basis(), s );

    //         // y = y + r
    //         blas::add( value_t(alpha), r, hpro::blas_vec< value_t >( y ) );
    //     }// if
    // }// if
    // else
    //     HLR_ERROR( "todo" );
}

//
// return copy of matrix
//
std::unique_ptr< hpro::TMatrix >
dense_matrix::copy () const
{
    auto  M = std::make_unique< dense_matrix >( _row_is, _col_is );

    M->copy_struct_from( this );
    
    std::visit( [&M] ( auto &&  D ) { M->set_matrix( D ); }, _M );
    
    return M;
}

//
// return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
//
std::unique_ptr< hpro::TMatrix >
dense_matrix::copy ( const hpro::TTruncAcc &,
                     const bool       ) const
{
    return copy();
}

//
// return structural copy of matrix
//
std::unique_ptr< hpro::TMatrix >
dense_matrix::copy_struct () const
{
    return std::make_unique< dense_matrix >( _row_is, _col_is );
}

//
// copy matrix data to \a A
//
void
dense_matrix::copy_to ( hpro::TMatrix *  A ) const
{
    hpro::TMatrix::copy_to( A );
    
    HLR_ASSERT( IS_TYPE( A, dense_matrix ) );

    auto  D = ptrcast( A, dense_matrix );

    D->_row_is = _row_is;
    D->_col_is = _col_is;
    D->_vtype  = _vtype;
    
    std::visit( [&D] ( auto &&  M ) { D->set_matrix( M ); }, _M );
}

//
// copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
//
void
dense_matrix::copy_to ( hpro::TMatrix *          A,
                        const hpro::TTruncAcc &,
                        const bool          ) const
{
    return copy_to( A );
}

//
// return size in bytes used by this object
//
size_t
dense_matrix::byte_size () const
{
    size_t  size = hpro::TMatrix::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is) + sizeof(_vtype);

    std::visit( [&size] ( auto &&  M ) { size += M.byte_size(); }, _M );

    return size;
}

}} // namespace hlr::matrix

//
// Project     : HLR
// Module      : dense_matrix
// Description : low-rank matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/matrix/dense_matrix.hh>

namespace hlr
{ 

namespace matrix
{

namespace
{

template < typename dest_value_t,
           typename src_value_t >
blas::vector< dest_value_t >
convert ( const Hpro::TScalarVector< value_t > &  v )
{
    if ( v.is_complex() )
    {
        if constexpr( std::is_same_v< Hpro::complex, dest_value_t > )
            return blas::vec< Hpro::complex >( v );
        if constexpr( Hpro::is_complex_type_v< dest_value_t > )
            return blas::copy< dest_value_t, Hpro::complex >( blas::vec< Hpro::complex >( v ) );
        else
            HLR_ERROR( "real <- complex" );
    }// if
    else
    {
        if constexpr( std::is_same_v< Hpro::real, dest_value_t > )
            return blas::vec< Hpro::real >( v );
        else
            return blas::copy< dest_value_t >( blas::vec< Hpro::real >( v ) );
    }// else
}

template < typename value_t >
void
vec_add ( const blas::vector< value_t > &  t,
          Hpro::TScalarVector &            v )
{
    if ( v.is_complex() )
    {
        if constexpr( std::is_same_v< Hpro::complex, value_t > )
            blas::add( value_t(1), t, blas::vec< Hpro::complex >( v ) );
        else
        {
            auto  bv = blas::vec< Hpro::complex >( v );
            
            for ( size_t  i = 0; i < t.length(); ++i )
                bv( i ) += t( i );
        }// else
    }// if
    else
    {
        if constexpr( std::is_same_v< value_t, Hpro::real > )
        {
            blas::add( value_t(1), t, blas::vec< Hpro::real >( v ) );
        }// if
        else if constexpr( ! Hpro::is_complex_type_v< value_t > )
        {
            auto  bv = blas::vec< Hpro::real >( v );
            
            for ( size_t  i = 0; i < t.length(); ++i )
                bv( i ) += t( i );
        }// if
        else
            HLR_ERROR( "real <- complex" );
    }// else
}

//
// return uncompressed matrix
//
#if defined(HAS_SZ)

template < typename value_t >
blas::matrix< value_t >
sz_uncompress ( const sz::carray_view &  v,
                const size_t             nrows,
                const size_t             ncols )
{
    auto  M = blas::matrix< value_t >( nrows, ncols );
    
    sz::uncompress< value_t >( v, M.data(), nrows, ncols );

    return M;
}

#elif defined(HAS_ZFP)

template < typename value_t >
blas::matrix< value_t >
zfp_uncompress ( const zfp::carray &  v,
                 const size_t         nrows,
                 const size_t         ncols )
{
    auto  M = blas::matrix< value_t >( nrows, ncols );
    
    zfp::uncompress< value_t >( v, M.data(), nrows, ncols );

    return M;
}
#endif

}// namespace anonymous

//
// matrix vector multiplication
//
void
dense_matrix::mul_vec ( const Hpro::real       alpha,
                        const Hpro::TVector *  vx,
                        const Hpro::real       beta,
                        Hpro::TVector *        vy,
                        const Hpro::matop_t    op ) const
{
    HLR_ASSERT( vx->is() == this->col_is( op ) );
    HLR_ASSERT( vy->is() == this->row_is( op ) );
    HLR_ASSERT( is_scalar_all( vx, vy ) );

    const auto  sx = cptrcast( vx, Hpro::TScalarVector );
    const auto  sy = ptrcast(  vy, Hpro::TScalarVector );

    // y := β·y
    if ( beta != Hpro::real(1) )
        vy->scale( beta );

    std::visit(
        [=] ( auto &&  M )
        {
            using  value_t = typename std::decay_t< decltype(M) >::value_t;
            
            auto  x = convert< value_t >( *sx );
            auto  y = blas::vector< value_t >( sy->size() );

            if ( is_compressed() )
            {
                #if defined(HAS_SZ)
                
                auto  cM = sz_uncompress< value_t >( _zdata, nrows(), ncols() );

                blas::mulvec( value_t(alpha), blas::mat_view( op, cM ), x, value_t(0), y );

                #elif defined(HAS_ZFP)

                auto  cM = zfp_uncompress< value_t >( _zdata, nrows(), ncols() );

                blas::mulvec( value_t(alpha), blas::mat_view( op, cM ), x, value_t(0), y );
                
                #endif
            }// if
            else
            {
                blas::mulvec( value_t(alpha), blas::mat_view( op, M ), x, value_t(0), y );
            }// else

            vec_add< value_t >( y, *sy );
        },
        _M );
}

void
dense_matrix::cmul_vec ( const Hpro::complex    alpha,
                         const Hpro::TVector *  vx,
                         const Hpro::complex    beta,
                         Hpro::TVector *        vy,
                         const Hpro::matop_t    op ) const
{
    // HLR_ASSERT( vx->is_complex() == this->is_complex() );
    // HLR_ASSERT( vy->is_complex() == this->is_complex() );
    // HLR_ASSERT( vx->is() == this->col_is( op ) );
    // HLR_ASSERT( vy->is() == this->row_is( op ) );
    // HLR_ASSERT( is_scalar_all( vx, vy ) );

    // if constexpr( std::is_same_v< value_t, complex > )
    // {
    //     const auto  x = cptrcast( vx, Hpro::TScalarVector );
    //     const auto  y = ptrcast(  vy, Hpro::TScalarVector );
        
    //     // y := β·y
    //     if ( beta != complex(1) )
    //         blas::scale( value_t(beta), Hpro::blas_vec< value_t >( y ) );
                     
    //     if ( op == Hpro::apply_normal )
    //     {
    //         //
    //         // y = y + U·S·V^H x
    //         //
            
    //         // t := V^H x
    //         auto  t = blas::mulvec( blas::adjoint( col_basis() ), Hpro::blas_vec< value_t >( x ) );

    //         // s := S t
    //         auto  s = blas::mulvec( _S, t );
        
    //         // r := U s
    //         auto  r = blas::mulvec( row_basis(), s );

    //         // y = y + r
    //         blas::add( value_t(alpha), r, Hpro::blas_vec< value_t >( y ) );
    //     }// if
    //     else if ( op == Hpro::apply_transposed )
    //     {
    //         //
    //         // y = y + (U·S·V^H)^T x
    //         //   = y + conj(V)·S^T·U^T x
    //         //
        
    //         // t := U^T x
    //         auto  t = blas::mulvec( blas::transposed( row_basis() ), Hpro::blas_vec< value_t >( x ) );
        
    //         // s := S^T t
    //         auto  s = blas::mulvec( blas::transposed(_S), t );

    //         // r := conj(V) s
    //         blas::conj( s );
            
    //         auto  r = blas::mulvec( col_basis(), s );

    //         blas::conj( r );

    //         // y = y + r
    //         blas::add( value_t(alpha), r, Hpro::blas_vec< value_t >( y ) );
    //     }// if
    //     else if ( op == Hpro::apply_adjoint )
    //     {
    //         //
    //         // y = y + (U·S·V^H)^H x
    //         //   = y + V·S^H·U^H x
    //         //
        
    //         // t := U^H x
    //         auto  t = blas::mulvec( blas::adjoint( row_basis() ), Hpro::blas_vec< value_t >( x ) );

    //         // s := S t
    //         auto  s = blas::mulvec( blas::adjoint(_S), t );
        
    //         // r := V s
    //         auto  r = blas::mulvec( col_basis(), s );

    //         // y = y + r
    //         blas::add( value_t(alpha), r, Hpro::blas_vec< value_t >( y ) );
    //     }// if
    // }// if
    // else
    //     HLR_ERROR( "todo" );
}

//
// return copy of matrix
//
std::unique_ptr< Hpro::TMatrix >
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
std::unique_ptr< Hpro::TMatrix >
dense_matrix::copy ( const Hpro::TTruncAcc &,
                     const bool       ) const
{
    return copy();
}

//
// return structural copy of matrix
//
std::unique_ptr< Hpro::TMatrix >
dense_matrix::copy_struct () const
{
    return std::make_unique< dense_matrix >( _row_is, _col_is );
}

//
// copy matrix data to \a A
//
void
dense_matrix::copy_to ( Hpro::TMatrix *  A ) const
{
    Hpro::TMatrix::copy_to( A );
    
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
dense_matrix::copy_to ( Hpro::TMatrix *          A,
                        const Hpro::TTruncAcc &,
                        const bool          ) const
{
    return copy_to( A );
}

//
// compress internal data
//
void
dense_matrix::compress ( const zconfig_t &  config )
{
    #if defined(HAS_SZ)

    if ( is_compressed() )
        return;
    
    std::visit(
        [this,&config] ( auto &&  M )
        {
            using  value_t = typename std::decay_t< decltype(M) >::value_t;

            const size_t  mem_dense = sizeof(value_t) * M.nrows() * M.ncols();
            auto          v         = sz::compress( config, M.data(), M.nrows(), M.ncols() );

            if ( v.size() < mem_dense )
            {
                _zdata = std::move( v );
                M      = std::move( blas::matrix< value_t >( 0, 0 ) );
            }// if
        },
        _M
    );

    #elif defined(HAS_ZFP)
    
    if ( is_compressed() )
        return;
    
    std::visit(
        [this,&config] ( auto &&  M )
        {
            using  value_t = typename std::decay_t< decltype(M) >::value_t;

            const size_t  mem_dense = sizeof(value_t) * M.nrows() * M.ncols();
            auto          v         = zfp::compress< value_t >( config, M.data(), M.nrows(), M.ncols() );

            if ( v.size() < mem_dense )
            {
                _zdata = std::move( v );
                M      = std::move( blas::matrix< value_t >( 0, 0 ) );
            }// if
        },
        _M
    );

    #endif
    
}

//
// uncompress internal data
//
void
dense_matrix::uncompress ()
{
    #if defined(HAS_SZ)
    
    if ( ! is_compressed() )
        return;

    std::visit( 
        [this] ( auto &&  M )
        {
            using  value_t = typename std::decay_t< decltype(M) >::value_t;

            M = std::move( sz_uncompress< value_t >( _zdata, nrows(), ncols() ) );
        },
        _M );

    #elif defined(HAS_ZFP)
    
    if ( ! is_compressed() )
        return;

    std::visit( 
        [this] ( auto &&  M )
        {
            using  value_t = typename std::decay_t< decltype(M) >::value_t;

            M = std::move( zfp_uncompress< value_t >( _zdata, nrows(), ncols() ) );
        },
        _M );
    
    #endif
}

//
// return size in bytes used by this object
//
size_t
dense_matrix::byte_size () const
{
    size_t  size = Hpro::TMatrix::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is) + sizeof(_vtype);

    std::visit( [&size] ( auto &&  M ) { size += M.byte_size(); }, _M );

    #if defined(HAS_SZ)

    size += sizeof(_zdata) + _zdata.size();
    
    #elif defined(HAS_ZFP)

    size += sizeof(_zdata) + _zdata.size();
    
    #endif
        
    return size;
}

}} // namespace hlr::matrix

//
// Project     : HLR
// Module      : lrmatrix
// Description : low-rank matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/matrix/lrmatrix.hh>

namespace hlr
{ 

namespace matrix
{

namespace
{

template < typename T_alpha,
           typename T_value >
void
lr_mul_vec ( const T_alpha                    alpha,
             const blas::matrix< T_value > &  U,
             const blas::matrix< T_value > &  V,
             const matop_t                    op,
             const blas::vector< T_value > &  x,
             blas::vector< T_value > &        y )
{
    using  value_t = T_value;
    
    if ( op == hpro::apply_normal )
    {
        //
        // y = y + U·V^H x
        //
        
        // t := V^H x
        auto  t = blas::mulvec( blas::adjoint( V ), x );

        // t := α·t
        blas::scale( value_t(alpha), t );
        
        // y := y + U t
        blas::mulvec( U, t, y );
    }// if
    else if ( op == hpro::apply_transposed )
    {
        //
        // y = y + (U·V^H)^T x
        //   = y + conj(V)·U^T x
        //
        
        // t := U^T x
        auto  t = blas::mulvec( blas::transposed( U ), x );

        // t := α·t
        blas::scale( value_t(alpha), t );
        
        // r := conj(V) t = conj( V · conj(t) )
        blas::conj( t );
            
        auto  r = blas::mulvec( V, t );

        blas::conj( r );

        // y = y + r
        blas::add( value_t(1), r, y );
    }// if
    else if ( op == hpro::apply_adjoint )
    {
        //
        // y = y + (U·V^H)^H x
        //   = y + V·U^H x
        //
        
        // t := U^H x
        auto  t = blas::mulvec( blas::adjoint( U ), x );

        // t := α·t
        blas::scale( value_t(alpha), t );
        
        // y := t + V t
        blas::mulvec( V, t, y );
    }// if
}

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

//
// return uncompressed matrix
//
#if defined(HAS_ZFP)
template < typename mat_value_t,
           typename zfp_value_t >
blas::matrix< mat_value_t >
zfp_uncompress ( zfp::const_array2< zfp_value_t > &  z,
                 const size_t                        nrows,
                 const size_t                        ncols )
{
    auto  M = blas::matrix< mat_value_t >( nrows, ncols );
    
    z.get( (zfp_value_t*) M.data() );

    return M;
}
#endif

}// namespace anonymous

//
// matrix vector multiplication
//
void
lrmatrix::mul_vec ( const hpro::real       alpha,
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

            // if ((   sy->is_complex() && std::is_same_v< value_t, hpro::complex > ) ||
            //     ( ! sy->is_complex() && std::is_same_v< value_t, hpro::real > ))
            // {
            //     lr_mul_vec< value_t >( alpha, *this, op, x, blas::vec< value_t >( sy ) );
            // }// if
            // else
            {
                auto  y = blas::vector< value_t >( sy->size() );

                #if defined(HAS_ZFP)
                if ( is_compressed() )
                {
                    auto  cUV = std::visit( 
                        [this,&M] ( auto &&  zM )
                        {
                            using  value_t     = typename std::decay_t< decltype(M) >::value_t;
                            using  zfp_value_t = typename std::decay_t< decltype(zM->U) >::value_type;
                        
                            auto cU  = zfp_uncompress< value_t, zfp_value_t >( zM->U, nrows(), rank() );
                            auto cV  = zfp_uncompress< value_t, zfp_value_t >( zM->V, ncols(), rank() );
                            auto cUV = lrfactors< value_t >{ std::move(cU), std::move(cV) };

                            return cUV;
                        },
                        _zdata );

                    lr_mul_vec< value_t >( alpha, cUV.U, cUV.V, op, x, y );
                }// if
                else
                #endif
                {
                    lr_mul_vec< value_t >( alpha, M.U, M.V, op, x, y );
                }// else
                
                vec_add< value_t >( y, *sy );
            }// else
        },
        _UV );
    
}

void
lrmatrix::cmul_vec ( const hpro::complex    alpha,
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
// truncate matrix to accuracy <acc>
//
void
lrmatrix::truncate ( const hpro::TTruncAcc & )
{
}

//
// return copy of matrix
//
std::unique_ptr< hpro::TMatrix >
lrmatrix::copy () const
{
    auto  M = std::make_unique< lrmatrix >( _row_is, _col_is );

    M->copy_struct_from( this );
    
    std::visit( [&M] ( auto &&  UV ) { M->set_lrmat( UV.U, UV.V ); }, _UV );
    
    return M;
}

//
// return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
//
std::unique_ptr< hpro::TMatrix >
lrmatrix::copy ( const hpro::TTruncAcc &,
                 const bool       ) const
{
    return copy();
}

//
// return structural copy of matrix
//
std::unique_ptr< hpro::TMatrix >
lrmatrix::copy_struct () const
{
    return std::make_unique< lrmatrix >( _row_is, _col_is );
}

//
// copy matrix data to \a A
//
void
lrmatrix::copy_to ( hpro::TMatrix *  A ) const
{
    hpro::TMatrix::copy_to( A );
    
    HLR_ASSERT( IS_TYPE( A, lrmatrix ) );

    auto  R = ptrcast( A, lrmatrix );

    R->_row_is = _row_is;
    R->_col_is = _col_is;
    
    std::visit( [&R] ( auto &&  UV ) { R->set_lrmat( UV.U, UV.V ); }, _UV );
}

//
// copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
//
void
lrmatrix::copy_to ( hpro::TMatrix *          A,
                    const hpro::TTruncAcc &,
                    const bool          ) const
{
    return copy_to( A );
}

//
// compress internal data
//
void
lrmatrix::compress ( const zfp_config &  config )
{
    #if defined(HAS_ZFP)
    if ( is_compressed() )
        return;
    
    std::visit(
        [this,&config] ( auto &&  UV )
        {
            using  value_t    = typename std::decay_t< decltype(UV) >::value_t;
            using  real_t     = typename hpro::real_type_t< value_t >;
            using  real_ptr_t = real_t *;

            const auto    rank_UV   = UV.U.ncols();
            // auto          config    = zfp_config_rate( rate, false );
            uint          factor    = sizeof(value_t) / sizeof(real_t);
            const size_t  mem_dense = sizeof(value_t) * rank_UV * ( UV.U.nrows() + UV.V.nrows() );

            if constexpr( std::is_same_v< value_t, real_t > )
            {
                auto  zM = std::make_unique< compressed_factors< value_t > >();

                zM->U.set_config( config );
                zM->V.set_config( config );
                
                zM->U.resize( UV.U.nrows(), UV.U.ncols() );
                zM->V.resize( UV.V.nrows(), UV.V.ncols() );
                
                zM->U.set( UV.U.data() );
                zM->V.set( UV.V.data() );

                const auto  mem_zfp = zM->U.compressed_size() + zM->V.compressed_size();

                if ( mem_zfp < mem_dense )
                {
                    _zdata = std::move( zM );
                    UV.U   = std::move( blas::matrix< value_t >( 0, rank_UV ) ); // remember rank !!!
                    UV.V   = std::move( blas::matrix< value_t >( 0, rank_UV ) );
                }// if
            }// if
            else
            {
                auto  zM = std::make_unique< compressed_factors< real_t > >();

                zM->U.set_config( config );
                zM->V.set_config( config );
                
                zM->U.resize( factor * UV.U.nrows(), UV.U.ncols() );
                zM->V.resize( factor * UV.V.nrows(), UV.V.ncols() );
                
                zM->U.set( real_ptr_t(UV.U.data()) );
                zM->V.set( real_ptr_t(UV.V.data()) );

                const auto  mem_zfp = zM->U.compressed_size() + zM->V.compressed_size();

                if ( mem_zfp < mem_dense )
                {
                    _zdata = std::move( zM );
                    UV.U   = std::move( blas::matrix< value_t >( 0, 0 ) );
                    UV.V   = std::move( blas::matrix< value_t >( 0, 0 ) );
                }// if
            }// else
        },
        _UV
    );
    #endif
}

//
// uncompress internal data
//
void
lrmatrix::uncompress ()
{
}

//
// return size in bytes used by this object
//
size_t
lrmatrix::byte_size () const
{
    size_t  size = hpro::TMatrix::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is) + sizeof(_vtype);

    std::visit( [&size] ( auto &&  M ) { size += M.U.byte_size() + M.V.byte_size(); }, _UV );

    size += sizeof(_zdata);

    std::visit( [&size] ( auto &&  zM )
    {
        if ( ! is_null(zM) )
        {
            size += sizeof(zM->U) + zM->U.compressed_size();
            size += sizeof(zM->V) + zM->V.compressed_size();
        }// if
    }, _zdata );
    
    return size;
}

}} // namespace hlr::matrix

#ifndef __HLR_BLAS_DETAIL_HOSVD_HH
#define __HLR_BLAS_DETAIL_HOSVD_HH

#include <hlr/utils/io.hh> // DEBUG

namespace hlr { namespace blas {

//
// standard HOSVD
//

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
hosvd ( const tensor3< value_t > &  X,
        const accuracy &            acc,
        const approx_t &            apx )
{
    auto  X0 = X.unfold( 0 );
    auto  U0 = apx.column_basis( X0, acc );

    auto  X1 = X.unfold( 1 );
    auto  U1 = apx.column_basis( X1, acc );

    auto  X2 = X.unfold( 2 );
    auto  U2 = apx.column_basis( X2, acc );

    auto  Y0 = tensor_product( X,  adjoint( U0 ), 0 );
    auto  Y1 = tensor_product( Y0, adjoint( U1 ), 1 );
    auto  G  = tensor_product( Y1, adjoint( U2 ), 2 );

    return { std::move(G), std::move(U0), std::move(U1), std::move(U2) };
}

template < typename  value_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
hosvd ( const tensor3< value_t > &  X,
        const accuracy &            acc )
{
    const auto  apx = approx::SVD< value_t >();

    return hosvd( X, acc, apx );
}

//
// sequentially truncated HOSVD
//
template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
sthosvd ( const tensor3< value_t > &  X,
          const accuracy &            acc,
          const approx_t &            apx )
{
    auto  Y  = copy( X );
    auto  U0 = blas::matrix< value_t >();
    auto  U1 = blas::matrix< value_t >();
    auto  U2 = blas::matrix< value_t >();
    
    for ( uint  d = 0; d < 3; ++d )
    {
        auto  Yd = Y.unfold( d );
        auto  Ud = apx.column_basis( Yd, acc );
        auto  T  = tensor_product( Y, adjoint( Ud ), d );

        Y = std::move( T );

        switch ( d )
        {
            case 0 : U0 = std::move( Ud ); break;
            case 1 : U1 = std::move( Ud ); break;
            case 2 : U2 = std::move( Ud ); break;
        }// switch
    }// for

    return { std::move(Y), std::move(U0), std::move(U1), std::move(U2) };
}

template < typename  value_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
sthosvd ( const tensor3< value_t > &  X,
          const accuracy &            acc )
{
    const auto  apx = approx::SVD< value_t >();

    return sthosvd( X, acc, apx );
}

//
// greedy HOSVD
//
template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
greedy_hosvd ( const tensor3< value_t > &  X,
               const accuracy &            acc,
               const approx_t &            apx )
{
    //
    // compute full column bases for unfolded matrices
    // for all dimensions
    //
    
    auto  X0         = X.unfold( 0 );
    auto  [ U0, S0 ] = apx.column_basis( X0 );

    auto  X1         = X.unfold( 1 );
    auto  [ U1, S1 ] = apx.column_basis( X1 );

    auto  X2         = X.unfold( 2 );
    auto  [ U2, S2 ] = apx.column_basis( X2 );

    // for index-based access
    matrix< value_t >  U[3] = { U0, U1, U2 };
    vector< value_t >  S[3] = { S0, S1, S2 };

    //
    // iterate until error is met increasing rank of
    // dimension with highest error contribution, i.e.,
    // largest _next_ singular value
    //
    // error = √( Σ_d Σ_i>k_i σ²_d,i )
    //

    const auto  tol      = acc.abs_eps() * acc.abs_eps();
    value_t     error[3] = { 0, 0, 0 };
    size_t      k[3]     = { 1, 1, 1 }; // start with at least one rank per dimension

    // initial error
    for ( uint  d = 0; d < 3; ++d )
        for ( uint  i = k[d]; i < S[d].length(); ++i )
            error[d] += S[d](i) * S[d](i);

    // iteration
    while ( error[0] + error[1] + error[2] > tol )
    {
        int      max_dim = -1; // to signal error
        value_t  max_sig = 0;

        // look for maximal σ in all dimensions
        for ( uint  d = 0; d < 3; ++d )
        {
            // skip fully exhausted dimensions
            if ( k[d] == S[d].length() )
                continue;
            
            if ( S[d](k[d]) > max_sig )
            {
                max_sig = S[d](k[d]);
                max_dim = d;
            }// if
        }// for

        if ( max_dim < 0 )
        {
            // no unused singular values left; error should be zero
            break;
        }// if

        error[ max_dim ] -= max_sig * max_sig;
        k[ max_dim ]     += 1;
        
        // std::cout << "  max_dim " << max_dim << ", error = " << std::sqrt( error[0] + error[1] + error[2] ) << std::flush;
    }// while

    auto  U0k = matrix< value_t >( U0, range::all, range( 0, k[0]-1 ) );
    auto  U1k = matrix< value_t >( U1, range::all, range( 0, k[1]-1 ) );
    auto  U2k = matrix< value_t >( U2, range::all, range( 0, k[2]-1 ) );

    auto  W0  = blas::copy( U0k );
    auto  W1  = blas::copy( U1k );
    auto  W2  = blas::copy( U2k );
    
    auto  Y0 = tensor_product( X,  adjoint( W0 ), 0 );
    auto  Y1 = tensor_product( Y0, adjoint( W1 ), 1 );
    auto  G  = tensor_product( Y1, adjoint( W2 ), 2 );

    io::matlab::write( W0, "W0" );
    io::matlab::write( W1, "W1" );
    io::matlab::write( W2, "W2" );

    io::matlab::write( S0, "S0" );
    io::matlab::write( S1, "S1" );
    io::matlab::write( S2, "S2" );

    auto  G0 = G.unfold( 0 );
    auto  G1 = G.unfold( 1 );
    auto  G2 = G.unfold( 2 );

    io::matlab::write( G0, "G0" );
    io::matlab::write( G1, "G1" );
    io::matlab::write( G2, "G2" );
    
    // // print compressed memory
    // if ( false )
    // {
    //     size_t  mem  = 0;
    //     size_t  zmem = 0;

    //     {
    //         auto    zconf = compress::afloat::get_config( zacc.rel_eps() );
    //         auto    Zc    = compress::afloat::compress( zconf, G.data(), G.size(0), G.size(1), G.size(2) );
    //         size_t  memc  = sizeof(value_t) * G.size(0) * G.size(1) * G.size(2);
    //         auto    zmemc = compress::afloat::byte_size( Zc );

    //         mem  += memc;
    //         zmem += zmemc;
    //     }
        
    //     {
    //         auto  S0k   = vector< value_t >( S0, range( 0, k[0]-1 ) );
    //         auto  norm0 = std::accumulate( S0k.data(), S0k.data() + k[0], value_t(0), std::plus< value_t >() );
    //         auto  tol0  = norm0 * zacc.rel_eps();

    //         std::for_each( S0k.data(), S0k.data() + k[0], [tol0] ( auto & f ) { f *= tol0; } );
        
    //         auto  Z0    = compress::afloat::compress_lr( W0, S0k );
    //         auto  mem0  = sizeof(value_t) * W0.nrows() * W0.ncols();
    //         auto  zmem0 = compress::afloat::byte_size( Z0 );

    //         mem  += mem0;
    //         zmem += zmem0;
    //     }
        
    //     {
    //         auto  S1k   = vector< value_t >( S1, range( 0, k[1]-1 ) );
    //         auto  norm1 = std::accumulate( S1k.data(), S1k.data() + k[1], value_t(0), std::plus< value_t >() );
    //         auto  tol1  = norm1 * zacc.rel_eps();

    //         std::for_each( S1k.data(), S1k.data() + k[1], [tol1] ( auto & f ) { f *= tol1; } );
        
    //         auto  Z1    = compress::afloat::compress_lr( W1, S1k );
    //         auto  mem1  = sizeof(value_t) * W1.nrows() * W1.ncols();
    //         auto  zmem1 = compress::afloat::byte_size( Z1 );

    //         mem  += mem1;
    //         zmem += zmem1;
    //     }
        
    //     {
    //         auto  S2k   = vector< value_t >( S2, range( 0, k[2]-1 ) );
    //         auto  norm2 = std::accumulate( S2k.data(), S2k.data() + k[2], value_t(0), std::plus< value_t >() );
    //         auto  tol2  = norm2 * zacc.rel_eps();

    //         std::for_each( S2k.data(), S2k.data() + k[2], [tol2] ( auto & f ) { f *= tol2; } );
        
    //         auto  Z2    = compress::afloat::compress_lr( W2, S2k );
    //         auto  mem2  = sizeof(value_t) * W2.nrows() * W2.ncols();
    //         auto  zmem2 = compress::afloat::byte_size( Z2 );

    //         mem  += mem2;
    //         zmem += zmem2;
    //     }

    //     std::cout << mem << " / " << zmem << std::endl;
    // }
    
    return { std::move(G), std::move(W0), std::move(W1), std::move(W2) };
}

//
// recompress given tucker tensor
//
template < typename                    value_t,
           approx::approximation_type  approx_t,
           typename                    hosvd_func_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
recompress ( tensor3< value_t > &  G,
             matrix< value_t > &   X0,
             matrix< value_t > &   X1,
             matrix< value_t > &   X2,
             const accuracy &      acc,
             const approx_t &      apx,
             hosvd_func_t &&       func )
{
    auto  [ G2, Y0, Y1, Y2 ] = func( G, acc, apx );

    auto  W0 = blas::prod( X0, Y0 );
    auto  W1 = blas::prod( X1, Y1 );
    auto  W2 = blas::prod( X2, Y2 );

    return { std::move(G2), std::move(W0), std::move(W1), std::move(W2) };
}
    
//
// error of Tucker decomposition D - G ×₀ X₀ ×₁ X₁ ×₂ X₂ 
//
template < typename value_t >
Hpro::real_type_t< value_t >
tucker_error ( const tensor3< value_t > &  D,
               const tensor3< value_t > &  G,
               const matrix< value_t > &   X0,
               const matrix< value_t > &   X1,
               const matrix< value_t > &   X2 )
{
    auto  T0 = tensor_product( G,  X0, 0 );
    auto  T1 = tensor_product( T0, X1, 1 );
    auto  Y  = tensor_product( T1, X2, 2 );
        
    add( -1, D, Y );

    return norm_F( Y );
}

}}// namespace hlr::blas

#endif // __HLR_BLAS_DETAIL_HOSVD_HH

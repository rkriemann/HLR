#ifndef __HLR_BLAS_DETAIL_HOSVD_HH
#define __HLR_BLAS_DETAIL_HOSVD_HH

#include <hlr/utils/io.hh> // DEBUG

namespace hlr { namespace blas {

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
// recompress given tucker tensor
//
template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
recompress ( tensor3< value_t > &  G,
             matrix< value_t > &   X0,
             matrix< value_t > &   X1,
             matrix< value_t > &   X2,
             const accuracy &      acc,
             const approx_t &      apx )
{
    auto  [ G2, Y0, Y1, Y2 ] = hosvd( G, acc, apx );

    auto  W0 = blas::prod( X0, Y0 );
    auto  W1 = blas::prod( X1, Y1 );
    auto  W2 = blas::prod( X2, Y2 );

    return { std::move(G2), std::move(W0), std::move(W1), std::move(W2) };
}

//
// return difference D - G ×₀ X₀ ×₁ X₁ ×₂ X₂ 
//
template < typename value_t >
tensor3< value_t >
tucker_diff ( const tensor3< value_t > &  D,
              const tensor3< value_t > &  G,
              const matrix< value_t > &   X0,
              const matrix< value_t > &   X1,
              const matrix< value_t > &   X2 )
{
    auto  T0 = tensor_product( G,  X0, 0 );
    auto  T1 = tensor_product( T0, X1, 1 );
    auto  Y  = tensor_product( T1, X2, 2 );
        
    add( -1, D, Y );

    return Y;
}

template < typename value_t >
tensor4< value_t >
tucker_diff ( const tensor4< value_t > &  D,
              const tensor4< value_t > &  G,
              const matrix< value_t > &   X0,
              const matrix< value_t > &   X1,
              const matrix< value_t > &   X2,
              const matrix< value_t > &   X3 )
{
    auto  T0 = tensor_product( G,  X0, 0 );
    auto  T1 = tensor_product( T0, X1, 1 );
    auto  T2 = tensor_product( T1, X2, 2 );
    auto  Y  = tensor_product( T2, X3, 3 );
        
    add( -1, D, Y );

    return Y;
}

//
// error of Tucker decomposition D - G ×₀ X₀ ×₁ X₁ ×₂ X₂ 
//
template < typename value_t >
real_type_t< value_t >
tucker_error ( const tensor3< value_t > &  D,
               const tensor3< value_t > &  G,
               const matrix< value_t > &   X0,
               const matrix< value_t > &   X1,
               const matrix< value_t > &   X2 )
{
    auto  Y = tucker_diff( D, G, X0, X1, X2 );

    return norm_F( Y );
}

template < typename value_t >
real_type_t< value_t >
tucker_error ( const tensor4< value_t > &  D,
               const tensor4< value_t > &  G,
               const matrix< value_t > &   X0,
               const matrix< value_t > &   X1,
               const matrix< value_t > &   X2,
               const matrix< value_t > &   X3 )
{
    auto  Y = tucker_diff( D, G, X0, X1, X2, X3 );

    return norm_F( Y );
}

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

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor4< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t >,
            matrix< value_t > >
hosvd ( const tensor4< value_t > &  X,
        const accuracy &            acc,
        const approx_t &            apx )
{
    auto  X0 = X.unfold( 0 );
    auto  U0 = apx.column_basis( X0, acc );

    auto  X1 = X.unfold( 1 );
    auto  U1 = apx.column_basis( X1, acc );

    auto  X2 = X.unfold( 2 );
    auto  U2 = apx.column_basis( X2, acc );

    auto  X3 = X.unfold( 3 );
    auto  U3 = apx.column_basis( X3, acc );

    auto  Y0 = tensor_product( X,  adjoint( U0 ), 0 );
    auto  Y1 = tensor_product( Y0, adjoint( U1 ), 1 );
    auto  Y2 = tensor_product( Y1, adjoint( U2 ), 2 );
    auto  G  = tensor_product( Y2, adjoint( U3 ), 3 );

    return { std::move(G), std::move(U0), std::move(U1), std::move(U2), std::move(U3) };
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

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor3< value_t >,
            matrix< value_t >,
            vector< real_type_t< value_t > >,
            matrix< value_t >,
            vector< real_type_t< value_t > >,
            matrix< value_t >,
            vector< real_type_t< value_t > > >
hosvd_sv ( const tensor3< value_t > &  X,
           const accuracy &            acc,
           const approx_t &            apx )
{
    using  real_t = real_type_t< value_t >;
    
    auto  X0 = X.unfold( 0 );
    auto  S0 = vector< real_t >();
    auto  U0 = apx.column_basis( X0, acc, S0 );

    auto  X1 = X.unfold( 1 );
    auto  S1 = vector< real_t >();
    auto  U1 = apx.column_basis( X1, acc, S1 );

    auto  X2 = X.unfold( 2 );
    auto  S2 = vector< real_t >();
    auto  U2 = apx.column_basis( X2, acc, S2 );

    auto  Y0 = tensor_product( X,  adjoint( U0 ), 0 );
    auto  Y1 = tensor_product( Y0, adjoint( U1 ), 1 );
    auto  G  = tensor_product( Y1, adjoint( U2 ), 2 );

    return { std::move(G),
             std::move(U0), std::move( S0 ),
             std::move(U1), std::move( S1 ), 
             std::move(U2), std::move( S2 ) };
}

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::tuple< tensor4< value_t >,
            matrix< value_t >,
            vector< real_type_t< value_t > >,
            matrix< value_t >,
            vector< real_type_t< value_t > >,
            matrix< value_t >,
            vector< real_type_t< value_t > >,
            matrix< value_t >,
            vector< real_type_t< value_t > > >
hosvd_sv ( const tensor4< value_t > &  X,
           const accuracy &            acc,
           const approx_t &            apx )
{
    using  real_t = real_type_t< value_t >;
    
    auto  X0 = X.unfold( 0 );
    auto  S0 = vector< real_t >();
    auto  U0 = apx.column_basis( X0, acc, S0 );

    auto  X1 = X.unfold( 1 );
    auto  S1 = vector< real_t >();
    auto  U1 = apx.column_basis( X1, acc, S1 );

    auto  X2 = X.unfold( 2 );
    auto  S2 = vector< real_t >();
    auto  U2 = apx.column_basis( X2, acc, S2 );

    auto  X3 = X.unfold( 3 );
    auto  S3 = vector< real_t >();
    auto  U3 = apx.column_basis( X3, acc, S3 );

    auto  Y0 = tensor_product( X,  adjoint( U0 ), 0 );
    auto  Y1 = tensor_product( Y0, adjoint( U1 ), 1 );
    auto  Y2 = tensor_product( Y1, adjoint( U2 ), 2 );
    auto  G  = tensor_product( Y2, adjoint( U3 ), 3 );

    return { std::move(G),
             std::move(U0), std::move( S0 ),
             std::move(U1), std::move( S1 ), 
             std::move(U2), std::move( S2 ), 
             std::move(U3), std::move( S3 ) };
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

    // io::matlab::write( W0, "W0" );
    // io::matlab::write( W1, "W1" );
    // io::matlab::write( W2, "W2" );

    // io::matlab::write( S0, "S0" );
    // io::matlab::write( S1, "S1" );
    // io::matlab::write( S2, "S2" );

    // auto  G0 = G.unfold( 0 );
    // auto  G1 = G.unfold( 1 );
    // auto  G2 = G.unfold( 2 );

    // io::matlab::write( G0, "G0" );
    // io::matlab::write( G1, "G1" );
    // io::matlab::write( G2, "G2" );
    
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
// tensor cross approximation with full pivot search
//
template < typename value_t >
std::tuple< blas::tensor3< value_t >,
            blas::matrix< value_t >,
            blas::matrix< value_t >,
            blas::matrix< value_t > >
tca_full ( const blas::tensor3< value_t > &  O,
           const accuracy &                  acc,
           const int                         verbosity = 0 )
{
    auto        X       = blas::copy( O );
    auto        C       = std::deque< value_t >();
    const auto  max_k   = 2 * std::min( X.size(0), std::min( X.size(1), X.size(2) ) );
    auto        V0      = std::deque< blas::vector< value_t > >();
    auto        V1      = std::deque< blas::vector< value_t > >();
    auto        V2      = std::deque< blas::vector< value_t > >();
    size_t      step    = 0;

    //
    // set stopping criterion
    //
    
    auto  tol     = acc.abs_eps();
    auto  snrm_X0 = blas::dot( X, X );
    auto  snrm_Xi = snrm_X0;

    if ( acc.abs_eps() != 0 )
    {
        tol = math::square( acc.abs_eps() );
    }// if
    else if ( acc.rel_eps() != 0 )
    {
        tol = math::square( acc.rel_eps() ) * snrm_X0;
    }// if
    else
        HLR_ERROR( "zero error" );

    if ( verbosity >= 1 )
        std::cout << "  |X|:       " << math::sqrt( snrm_X0 ) << std::endl
                  << "  tolerance: " << math::sqrt( tol ) << std::endl;
    
    //
    // cross approximation iteration (yields CP representation Σ c_i · v⁰_i × v¹_i × v²_i )
    //
    
    while ( step < max_k )
    {
        // std::cout << X << std::endl;
    
        //
        // determine maximal element in X
        //

        auto     max_pos = std::array< uint, 3 >{ 0, 0, 0 };
        value_t  max_val = std::abs( X(0,0,0) );

        for ( uint  l = 0; l < X.size(2); ++l )
        {
            for ( uint  j = 0; j < X.size(1); ++j )
            {
                for ( uint  i = 0; i < X.size(0); ++i )
                {
                    const auto  X_ijl = std::abs( X(i,j,l) );

                    if ( X_ijl > max_val )
                    {
                        max_val = X_ijl;
                        max_pos = { i, j, l };
                    }// if
                }// for
            }// for
        }// for

        //
        // use fibers as next vectors
        //

        auto  v0 = blas::copy( X.fiber( 0, max_pos[1], max_pos[2] ) );
        auto  v1 = blas::copy( X.fiber( 1, max_pos[0], max_pos[2] ) );
        auto  v2 = blas::copy( X.fiber( 2, max_pos[0], max_pos[1] ) );

        if ( max_val < 1e-20 ) // just fail-safe
            break;

        blas::scale( 1.0 / max_val, v0 );
        blas::scale( 1.0 / max_val, v1 );
        blas::scale( 1.0 / max_val, v2 );

        //
        // subtract current rank-1 tensor from X and
        // recompute norm of rest
        //

        snrm_Xi = 0;
        
        for ( uint  l = 0; l < X.size(2); ++l )
        {
            for ( uint  j = 0; j < X.size(1); ++j )
            {
                for ( uint  i = 0; i < X.size(0); ++i )
                {
                    const auto  v_ijk = max_val * v0(i) * v1(j) * v2(l);
                    
                    snrm_Xi  += math::square( X(i,j,l) - v_ijk );
                    X(i,j,l) -= v_ijk;;
                }// for
            }// for
        }// for

        //
        // test convergence
        //
        
        // const auto  snrm_Xi = blas::norm_F( X );

        if ( verbosity >= 1 )
            std::cout << "  " << step << " : "
                      << math::sqrt( snrm_Xi ) << " / "
                      << math::sqrt( snrm_Xi / snrm_X0 ) << " | "
                      << max_val << std::endl;
        
        C.push_back( max_val );
        V0.push_back( std::move( v0 ) );
        V1.push_back( std::move( v1 ) );
        V2.push_back( std::move( v2 ) );

        if ( snrm_Xi < tol )
            break;
        
        step++;
    }// while

    const auto  k = C.size();

    if ( verbosity >= 1 )
        std::cout << "  TCA rank   : " << k << std::endl;

    //
    // convert to Tucker with diagonal core tensor
    //
    
    auto  X0 = blas::matrix< value_t >( X.size(0), k );
    auto  X1 = blas::matrix< value_t >( X.size(1), k );
    auto  X2 = blas::matrix< value_t >( X.size(2), k );
    auto  G  = blas::tensor3< value_t >( k, k, k );

    for ( uint  i = 0; i < k; ++i )
    {
        auto  v0 = V0[i];
        auto  v1 = V1[i];
        auto  v2 = V2[i];
        auto  g  = C[i];

        auto  X0_i = X0.column( i );
        auto  X1_i = X1.column( i );
        auto  X2_i = X2.column( i );

        blas::copy( v0, X0_i );
        blas::copy( v1, X1_i );
        blas::copy( v2, X2_i );
        G(i,i,i) = g;
    }// for

    // return { std::move( G ), std::move( X0 ), std::move( X1 ), std::move( X2 ) };

    //
    // orthogonalize bases X_i:
    //
    //     Qi, Ri = qr(Xi)
    //     X      ≈ G × X0 × X1 × X2
    //            = G × ( Q0·R0 ) × ( Q1·R1 ) × ( Q2·R2 )
    //            = ( G × R0 × R1 × R2 ) × Q0 × Q1 × Q2
    //

    auto  R0 = blas::matrix< value_t >();
    auto  R1 = blas::matrix< value_t >();
    auto  R2 = blas::matrix< value_t >();

    blas::qr( X0, R0 );
    blas::qr( X1, R1 );
    blas::qr( X2, R2 );

    auto  T0 = tensor_product( G,  R0, 0 );
    auto  T1 = tensor_product( T0, R1, 1 );
    auto  G2 = tensor_product( T1, R2, 2 );

    //
    // recompress ( G × R0 × R1 × R2 )
    //
    
    auto  apx               = approx::SVD< value_t >();
    auto  [ H, Y0, Y1, Y2 ] = blas::recompress( G2, X0, X1, X2, acc, apx );

    if ( verbosity >= 1 )
        std::cout << "  recompress : " << H.size(0) << " × " << H.size(1) << " × " << H.size(2) << std::endl;
    
    return { std::move( H ), std::move( Y0 ), std::move( Y1 ), std::move( Y2 ) };
}

}}// namespace hlr::blas

#endif // __HLR_BLAS_DETAIL_HOSVD_HH

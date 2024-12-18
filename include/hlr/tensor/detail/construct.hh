#ifndef __HLR_TENSOR_DETAIL_CONSTRUCT_HH
#define __HLR_TENSOR_DETAIL_CONSTRUCT_HH
//
// Project     : HLR
// Module      : tensor/construct
// Description : adaptive construction of an hierarchical tensor
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/hosvd.hh>

#include <hlr/tensor/dense_tensor.hh>
#include <hlr/tensor/tucker_tensor.hh>
#include <hlr/tensor/structured_tensor.hh>

#include <hlr/utils/io.hh>

namespace hlr { namespace tensor {

//
// build hierarchical tensor representation from given dense tensor, starting
// with subblocks of size ntile × ntile × ntile and then trying to merge
// tucker blocks as long as not increasing memory
//
namespace detail
{

template < typename value_t >
using tensor3_container_t = hlr::tensor3< std::unique_ptr< base_tensor3< value_t > > >;

//
// try to merge all sub blocks at once
//
template < typename                    value_t,
           approx::approximation_type  approx_t,
           typename                    hosvd_func_t >
std::unique_ptr< base_tensor3< value_t > >
merge_all ( const indexset &                  is0,
            const indexset &                  is1,
            const indexset &                  is2,
            const blas::tensor3< value_t > &  D,
            tensor3_container_t< value_t > &  sub_D,
            const tensor_accuracy &           acc,
            const approx_t &                  apx,
            hosvd_func_t &&                   hosvd )
{
    // verbosity level
    constexpr int verbosity = 0;
    
    //
    // determine ranks (and memory)
    //

    uint    rank[3] = { 0, 0, 0 };
    size_t  mem_sub = 0;

    for ( uint  l = 0; l < 2; ++l )
    {
        for ( uint  j = 0; j < 2; ++j )
        {
            for ( uint  i = 0; i < 2; ++i )
            {
                auto  D_ijl = cptrcast( sub_D(i,j,l).get(), tucker_tensor3< value_t > );
                        
                rank[0] += D_ijl->rank(0);
                rank[1] += D_ijl->rank(1);
                rank[2] += D_ijl->rank(2);

                mem_sub += sizeof(value_t) * ( rank[0] * rank[1] * rank[2] +
                                               rank[0] * D_ijl->is(0).size() +
                                               rank[1] * D_ijl->is(1).size() +
                                               rank[2] * D_ijl->is(2).size() );
            }// for
        }// for
    }// for

    //
    // decide how to proceed based on merged ranks
    //
            
    auto  G3 = blas::tensor3< value_t >();
    auto  Y0 = blas::matrix< value_t >();
    auto  Y1 = blas::matrix< value_t >();
    auto  Y2 = blas::matrix< value_t >();
        
    if constexpr ( verbosity >= 1 )
        std::cout << "        " << is0 << " × " << is1 << " × " << is2 << " (merged ranks) : "
                  << rank[0] << " / " << rank[1] << " / " << rank[2] << std::endl;
            
    if ( std::min({ rank[0], rank[1], rank[2] }) >= std::min({ D.size(0), D.size(1), D.size(2) }) )
    {
        //
        // directly use HOSVD on D as merged ranks are too large
        //
                
        auto        Dc   = blas::copy( D );  // do not modify D (!)
        const auto  lacc = acc( is0, is1, is2 );

        std::tie( G3, Y0, Y1, Y2 ) = std::move( hosvd( Dc, lacc, apx ) );

        if constexpr ( verbosity >= 1 )
            std::cout << "        " << is0 << " × " << is1 << " × " << is2 << " (hosvd)        : "
                      << G3.size(0) << " / " << G3.size(1) << " / " << G3.size(2) << std::endl;
            
        if constexpr ( verbosity >= 2 )
            std::cout << "hosvd          : " << blas::tucker_error( D, G3, Y0, Y1, Y2 ) << std::endl;
    }// if
    else
    {
        //
        // construct merged tucker representation
        //

        auto  G      = blas::tensor3< value_t >( rank[0], rank[1], rank[2] );
        auto  X0     = blas::matrix< value_t >( is0.size(), rank[0] );
        auto  X1     = blas::matrix< value_t >( is1.size(), rank[1] );
        auto  X2     = blas::matrix< value_t >( is2.size(), rank[2] );
        uint  ofs[3] = { 0, 0, 0 };
            
        for ( uint  l = 0; l < 2; ++l )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                for ( uint  i = 0; i < 2; ++i )
                {
                    auto  D_ijl  = cptrcast( sub_D(i,j,l).get(), tucker_tensor3< value_t > );
                    auto  k0     = blas::range( ofs[0], ofs[0] + D_ijl->rank(0) - 1 );
                    auto  k1     = blas::range( ofs[1], ofs[1] + D_ijl->rank(1) - 1 );
                    auto  k2     = blas::range( ofs[2], ofs[2] + D_ijl->rank(2) - 1 );
                    auto  G_ijl  = blas::tensor3< value_t >( G, k0, k1, k2 );
                    auto  X0_ijl = blas::matrix< value_t >( X0, D_ijl->is(0) - is0.first(), k0 );
                    auto  X1_ijl = blas::matrix< value_t >( X1, D_ijl->is(1) - is1.first(), k1 );
                    auto  X2_ijl = blas::matrix< value_t >( X2, D_ijl->is(2) - is2.first(), k2 );

                    blas::copy( D_ijl->G_decompressed(), G_ijl );
                    blas::copy( D_ijl->X_decompressed( 0 ), X0_ijl );
                    blas::copy( D_ijl->X_decompressed( 1 ), X1_ijl );
                    blas::copy( D_ijl->X_decompressed( 2 ), X2_ijl );

                    ofs[0] += D_ijl->rank(0);
                    ofs[1] += D_ijl->rank(1);
                    ofs[2] += D_ijl->rank(2);
                }// for
            }// for
        }// for

        // io::hdf5::write( D, "D" );
        // io::hdf5::write( G, "G" );
        // io::matlab::write( X0, "X0" );
        // io::matlab::write( X1, "X1" );
        // io::matlab::write( X2, "X2" );

        if constexpr ( verbosity >= 2 )
            std::cout << "merged         : " << blas::tucker_error( D, G, X0, X1, X2 ) << std::endl;

        //
        // orthogonalize merged Tucker tensor
        //

        auto  R0 = blas::matrix< value_t >();
        auto  R1 = blas::matrix< value_t >();
        auto  R2 = blas::matrix< value_t >();
            
        blas::qr( X0, R0 );
        blas::qr( X1, R1 );
        blas::qr( X2, R2 );

        auto  W0 = blas::tensor_product( G,  R0, 0 );
        auto  W1 = blas::tensor_product( W0, R1, 1 );
        auto  G2 = blas::tensor_product( W1, R2, 2 );
            
        if constexpr ( verbosity >= 2 )
            std::cout << "orthogonalized : " << blas::tucker_error( D, G2, X0, X1, X2 ) << std::endl;

        //
        // compress with respect to local accuracy
        //
            
        const auto  lacc = acc( is0, is1, is2 );

        std::tie( G3, Y0, Y1, Y2 ) = std::move( blas::recompress( G2, X0, X1, X2, lacc, apx, hosvd ) );

        if constexpr ( verbosity >= 1 )
            std::cout << "        " << is0 << " × " << is1 << " × " << is2 << " (recompressed) : "
                      << G3.size(0) << " / " << G3.size(1) << " / " << G3.size(2) << std::endl;
            
        if constexpr ( verbosity >= 2 )
            std::cout << "recompressed   : " << blas::tucker_error( D, G3, Y0, Y1, Y2 ) << std::endl;
    }// if
            
    // io::matlab::write( Y0, "Y0" );
    // io::matlab::write( Y1, "Y1" );
    // io::matlab::write( Y2, "Y2" );
            
    //
    // return coarse tucker tensor if more memory efficient
    //

    const auto  mem_full   = sizeof(value_t) * ( D.size(0) * D.size(1) * D.size(2) );
    const auto  mem_merged = sizeof(value_t) * ( G3.size(0) * G3.size(1) * G3.size(2) +
                                                 Y0.nrows() * Y0.ncols() +
                                                 Y1.nrows() * Y1.ncols() +
                                                 Y2.nrows() * Y2.ncols() );

    if constexpr ( verbosity >= 1 )
        std::cout << "        " << is0 << " × " << is1 << " × " << is2 << " (memory S/M/F) : " << mem_sub << " / " << mem_merged << " / " << mem_full << std::endl;
            
    if ( mem_merged < std::min( mem_sub, mem_full ) )
    {
        if constexpr ( verbosity >= 1 )
            std::cout << "TUCKER: " << to_string( is0 ) << " × " << to_string( is1 ) << " × " << to_string( is2 ) << std::endl;
                
        return std::make_unique< tucker_tensor3< value_t > >( is0, is1, is2,
                                                              std::move( G3 ),
                                                              std::move( Y0 ),
                                                              std::move( Y1 ),
                                                              std::move( Y2 ) );
    }// if
            
    if ( mem_full < mem_sub )
    {
        if constexpr ( verbosity >= 1 )
            std::cout << "DENSE : " << to_string( is0 ) << " × " << to_string( is1 ) << " × " << to_string( is2 ) << std::endl;
                
        return std::make_unique< dense_tensor3< value_t > >( is0, is1, is2, std::move( blas::copy( D ) ) );
    }// if

    //
    // don't merge => empty tensor as result
    //

    return std::unique_ptr< base_tensor3< value_t > >();
}

//
// merge two given tensors
//
template < typename                    value_t,
           approx::approximation_type  approx_t,
           typename                    hosvd_func_t >
std::unique_ptr< base_tensor3< value_t > >
merge_two ( const indexset &                   is0,
            const indexset &                   is1,
            const indexset &                   is2,
            const blas::tensor3< value_t > &   D,
            const tucker_tensor3< value_t > &  T0,
            const tucker_tensor3< value_t > &  T1,
            const tensor_accuracy &            acc,
            const approx_t &                   apx,
            hosvd_func_t &&                    hosvd )
{
    // verbosity level
    constexpr int verbosity = 0;

    //
    // put tensors into array for simpler access below
    //

    auto  T = std::array< const tucker_tensor3< value_t > *, 2 >({ & T0, & T1 });
    
    //
    // determine ranks (and memory)
    //

    const uint  rank[3] = { T0.rank(0) + T1.rank(0), T0.rank(1) + T1.rank(1), T0.rank(2) + T1.rank(2) };
    size_t      mem_sub = 0;

    for ( auto  t : T )
        mem_sub += sizeof(value_t) * ( t->rank(0) * t->rank(1) * t->rank(2) +
                                       t->rank(0) * t->is(0).size() +
                                       t->rank(1) * t->is(1).size() +
                                       t->rank(2) * t->is(2).size() );
    
    //
    // decide how to proceed based on merged ranks
    //
            
    auto  G3 = blas::tensor3< value_t >();
    auto  Y0 = blas::matrix< value_t >();
    auto  Y1 = blas::matrix< value_t >();
    auto  Y2 = blas::matrix< value_t >();
        
    if constexpr ( verbosity >= 1 )
        std::cout << "        " << is0 << " × " << is1 << " × " << is2 << " (merged ranks) : "
                  << rank[0] << " / " << rank[1] << " / " << rank[2] << std::endl;
            
    if ( std::min({ rank[0], rank[1], rank[2] }) >= std::min({ D.size(0), D.size(1), D.size(2) }) )
    {
        //
        // directly use HOSVD on D as merged ranks are too large
        //
                
        auto        Dc   = blas::copy( D );  // do not modify D (!)
        const auto  lacc = acc( is0, is1, is2 );

        std::tie( G3, Y0, Y1, Y2 ) = std::move( hosvd( Dc, lacc, apx ) );

        if constexpr ( verbosity >= 1 )
            std::cout << "        " << is0 << " × " << is1 << " × " << is2 << " (hosvd)        : "
                      << G3.size(0) << " / " << G3.size(1) << " / " << G3.size(2) << std::endl;
            
        if constexpr ( verbosity >= 2 )
            std::cout << "hosvd          : " << blas::tucker_error( D, G3, Y0, Y1, Y2 ) << std::endl;
    }// if
    else
    {
        //
        // construct merged tucker representation
        //

        auto  G      = blas::tensor3< value_t >( rank[0], rank[1], rank[2] );
        auto  X0     = blas::matrix< value_t >( is0.size(), rank[0] );
        auto  X1     = blas::matrix< value_t >( is1.size(), rank[1] );
        auto  X2     = blas::matrix< value_t >( is2.size(), rank[2] );
        uint  ofs[3] = { 0, 0, 0 };

        for ( auto  t : T )
        {
            auto  k0     = blas::range( ofs[0], ofs[0] + t->rank(0) - 1 );
            auto  k1     = blas::range( ofs[1], ofs[1] + t->rank(1) - 1 );
            auto  k2     = blas::range( ofs[2], ofs[2] + t->rank(2) - 1 );
            auto  G_ijl  = blas::tensor3< value_t >( G, k0, k1, k2 );
            auto  X0_ijl = blas::matrix< value_t >( X0, t->is(0) - is0.first(), k0 );
            auto  X1_ijl = blas::matrix< value_t >( X1, t->is(1) - is1.first(), k1 );
            auto  X2_ijl = blas::matrix< value_t >( X2, t->is(2) - is2.first(), k2 );
        
            blas::copy( t->G_decompressed(), G_ijl );
            blas::copy( t->X_decompressed( 0 ), X0_ijl );
            blas::copy( t->X_decompressed( 1 ), X1_ijl );
            blas::copy( t->X_decompressed( 2 ), X2_ijl );
        
            ofs[0] += t->rank(0);
            ofs[1] += t->rank(1);
            ofs[2] += t->rank(2);
        }// for
        
        // io::hdf5::write( D, "D" );
        // io::hdf5::write( G, "G" );
        // io::matlab::write( X0, "X0" );
        // io::matlab::write( X1, "X1" );
        // io::matlab::write( X2, "X2" );

        if constexpr ( verbosity >= 2 )
            std::cout << "merged         : " << blas::tucker_error( D, G, X0, X1, X2 ) << std::endl;

        //
        // orthogonalize merged Tucker tensor
        //

        auto  R0 = blas::matrix< value_t >();
        auto  R1 = blas::matrix< value_t >();
        auto  R2 = blas::matrix< value_t >();
            
        blas::qr( X0, R0 );
        blas::qr( X1, R1 );
        blas::qr( X2, R2 );

        auto  W0 = blas::tensor_product( G,  R0, 0 );
        auto  W1 = blas::tensor_product( W0, R1, 1 );
        auto  G2 = blas::tensor_product( W1, R2, 2 );
            
        if constexpr ( verbosity >= 2 )
            std::cout << "orthogonalized : " << blas::tucker_error( D, G2, X0, X1, X2 ) << std::endl;

        //
        // compress with respect to local accuracy
        //
            
        const auto  lacc = acc( is0, is1, is2 );

        std::tie( G3, Y0, Y1, Y2 ) = std::move( blas::recompress( G2, X0, X1, X2, lacc, apx, hosvd ) );

        if constexpr ( verbosity >= 1 )
            std::cout << "        " << is0 << " × " << is1 << " × " << is2 << " (recompressed) : "
                      << G3.size(0) << " / " << G3.size(1) << " / " << G3.size(2) << std::endl;
            
        if constexpr ( verbosity >= 2 )
            std::cout << "recompressed   : " << blas::tucker_error( D, G3, Y0, Y1, Y2 ) << std::endl;
    }// if
            
    // io::matlab::write( Y0, "Y0" );
    // io::matlab::write( Y1, "Y1" );
    // io::matlab::write( Y2, "Y2" );
            
    //
    // return coarse tucker tensor if more memory efficient
    //

    const auto  mem_full   = sizeof(value_t) * ( D.size(0) * D.size(1) * D.size(2) );
    const auto  mem_merged = sizeof(value_t) * ( G3.size(0) * G3.size(1) * G3.size(2) +
                                                 Y0.nrows() * Y0.ncols() +
                                                 Y1.nrows() * Y1.ncols() +
                                                 Y2.nrows() * Y2.ncols() );

    if constexpr ( verbosity >= 1 )
        std::cout << "        " << is0 << " × " << is1 << " × " << is2 << " (memory S/M/F) : " << mem_sub << " / " << mem_merged << " / " << mem_full << std::endl;
            
    if ( mem_merged < std::min( mem_sub, mem_full ) )
    {
        if constexpr ( verbosity >= 1 )
            std::cout << "TUCKER: " << to_string( is0 ) << " × " << to_string( is1 ) << " × " << to_string( is2 ) << std::endl;
                
        return std::make_unique< tucker_tensor3< value_t > >( is0, is1, is2,
                                                              std::move( G3 ),
                                                              std::move( Y0 ),
                                                              std::move( Y1 ),
                                                              std::move( Y2 ) );
    }// if
            
    if ( mem_full < mem_sub )
    {
        if constexpr ( verbosity >= 1 )
            std::cout << "DENSE : " << to_string( is0 ) << " × " << to_string( is1 ) << " × " << to_string( is2 ) << std::endl;
                
        return std::make_unique< dense_tensor3< value_t > >( is0, is1, is2, std::move( blas::copy( D ) ) );
    }// if

    //
    // don't merge => empty tensor as result
    //

    return std::unique_ptr< base_tensor3< value_t > >();
}

//
// merge sub blocks per dimension
//
template < typename                    value_t,
           approx::approximation_type  approx_t,
           typename                    hosvd_func_t >
std::unique_ptr< base_tensor3< value_t > >
merge_dim ( const indexset &                  is0,
            const indexset &                  is1,
            const indexset &                  is2,
            const blas::tensor3< value_t > &  D,
            tensor3_container_t< value_t > &  sub_D,
            const tensor_accuracy &           acc,
            const approx_t &                  apx,
            hosvd_func_t &&                   hosvd )
{
    //
    // merge dimension 0
    //

    auto    T_l     = std::array< std::unique_ptr< base_tensor3< value_t > >, 2 >();
    size_t  mem_sub = 0;
    
    for ( uint  l = 0; l < 2; ++l )
    {
        auto  T_jl = std::array< std::unique_ptr< base_tensor3< value_t > >, 2 >();
        
        for ( uint  j = 0; j < 2; ++j )
        {
            auto  T_0jl = cptrcast( sub_D(0,j,l).get(), tucker_tensor3< value_t > );
            auto  T_1jl = cptrcast( sub_D(1,j,l).get(), tucker_tensor3< value_t > );
            auto  D_jl  = blas::tensor3< value_t >( D, is0, T_0jl->is(1) - is1.first(), T_0jl->is(2) - is2.first() );

            mem_sub += T_0jl->byte_size() + T_1jl->byte_size();
            
            T_jl[j] = std::move( merge_two( is0, T_0jl->is(1), T_0jl->is(2), D_jl, *T_0jl, *T_1jl, acc, apx, hosvd ) );

            // stop immediately if not mergable
            if ( ! is_tucker( *T_jl[j] ) )
                return std::unique_ptr< base_tensor3< value_t > >();
        }// for

        auto  D_l  = blas::tensor3< value_t >( D, is0, is1, T_jl[0]->is(2) - is2.first() );
        
        T_l[l] = std::move( merge_two( is0, is1, T_jl[0]->is(2), D_l,
                                       *( ptrcast( T_jl[0].get(), tucker_tensor3< value_t > ) ),
                                       *( ptrcast( T_jl[1].get(), tucker_tensor3< value_t > ) ),
                                       acc, apx, hosvd ) );

        if ( ! is_tucker( *T_l[l] ) )
            return std::unique_ptr< base_tensor3< value_t > >();
    }// for

    auto  T_merged = merge_two( is0, is1, is2, D,
                                *( ptrcast( T_l[0].get(), tucker_tensor3< value_t > ) ),
                                *( ptrcast( T_l[1].get(), tucker_tensor3< value_t > ) ),
                                acc, apx, hosvd );

    if ( mem_sub < T_merged->byte_size() )
        return T_merged;
    else
        return std::unique_ptr< base_tensor3< value_t > >();
}

//
// greedy merge of sub blocks
//
template < typename                    value_t,
           approx::approximation_type  approx_t,
           typename                    hosvd_func_t >
std::unique_ptr< base_tensor3< value_t > >
merge_greedy ( const indexset &                  is0,
               const indexset &                  is1,
               const indexset &                  is2,
               const blas::tensor3< value_t > &  D,
               tensor3_container_t< value_t > &  sub_D,
               const tensor_accuracy &           acc,
               const approx_t &                  apx,
               hosvd_func_t &&                   hosvd )
{
    using  real_t = Hpro::real_type_t< value_t >;
    
    // verbosity level
    constexpr int verbosity = 0;
    
    //
    // determine memory of partitioning
    // - also compute singular values for all mode matrices
    //

    size_t  mem_sub = 0;
    auto    S       = hlr::tensor3< std::array< blas::vector< real_t >, 3 > >( 2, 2, 2 );
    auto    U       = hlr::tensor3< std::array< blas::matrix< value_t >, 3 > >( 2, 2, 2 );
    auto    maxrank = std::vector< uint >( 3, 0 );

    for ( uint  l = 0; l < 2; ++l )
    {
        for ( uint  j = 0; j < 2; ++j )
        {
            for ( uint  i = 0; i < 2; ++i )
            {
                auto  D_ijl   = cptrcast( sub_D(i,j,l).get(), tucker_tensor3< value_t > );
                uint  rank[3] = { 0, 0, 0 };
                
                rank[0] += D_ijl->rank(0);
                rank[1] += D_ijl->rank(1);
                rank[2] += D_ijl->rank(2);

                maxrank[0] += rank[0];
                maxrank[1] += rank[1];
                maxrank[2] += rank[2];

                mem_sub += sizeof(value_t) * ( rank[0] * rank[1] * rank[2] +
                                               rank[0] * D_ijl->is(0).size() +
                                               rank[1] * D_ijl->is(1).size() +
                                               rank[2] * D_ijl->is(2).size() );

                //
                // compute singular values for all mode matrices
                //

                auto  G_ijl  = D_ijl->G();

                // auto  X0         = G_ijl.unfold( 0 );
                // auto  [ U0, S0 ] = blas::svd( X0, true );
                
                // auto  X1         = G_ijl.unfold( 1 );
                // auto  [ U1, S1 ] = blas::svd( X1, true );
                
                // auto  X2         = G_ijl.unfold( 2 );
                // auto  [ U2, S2 ] = blas::svd( X2, true );

                auto  X0 = G_ijl.unfold( 0 );
                auto  S0 = blas::sv( X0 );
                
                auto  X1 = G_ijl.unfold( 1 );
                auto  S1 = blas::sv( X1 );
                
                auto  X2 = G_ijl.unfold( 2 );
                auto  S2 = blas::sv( X2 );

                S(i,j,l)[0] = std::move( S0 );
                S(i,j,l)[1] = std::move( S1 );
                S(i,j,l)[2] = std::move( S2 );

                // U(i,j,l)[0] = std::move( U0 );
                // U(i,j,l)[1] = std::move( U1 );
                // U(i,j,l)[2] = std::move( U2 );
            }// for
        }// for
    }// for

    if constexpr ( verbosity >= 1 )
        std::cout << "max ranks : " << maxrank[0] << " / " << maxrank[1] << " / " << maxrank[2] << std::endl;

    //
    // compute norm of merged tensor and set relative tolerance
    //
    
    auto    R      = hlr::tensor3< std::array< uint, 3 > >( 2, 2, 2 );
    real_t  sqnorm = 0;

    for ( uint  l = 0; l < 2; ++l )
        for ( uint  j = 0; j < 2; ++j )
            for ( uint  i = 0; i < 2; ++i )
            {
                for ( uint  d = 0; d < 3; ++d )
                    R(i,j,l)[d] = 0;

                // only count for first dimension as sum of sv in other dimensions is identical
                for ( uint  k = 0; k < S(i,j,l)[0].length(); ++k )
                    sqnorm += S(i,j,l)[0](k) * S(i,j,l)[0](k);
            }// for

    if constexpr ( verbosity >= 1 )
        std::cout << "|T|  = " << std::sqrt( sqnorm ) << std::endl;

    const auto  sqtol = acc.abs_eps() * acc.abs_eps() * sqnorm;

    if constexpr ( verbosity >= 1 )
        std::cout << "tol² = " << sqtol << std::endl;

    //
    // greedily choose next largest singular value in all sub blocks
    // until error is met
    //
    
    uint    ranks[3] = { 0, 0, 0 };
    real_t  sqerror  = sqnorm;

    while ( sqerror > sqtol )
    {
        auto  max_pos = std::array< uint, 5 >();
        auto  max_sv  = real_t(0);
            
        for ( uint  l = 0; l < 2; ++l )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                for ( uint  i = 0; i < 2; ++i )
                {
                    auto  D_ijl = cptrcast( sub_D(i,j,l).get(), tucker_tensor3< value_t > );

                    for ( uint  d = 0; d < 3; ++d )
                    {
                        const auto  k = R(i,j,l)[d];
                            
                        if ( k >= S(i,j,l)[d].length() )
                            continue;
                        
                        if ( S(i,j,l)[d](k) > max_sv )
                        {
                            max_sv  = S(i,j,l)[d](k);
                            max_pos = { i, j, l, d, k };
                        }// if
                    }// for
                }// for
            }// for
        }// for

        // remember which singular value was consumed
        R( max_pos[0], max_pos[1], max_pos[2] )[ max_pos[3] ] += 1;
        ranks[max_pos[3]]++;

        //
        // update squared error (recompute to avoid floating point issues)
        // - iterate over all dimensions to really catch reduction
        //   in error, e.g., if sv in dimension != 0 was consumed
        //
        
        sqerror = real_t(0);
        
        for ( uint  l = 0; l < 2; ++l )
            for ( uint  j = 0; j < 2; ++j )
                for ( uint  i = 0; i < 2; ++i )
                    for ( uint  d = 0; d < 3; ++d )
                        for ( uint  k = R(i,j,l)[d]; k < S(i,j,l)[d].length(); ++k )
                            sqerror += S(i,j,l)[d](k) * S(i,j,l)[d](k);
        sqerror /= real_t(3); // correct for multiple (x dims) summation 
        
        // error -= max_sv * max_sv;

        if constexpr ( verbosity >= 1 )
            std::cout << max_pos[0] << " / " << max_pos[1] << " / " << max_pos[2] << " / " << max_pos[3] << " / " << max_pos[4] << " : "
                      << max_sv << " / " << max_sv*max_sv << " / " << sqerror << std::endl;
    }// while

    if constexpr ( verbosity >= 1 )
        std::cout << ranks[0] << " / " << ranks[1] << " / " << ranks[2] << std::endl;

    //
    // decide what to do based on ranks (and memory size)
    //
    
    const auto  mem_full   = sizeof(value_t) * ( D.size(0) * D.size(1) * D.size(2) );
    const auto  mem_merged = sizeof(value_t) * ( ranks[0] * ranks[1] * ranks[2] +
                                                 is0.size() * ranks[0] +
                                                 is1.size() * ranks[1] +
                                                 is2.size() * ranks[2] );

    if constexpr ( verbosity >= 1 )
        std::cout << "        " << is0 << " × " << is1 << " × " << is2 << " (memory S/M/F) : " << mem_sub << " / " << mem_merged << " / " << mem_full << std::endl;
            
    if ( mem_merged >= std::min( mem_sub, mem_full ) )
    {
        //
        // merged approximation uses more memory, so stop immediately
        //

        return std::unique_ptr< base_tensor3< value_t > >();
    }// if

    //
    // construct new tensor from all collected singular values in all subblocks
    //

    auto  G      = blas::tensor3< value_t >( ranks[0], ranks[1], ranks[2] );
    auto  X0     = blas::matrix< value_t >( is0.size(), ranks[0] );
    auto  X1     = blas::matrix< value_t >( is1.size(), ranks[1] );
    auto  X2     = blas::matrix< value_t >( is2.size(), ranks[2] );
    uint  ofs[3] = { 0, 0, 0 };
            
    for ( uint  l = 0; l < 2; ++l )
    {
        for ( uint  j = 0; j < 2; ++j )
        {
            for ( uint  i = 0; i < 2; ++i )
            {
                auto  D_ijl  = cptrcast( sub_D(i,j,l).get(), tucker_tensor3< value_t > );
                auto  G_ijl  = D_ijl->G();
                auto  X0_ijl = D_ijl->X( 0 );
                auto  X1_ijl = D_ijl->X( 1 );
                auto  X2_ijl = D_ijl->X( 2 );

                // get left singular vectors of unfolded core tensor
                auto  GX0        = G_ijl.unfold( 0 );
                auto  [ U0, S0 ] = blas::svd( GX0, true );
                auto  GX1        = G_ijl.unfold( 1 );
                auto  [ U1, S1 ] = blas::svd( GX1, true );
                auto  GX2        = G_ijl.unfold( 2 );
                auto  [ U2, S2 ] = blas::svd( GX2, true );
                // auto  U0 = U(i,j,l)[0];
                // auto  U1 = U(i,j,l)[1];
                // auto  U2 = U(i,j,l)[2];

                // ranges needed to be copied
                auto  k0 = blas::range( ofs[0], ofs[0] + R(i,j,l)[0] - 1 );
                auto  k1 = blas::range( ofs[1], ofs[1] + R(i,j,l)[1] - 1 );
                auto  k2 = blas::range( ofs[2], ofs[2] + R(i,j,l)[2] - 1 );

                // selected parts of mode matrices
                auto  TU0 = blas::matrix< value_t >( U0, blas::range::all, blas::range( 0, R(i,j,l)[0] - 1 ) );
                auto  TU1 = blas::matrix< value_t >( U1, blas::range::all, blas::range( 0, R(i,j,l)[1] - 1 ) );
                auto  TU2 = blas::matrix< value_t >( U2, blas::range::all, blas::range( 0, R(i,j,l)[2] - 1 ) );

                // reduced core tensor
                auto  TY0 = tensor_product( G_ijl, adjoint( TU0 ), 0 );
                auto  TY1 = tensor_product( TY0,   adjoint( TU1 ), 1 );
                auto  TG  = tensor_product( TY1,   adjoint( TU2 ), 2 );

                // reduced mode matrices
                auto  TX0 = blas::prod( X0_ijl, TU0 );
                auto  TX1 = blas::prod( X1_ijl, TU1 );
                auto  TX2 = blas::prod( X2_ijl, TU2 );
                                        
                // parts in global tensor data
                auto  G_sub  = blas::tensor3< value_t >( G, k0, k1, k2 );
                auto  X0_sub = blas::matrix< value_t >( X0, D_ijl->is(0) - is0.first(), k0 );
                auto  X1_sub = blas::matrix< value_t >( X1, D_ijl->is(1) - is1.first(), k1 );
                auto  X2_sub = blas::matrix< value_t >( X2, D_ijl->is(2) - is2.first(), k2 );

                blas::copy( TG,  G_sub );
                blas::copy( TX0, X0_sub );
                blas::copy( TX1, X1_sub );
                blas::copy( TX2, X2_sub );

                ofs[0] += R(i,j,l)[0];
                ofs[1] += R(i,j,l)[1];
                ofs[2] += R(i,j,l)[2];
            }// for
        }// for
    }// for

    if constexpr ( verbosity >= 1 )
        std::cout << "error = " << blas::tucker_error( D, G, X0, X1, X2 ) << std::endl
                  << "        " << blas::tucker_error( D, G, X0, X1, X2 ) / blas::norm_F( D ) << std::endl;

    //
    // don't merge => empty tensor as result
    //

    return std::make_unique< tucker_tensor3< value_t > >( is0, is1, is2,
                                                          std::move( G ),
                                                          std::move( X0 ),
                                                          std::move( X1 ),
                                                          std::move( X2 ) );
}

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< base_tensor3< value_t > >
build_hierarchical_tucker ( const indexset &                  is0,
                            const indexset &                  is1,
                            const indexset &                  is2,
                            const blas::tensor3< value_t > &  D,
                            const tensor_accuracy &           acc,
                            const approx_t &                  apx,
                            const size_t                      ntile )
{
    // verbosity level
    constexpr int verbosity = 0;
    
    // choose one to be used below
    // auto  hosvd = blas::hosvd< value_t, approx_t >;
    // auto  hosvd = blas::sthosvd< value_t, approx_t >;
    auto  hosvd = blas::greedy_hosvd< value_t, approx_t >;
        
    if ( std::min( D.size(0), std::min( D.size(1), D.size(2) ) ) <= ntile )
    {
        //
        // build leaf
        //

        const auto  lacc = acc( is0, is1, is2 );
        
        if ( ! lacc.is_exact() )
        {
            auto  Dc                = blas::copy( D );  // do not modify D (!)
            auto  [ G, X0, X1, X2 ] = hosvd( Dc, lacc, apx );
            auto  mem_tucker        = G.byte_size() + X0.byte_size() + X1.byte_size() + X2.byte_size();
            
            if ( mem_tucker < Dc.byte_size() )
            {
                if constexpr ( verbosity >= 1 )
                    std::cout << "TUCKER: " << to_string( is0 ) << " × " << to_string( is1 ) << " × " << to_string( is2 )
                              << " : " << G.size(0) << " / " << G.size(1) << " / " << G.size(2)
                              << " ( " << mem_tucker << " / " << Dc.byte_size() << " )"
                              << std::endl;

                if constexpr ( verbosity >= 2 )
                    std::cout << "hosvd          : " << blas::tucker_error( D, G, X0, X1, X2 ) << std::endl;

                return std::make_unique< tucker_tensor3< value_t > >( is0, is1, is2,
                                                                      std::move( G ),
                                                                      std::move( X0 ),
                                                                      std::move( X1 ),
                                                                      std::move( X2 ) );
            }// if
        }// if

        if constexpr ( verbosity >= 1 )
            std::cout << "DENSE : " << to_string( is0 ) << " × " << to_string( is1 ) << " × " << to_string( is2 ) << std::endl;

        return std::make_unique< dense_tensor3< value_t > >( is0, is1, is2, std::move( blas::copy( D ) ) );
    }// if
    else
    {
        //
        // Recursion
        //

        const auto  mid0 = ( is0.first() + is0.last() + 1 ) / 2;
        const auto  mid1 = ( is1.first() + is1.last() + 1 ) / 2;
        const auto  mid2 = ( is2.first() + is2.last() + 1 ) / 2;

        indexset    sub_is0[2] = { indexset( is0.first(), mid0-1 ), indexset( mid0, is0.last() ) };
        indexset    sub_is1[2] = { indexset( is1.first(), mid1-1 ), indexset( mid1, is1.last() ) };
        indexset    sub_is2[2] = { indexset( is2.first(), mid2-1 ), indexset( mid2, is2.last() ) };
        auto        sub_D      = hlr::tensor3< std::unique_ptr< base_tensor3< value_t > > >( 2, 2, 2 );
        bool        all_tucker = true;
        bool        all_dense  = true;

        for ( uint  l = 0; l < 2; ++l )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                for ( uint  i = 0; i < 2; ++i )
                {
                    const auto  D_sub = D( sub_is0[i] - is0.first(),
                                           sub_is1[j] - is1.first(),
                                           sub_is2[l] - is2.first() );
                    
                    sub_D(i,j,l) = build_hierarchical_tucker( sub_is0[i], sub_is1[j], sub_is2[l], D_sub, acc, apx, ntile );
                    
                    HLR_ASSERT( sub_D(i,j,l).get() != nullptr );

                    if ( ! is_tucker_tensor3( *sub_D(i,j,l) ) )
                        all_tucker = false;

                    if ( ! is_dense_tensor3( *sub_D(i,j,l) ) )
                        all_dense = false;
                }// for
            }// for
        }// for

        //
        // if all sub blocks are Tucker tensors, try to merge
        //

        auto  T = std::unique_ptr< base_tensor3< value_t > >();
        
        if ( all_tucker )
        {
            // T = std::move( merge_all( is0, is1, is2, D, sub_D, acc, apx, hosvd ) );
            T = std::move( merge_greedy( is0, is1, is2, D, sub_D, acc, apx, hosvd ) );
            // T = std::move( merge_dim( is0, is1, is2, D, sub_D, acc, apx, hosvd ) );
        }// if

        if ( ! is_null( T ) )
            return T;
        
        //
        // also return full tensor if all sub blocks are dense
        //

        if ( all_dense )
            return std::make_unique< dense_tensor3< value_t > >( is0, is1, is2, std::move( blas::copy( D ) ) );
        
        //
        // construct structured tensor
        //

        if constexpr ( verbosity >= 1 )
            std::cout << "BLOCK : " << to_string( is0 ) << " × " << to_string( is1 ) << " × " << to_string( is2 ) << std::endl;
        
        auto  B = std::make_unique< structured_tensor3< value_t > >( is0, is1, is2 );

        B->set_structure( 2, 2, 2 );
        
        for ( uint  l = 0; l < 2; ++l )
            for ( uint  i = 0; i < 2; ++i )
                for ( uint  j = 0; j < 2; ++j )
                    B->set_block( i, j, l, sub_D(i,j,l).release() );
            
        return B;
    }// else
}

}}}// namespace hlr::tensor::detail

#endif // __HLR_TENSOR_DETAIL_CONSTRUCT_HH

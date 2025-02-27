#ifndef __HLR_TBB_TENSOR_HH
#define __HLR_TBB_TENSOR_HH
//
// Project     : HLR
// Module      : tbb/tensor
// Description : tensor algorithms using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <tbb/blocked_range3d.h>
#include <tbb/parallel_for.h>

#include <hlr/tensor/construct.hh>
#include <hlr/tensor/convert.hh>
#include <hlr/tensor/dense_tensor.hh>
#include <hlr/tensor/tucker_tensor.hh>
#include <hlr/tensor/structured_tensor.hh>

#include <hlr/tbb/detail/tensor.hh>

namespace hlr { namespace tbb { namespace tensor {

using hlr::tensor::base_tensor3;
using hlr::tensor::tucker_tensor3;
using hlr::tensor::dense_tensor3;
using hlr::tensor::structured_tensor3;

using hlr::tensor::base_tensor4;
using hlr::tensor::tucker_tensor4;
using hlr::tensor::dense_tensor4;
using hlr::tensor::structured_tensor4;

//
// build hierarchical tensor representation from given dense tensor, starting
// with subblocks of size ntile × ntile × ntile and then trying to merge
// tucker blocks as long as not increasing memory
//
namespace detail
{

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< base_tensor3< value_t > >
build_hierarchical_tucker ( const indexset &                  is0,
                            const indexset &                  is1,
                            const indexset &                  is2,
                            const blas::tensor3< value_t > &  D,
                            const size_t                      ntile,
                            const accuracy &                  tacc,
                            const approx_t &                  apx,
                            const accuracy &                  fpacc )
{
    // choose one to be used below
    // auto  hosvd = blas::hosvd< value_t, approx_t >;
    // auto  hosvd = blas::sthosvd< value_t, approx_t >;
    // auto  hosvd = blas::greedy_hosvd< value_t, approx_t >;

    if ( std::min( D.size(0), std::min( D.size(1), D.size(2) ) ) <= ntile )
    {
        //
        // build leaf
        //

        // const auto  lacc = acc( is0, is1, is2 );
        
        if ( ! tacc.is_exact() )
        {
            auto  Dc                = blas::copy( D );  // do not modify D (!)
            auto  [ G, X0, X1, X2 ] = blas::hosvd( Dc, tacc, apx );
            
            if ( G.data_byte_size() + X0.data_byte_size() + X1.data_byte_size() + X2.data_byte_size() < Dc.data_byte_size() )
            {
                return std::make_unique< tucker_tensor3< value_t > >( is0, is1, is2,
                                                                      std::move( G ),
                                                                      std::move( X0 ),
                                                                      std::move( X1 ),
                                                                      std::move( X2 ) );
            }// if
        }// if

        // either exact accuracy or lowrank is less efficient: build dense
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
        auto        sacc       = absolute_prec( Hpro::frobenius_norm, tacc.abs_eps() / 3.0 );

        ::tbb::parallel_for(
            ::tbb::blocked_range3d< uint >( 0, 2,
                                            0, 2,
                                            0, 2 ),
            [&,ntile] ( const auto &  r )
            {
                for ( auto  l = r.pages().begin(); l != r.pages().end(); ++l )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                        {
                            const auto  D_sub = D( sub_is0[i] - is0.first(),
                                                   sub_is1[j] - is1.first(),
                                                   sub_is2[l] - is2.first() );
                            
                            sub_D(i,j,l) = build_hierarchical_tucker( sub_is0[i], sub_is1[j], sub_is2[l], D_sub, ntile, sacc, apx, fpacc );
                            
                            HLR_ASSERT( sub_D(i,j,l).get() != nullptr );
                        }// for
                    }// for
                }// for
            } );

        //
        // look for type of sub blocks
        //

        bool    all_tucker = true;
        bool    all_dense  = true;
        size_t  mem_sub    = 0;
        
        for ( uint  l = 0; l < 2; ++l )
            for ( uint  j = 0; j < 2; ++j )
                for ( uint  i = 0; i < 2; ++i )
                {
                    if ( ! is_tucker_tensor3( *sub_D(i,j,l) ) )
                        all_tucker = false;
                    else
                        mem_sub += sub_D(i,j,l)->data_byte_size();

                    if ( ! is_dense_tensor3( *sub_D(i,j,l) ) )
                        all_dense = false;
                }// for
        
        //
        // if all sub blocks are Tucker tensors, try to merge
        //

        if ( all_tucker )
        {
            auto  T = std::unique_ptr< base_tensor3< value_t > >();
        
            // T = std::move( merge_all( is0, is1, is2, D, sub_D, acc, apx, hosvd ) );
            // T = std::move( merge_greedy( is0, is1, is2, D, sub_D, acc, apx, hosvd ) );
            // T = std::move( merge_dim( is0, is1, is2, D, sub_D, acc, apx, hosvd ) );

            //
            // construct from dense
            //

            // std::cout << "trying merge on " << is0.to_string() << " × " << is1.to_string() << " × " << is1.to_string() << std::endl;
            
            // const auto  lacc              = acc( is0, is1, is2 );
            auto        Dc                = blas::copy( D );  // do not modify D (!)
            auto        [ G, X0, X1, X2 ] = blas::hosvd( Dc, tacc, apx );
            
            if ( G.data_byte_size() + X0.data_byte_size() + X1.data_byte_size() + X2.data_byte_size() <= mem_sub )
            {
                // std::cout << "merged : " << G.data_byte_size() + X0.data_byte_size() + X1.data_byte_size() + X2.data_byte_size() << " / " << mem_sub << std::endl;
                
                return std::make_unique< tucker_tensor3< value_t > >( is0, is1, is2,
                                                                      std::move( G ),
                                                                      std::move( X0 ),
                                                                      std::move( X1 ),
                                                                      std::move( X2 ) );
            }// if
            
            // std::cout << "not merged : " << G.data_byte_size() + X0.data_byte_size() + X1.data_byte_size() + X2.data_byte_size() << " / " << mem_sub << std::endl;
        }// if
        
        //
        // also return full tensor if all sub blocks are dense
        //

        if ( all_dense )
            return std::make_unique< dense_tensor3< value_t > >( is0, is1, is2, std::move( blas::copy( D ) ) );

        //
        // as merge was not successful, construct structured tensor
        //

        auto  B = std::make_unique< structured_tensor3< value_t > >( is0, is1, is2 );

        B->set_structure( 2, 2, 2 );
        
        for ( uint  l = 0; l < 2; ++l )
            for ( uint  i = 0; i < 2; ++i )
                for ( uint  j = 0; j < 2; ++j )
                {
                    // compress FP representation
                    if ( ! fpacc.is_exact() )
                        sub_D(i,j,l)->compress( fpacc );
                    
                    B->set_block( i, j, l, sub_D(i,j,l).release() );
                }// for
            
        return B;
    }// else
}

}// namespace detail

//
// compression function with all user defined options
//
template < typename                    value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< base_tensor3< value_t > >
hierarchical_tucker ( const blas::tensor3< value_t > &  D,
                      const size_t                      ntile,
                      const accuracy &                  acc,
                      const approx_t &                  apx,
                      const accuracy &                  fpacc )
{
    //
    // handle parallel computation of norm because BLAS should be sequential
    //
    
    // const auto  norm_D = blas::norm_F( D );

    //
    // start compressing with per-block accuracy
    //
    
    // const auto  delta  = norm_D * rel_prec / D.size(0);
    // auto        acc_D  = Hpro::fixed_prec( delta );
    auto        M      = std::unique_ptr< base_tensor3< value_t > >();
    
    M = std::move( detail::build_hierarchical_tucker( indexset( 0, D.size(0)-1 ),
                                                      indexset( 0, D.size(1)-1 ),
                                                      indexset( 0, D.size(2)-1 ),
                                                      D, ntile, acc, apx, fpacc ) );

    // in case of global tucker/dense tensor
    if ( ! fpacc.is_exact() )
        M->compress( fpacc );

    HLR_ASSERT( M.get() != nullptr );

    return M;
}

namespace detail
{

//
// copy given tensor into D
//
template < typename value_t >
void
to_dense ( const base_tensor3< value_t > &  X,
           blas::tensor3< value_t > &       D )
{
    if ( is_structured( X ) )
    {
        auto  BX = cptrcast( &X, structured_tensor3< value_t > );

        ::tbb::parallel_for(
            ::tbb::blocked_range3d< uint >( 0, BX->nblocks(0),
                                            0, BX->nblocks(1),
                                            0, BX->nblocks(2) ),
            [BX,&D] ( const auto &  r )
            {
                for ( auto  l = r.pages().begin(); l != r.pages().end(); ++l )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                        {
                            if ( ! is_null( BX->block(i,j,l) ) )
                                to_dense( *BX->block(i,j,l), D );
                        }// for
                    }// for
                }// for
            }
        );
    }// if
    else if ( is_tucker_tensor3( X ) )
    {
        auto  TX    = cptrcast( &X, tucker_tensor3< value_t > );
        auto  T0    = blas::tensor_product( TX->G(), TX->X(0), 0 );
        auto  T1    = blas::tensor_product( T0,      TX->X(1), 1 );
        auto  DX    = blas::tensor_product( T1,      TX->X(2), 2 );
        auto  D_sub = D( X.is(0), X.is(1), X.is(2) );
        
        blas::copy( DX, D_sub );
    }// if
    else if ( is_dense_tensor3( X ) )
    {
        auto  DX    = cptrcast( &X, dense_tensor3< value_t > );
        auto  D_sub = D( X.is(0), X.is(1), X.is(2) );

        blas::copy( DX->tensor(), D_sub );
    }// if
    else
    {
        HLR_ERROR( "unknown tensor type" );
    }// if
}

}// namespace detail

//
// convert to dense tensor
//
template < typename value_t >
std::unique_ptr< dense_tensor3< value_t > >
to_dense ( const base_tensor3< value_t > &  X )
{
    // only zero offsets for now
    HLR_ASSERT( ( X.is(0).first() == 0 ) &&
                ( X.is(1).first() == 0 ) &&
                ( X.is(2).first() == 0 ) )
        
    if ( is_dense_tensor3( X ) )
    {
        auto  D = X.copy();

        return std::unique_ptr< dense_tensor3< value_t > >( ptrcast( D.release(), dense_tensor3< value_t > ) );
    }// if
    else
    {
        auto  DT = blas::tensor3< value_t >( X.dim(0), X.dim(1), X.dim(2) );
        
        detail::to_dense( X, DT );

        return std::make_unique< dense_tensor3< value_t > >( X.is(0), X.is(1), X.is(2), std::move( DT ) );
    }// if
}

//
// compress/uncompress tensor in-place and return memory of compressed version
//
template < typename value_t >
size_t
compress_tucker ( blas::tensor3< value_t > &  D,
                  const size_t                ntile,
                  const double &              eps,
                  const bool                  fpcompress )
{
    HLR_ASSERT(( D.size(0) == D.size(1) ) &&
               ( D.size(0) == D.size(2) )); // assuming equal size in all dimensions

    const auto  N    = D.size(0);
    const auto  Nc   = (N % ntile == 0 ? N / ntile : ( N / ntile ) + 1 );
    auto        apx  = approx::SVD< value_t >();
    auto        zmem = std::atomic< size_t >( 0 );

    const auto  norm_D = blas::norm_F( D );
    const auto  tol    = eps * norm_D / ( double(Nc) * std::sqrt( 3 )); // √3 = √d from HOSVD
    const auto  tacc   = accuracy( Hpro::frobenius_norm, eps, tol ); // absolute_prec( Hpro::frobenius_norm, tol );
    const auto  fpacc  = relative_prec( Hpro::frobenius_norm, eps );
    
    ::tbb::parallel_for(
        ::tbb::blocked_range3d< size_t >( 0, Nc,
                                          0, Nc,
                                          0, Nc ),
        [&,ntile] ( const auto &  r )
        {
            for ( auto  z = r.pages().begin(); z != r.pages().end(); ++z )
            {
                const auto  nz = z * ntile;
                
                for ( auto  y = r.cols().begin(); y != r.cols().end(); ++y )
                {
                    const auto  ny = y * ntile;
                
                    for ( auto  x = r.rows().begin(); x != r.rows().end(); ++x )
                    {
                        const auto  nx = x * ntile;

                        const auto  is0 = indexset( nx, std::min( nx + ntile - 1, N-1 ) );
                        const auto  is1 = indexset( ny, std::min( ny + ntile - 1, N-1 ) );
                        const auto  is2 = indexset( nz, std::min( nz + ntile - 1, N-1 ) );

                        auto        O_sub      = D( is0, is1, is2 );
                        auto        D_sub      = blas::copy( O_sub );
                        const auto  norm_D_sub = blas::norm_F( D_sub );

                        // O_sub.check_data();
                        // D_sub.check_data();
                        
                        if ( norm_D_sub == value_t(0) )
                            return;

                        // if ( verbose(4) )
                        //     std::cout << is0 << " × " << is1 << " × " << is2 << " : " << format_norm( norm_D_sub, acc.abs_eps() * norm_D_sub ) << std::endl;

                        auto  G  = blas::tensor3< value_t >();
                        auto  X0 = blas::matrix< value_t >();
                        auto  X1 = blas::matrix< value_t >();
                        auto  X2 = blas::matrix< value_t >();
                        
                        //
                        // HOSVD
                        //
                
                        // if (( cmdline::tapprox == "hosvd" ) || ( cmdline::tapprox == "default" ))
                        {
                            std::tie( G, X0, X1, X2 ) = blas::hosvd( D_sub, tacc, apx );
                        }// if

                        //
                        // decide about format
                        //

                        const auto  mem_full   = D_sub.data_byte_size();
                        const auto  mem_tucker = G.data_byte_size() + X0.data_byte_size() + X1.data_byte_size() + X2.data_byte_size();
                        
                        if ( mem_tucker < mem_full )
                        {
                            // if ( verbose(4) )
                            //     std::cout << is0 << " × " << is1 << " × " << is2 << " : ranks = "
                            //               << G.size(0) << " / " << G.size(1) << " / " << G.size(2) << " / " << G.size(3) << std::endl;
                         
                            //
                            // replace data by uncompressed version
                            //

                            auto  T_sub = std::make_unique< tensor::tucker_tensor3< value_t > >( is0, is1, is2,
                                                                                                 std::move( G ),
                                                                                                 std::move( X0 ),
                                                                                                 std::move( X1 ),
                                                                                                 std::move( X2 ) );

                            if ( fpcompress )
                                T_sub->compress( fpacc );
                         
                            auto  T0 = blas::tensor_product( T_sub->G(), T_sub->X(0), 0 );
                            auto  T1 = blas::tensor_product(         T0, T_sub->X(1), 1 );
                            auto  Y  = blas::tensor_product(         T1, T_sub->X(2), 2 );

                            blas::copy( Y, O_sub );

                            zmem += T_sub->data_byte_size();
                        }// if
                        else
                        {
                            //
                            // use original data (nothing to do)
                            //

                            auto  T_sub = std::make_unique< tensor::dense_tensor3< value_t > >( is0, is1, is2, std::move( D_sub ) );
                                
                            if ( fpcompress )
                                T_sub->compress( fpacc );

                            auto  Y = T_sub->tensor();

                            // Y.check_data();
                            
                            blas::copy( Y, O_sub );

                            zmem += T_sub->data_byte_size();
                        }// else

                        O_sub.check_data();
                    }// for
                }// for
            }// for
        } );

    return zmem.load();
}

template < typename value_t >
size_t
compress_tucker ( blas::tensor4< value_t > &  D,
                  const size_t                ntile,
                  const double &              eps,
                  const bool                  fpcompress )
{
    HLR_ASSERT(( D.size(0) == D.size(1) ) &&
               ( D.size(0) == D.size(2) )); // assuming equal size in all dimensions
    HLR_ASSERT( ( D.size(0) / ntile ) * ntile == D.size(0) );             // no padding for now

    const auto  N    = D.size(0);
    const auto  n3   = D.size(3);
    auto        apx  = approx::SVD< value_t >();
    auto        zmem = std::atomic< size_t >( 0 );

    const auto  norm_D = blas::norm_F( D );
    const auto  tol    = eps * norm_D / std::sqrt( 4 * double(N) / double(ntile) ); // √4 = √d from HOSVD
    const auto  tacc   = absolute_prec( Hpro::frobenius_norm, tol );
    // const auto  fpacc   = absolute_prec( Hpro::frobenius_norm, cmdline::eps );
    // const auto  acc    = relative_prec( Hpro::frobenius_norm, tol );
    const auto  fpacc   = relative_prec( Hpro::frobenius_norm, eps );
    
    ::tbb::parallel_for(
        ::tbb::blocked_range3d< size_t >( 0, N / ntile,
                                          0, N / ntile,
                                          0, N / ntile ),
        [&,n3,ntile] ( const auto &  r )
        {
            for ( auto  z = r.pages().begin(); z != r.pages().end(); ++z )
            {
                const auto  nz = z * ntile;
                
                for ( auto  y = r.cols().begin(); y != r.cols().end(); ++y )
                {
                    const auto  ny = y * ntile;
                
                    for ( auto  x = r.rows().begin(); x != r.rows().end(); ++x )
                    {
                        const auto  nx = x * ntile;

                        const auto  is0 = indexset( nx, std::min( nx + ntile - 1, N-1 ) );
                        const auto  is1 = indexset( ny, std::min( ny + ntile - 1, N-1 ) );
                        const auto  is2 = indexset( nz, std::min( nz + ntile - 1, N-1 ) );
                        const auto  is3 = indexset( 0, n3 - 1 );

                        auto        O_sub      = D( is0, is1, is2, is3 );
                        auto        D_sub      = blas::copy( O_sub );
                        const auto  norm_D_sub = blas::norm_F( D_sub );

                        // O_sub.check_data();
                        // D_sub.check_data();
                        
                        if ( norm_D_sub == value_t(0) )
                            return;

                        // if ( verbose(4) )
                        //     std::cout << is0 << " × " << is1 << " × " << is2 << " : " << format_norm( norm_D_sub, acc.abs_eps() * norm_D_sub ) << std::endl;

                        auto  G  = blas::tensor4< value_t >();
                        auto  X0 = blas::matrix< value_t >();
                        auto  X1 = blas::matrix< value_t >();
                        auto  X2 = blas::matrix< value_t >();
                        auto  X3 = blas::matrix< value_t >();
                        
                        //
                        // HOSVD
                        //
                
                        // if (( cmdline::tapprox == "hosvd" ) || ( cmdline::tapprox == "default" ))
                        {
                            std::tie( G, X0, X1, X2, X3 ) = blas::hosvd( D_sub, tacc, apx );
                        }// if

                        //
                        // decide about format
                        //

                        const auto  mem_full   = D_sub.data_byte_size();
                        const auto  mem_tucker = G.data_byte_size() + X0.data_byte_size() + X1.data_byte_size() + X2.data_byte_size() + X3.data_byte_size();
                        
                        if ( mem_tucker < mem_full )
                        {
                            // if ( verbose(4) )
                            //     std::cout << is0 << " × " << is1 << " × " << is2 << " : ranks = "
                            //               << G.size(0) << " / " << G.size(1) << " / " << G.size(2) << " / " << G.size(3) << std::endl;
                            
                            //
                            // replace data by uncompressed version
                            //

                            auto  T_sub = std::make_unique< tensor::tucker_tensor4< value_t > >( is0, is1, is2, is3,
                                                                                                 std::move( G ),
                                                                                                 std::move( X0 ),
                                                                                                 std::move( X1 ),
                                                                                                 std::move( X2 ),
                                                                                                 std::move( X3 ) );

                            if ( fpcompress )
                                T_sub->compress( fpacc );
                            
                            auto  T0 = blas::tensor_product( T_sub->G(), T_sub->X(0), 0 );
                            auto  T1 = blas::tensor_product(         T0, T_sub->X(1), 1 );
                            auto  T2 = blas::tensor_product(         T1, T_sub->X(2), 2 );
                            auto  Y  = blas::tensor_product(         T2, T_sub->X(3), 3 );

                            blas::copy( Y, O_sub );

                            zmem += T_sub->data_byte_size();
                        }// if
                        else
                        {
                            //
                            // use original data (nothing to do)
                            //

                            auto  T_sub = std::make_unique< tensor::dense_tensor4< value_t > >( is0, is1, is2, is3, std::move( D_sub ) );
                                
                            if ( fpcompress )
                                T_sub->compress( fpacc );

                            auto  Y = T_sub->tensor();

                            // Y.check_data();
                            
                            blas::copy( Y, O_sub );

                            zmem += T_sub->data_byte_size();
                        }// else

                        // O_sub.check_data();
                    }// for
                }// for
            }// for
        } );

    return zmem.load();
}

//
// construct blockwise compression representation of given dense tensor
//
template < typename                    value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< tensor::structured_tensor3< value_t > >
blockwise_tucker ( blas::tensor3< value_t > &  D,
                   const size_t                ntile,
                   const accuracy &            tacc,
                   const approx_t &            apx,
                   const accuracy &            fpacc )
{
    using  real_t = real_type_t< value_t >;
    
    const auto    N = D.size(0); // assuming equal size in all dimensions

    HLR_ASSERT( ( N / ntile ) * ntile == N ); // no padding for now

    auto  T = std::make_unique< tensor::structured_tensor3< value_t > >( indexset( 0, N-1 ), indexset( 0, N-1 ), indexset( 0, N-1 ) );

    T->set_structure( N / ntile,
                      N / ntile,
                      N / ntile );
    
    // for ( size_t  nz = 0; nz < N; nz += ntile )
    // {
    //     for ( size_t  ny = 0; ny < N; ny += ntile )
    //     {
    //         for ( size_t  nx = 0; nx < N; nx += ntile )
    //         {
    ::tbb::parallel_for(
        ::tbb::blocked_range3d< size_t >( 0, N / ntile,
                                          0, N / ntile,
                                          0, N / ntile ),
        [&,ntile] ( const auto &  r )
        {
            for ( auto  z = r.pages().begin(); z != r.pages().end(); ++z )
            {
                const auto  nz = z * ntile;
                
                for ( auto  y = r.cols().begin(); y != r.cols().end(); ++y )
                {
                    const auto  ny = y * ntile;
                
                    for ( auto  x = r.rows().begin(); x != r.rows().end(); ++x )
                    {
                        const auto  nx = x * ntile;

                        const auto  is0 = indexset( nx, std::min( nx + ntile - 1, N-1 ) );
                        const auto  is1 = indexset( ny, std::min( ny + ntile - 1, N-1 ) );
                        const auto  is2 = indexset( nz, std::min( nz + ntile - 1, N-1 ) );

                        auto        D_ijk      = blas::copy( D( is0, is1, is2 ) );
                        const auto  norm_D_ijk = blas::norm_F( D_ijk );

                        // std::cout << is0 << " × " << is1 << " × " << is2 << " : " << format_norm( norm_D_ijk, acc.abs_eps() * norm_D_ijk ) << std::endl;

                        if ( norm_D_ijk == value_t(0) )
                            return;

                        auto  G  = blas::tensor3< value_t >();
                        auto  X0 = blas::matrix< value_t >();
                        auto  X1 = blas::matrix< value_t >();
                        auto  X2 = blas::matrix< value_t >();
                        
                        //
                        // HOSVD
                        //
                
                        // if (( tapprox == "hosvd" ) || ( tapprox == "default" ))
                        {
                            auto  S0 = blas::vector< real_t >();
                            auto  S1 = blas::vector< real_t >();
                            auto  S2 = blas::vector< real_t >();
                            
                            std::tie( G, X0, S0, X1, S1, X2, S2 ) = blas::hosvd_sv( D_ijk, tacc, apx );
                            
                            // if ( verbose(2) && ( std::min( std::min( G.size(0), G.size(1) ), G.size(2) ) ) > 0 )
                                // std::cout << is0 << " × " << is1 << " × " << is2 << " : "
                                //           << boost::format( "%.4e" ) % std::max( std::max( S0(0), S1(0) ), S2(0) ) << std::endl;

                            // std::tie( G, X0, X1, X2 ) = blas::hosvd( D_ijk, tacc, apx );
                        }// if

                        //
                        // ST-HOSVD
                        //

                        // if ( tapprox == "sthosvd" )
                        // {
                        //     std::tie( G, X0, X1, X2 ) = blas::sthosvd( D_ijk, tacc, apx );
                        // }// if

                        //
                        // Greedy-HOSVD
                        //

                        // if ( tapprox == "ghosvd" )
                        // {
                        //     HLR_ERROR( "TODO" );
                        //     // std::tie( G, X0, X1, X2 ) = impl::blas::greedy_hosvd( D_ijk, tacc, apx );
                        // }// if

                        // std::cout << is0 << " × " << is1 << " × " << is2 << " : " << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << std::endl;
                        
                        //
                        // decide about format
                        //

                        const auto  mem_full   = D_ijk.data_byte_size();
                        const auto  mem_tucker = G.data_byte_size() + X0.data_byte_size() + X1.data_byte_size() + X2.data_byte_size();
                        
                        if ( mem_tucker < mem_full )
                        {
                            auto  T_sub = std::make_unique< tensor::tucker_tensor3< value_t > >( is0, is1, is2,
                                                                                                 std::move( G ),
                                                                                                 std::move( X0 ),
                                                                                                 std::move( X1 ),
                                                                                                 std::move( X2 ) );

                            if ( ! fpacc.is_exact() )
                                T_sub->compress( fpacc );
                    
                            T->set_block( x, y, z, T_sub.release() );
                        }// if
                        else
                        {
                            auto  T_sub = std::make_unique< tensor::dense_tensor3< value_t > >( is0, is1, is2, std::move( D_ijk ) );
                    
                            if ( ! fpacc.is_exact() )
                                T_sub->compress( fpacc );
                    
                            T->set_block( x, y, z, T_sub.release() );
                        }// else

                        // std::cout << is0 << " × " << is1 << " × " << is2 << " : "
                        //           << "mem = " << T->block(x,y,z)->data_byte_size() << " / "
                        //           << boost::format( "%.1fx" ) % ( double(mem_full) / double(T->block(x,y,z)->data_byte_size()) ) << std::endl;
                    }// for
                }// for
            }// for
        } );

    return  T;
}

}}}// namespace hlr::tbb::tensor

#endif // __HLR_TBB_TENSOR_HH

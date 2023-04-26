#ifndef __HLR_TBB_TENSOR_HH
#define __HLR_TBB_TENSOR_HH
//
// Project     : HLR
// Module      : tbb/tensor
// Description : tensor algorithms using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/blocked_range3d.h>
#include <tbb/parallel_for.h>

#include <hlr/tensor/construct.hh>
#include <hlr/tensor/convert.hh>

namespace hlr { namespace tbb { namespace tensor {

using namespace hlr::tensor;

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
                            const tensor_accuracy &           acc,
                            const approx_t &                  apx,
                            const size_t                      ntile )
{
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
            auto  [ G, X0, X1, X2 ] = blas::greedy_hosvd( Dc, lacc, apx );
            
            if ( G.byte_size() + X0.byte_size() + X1.byte_size() + X2.byte_size() < Dc.byte_size() )
            {
                return std::make_unique< tucker_tensor3< value_t > >( is0, is1, is2,
                                                                      std::move( G ),
                                                                      std::move( X0 ),
                                                                      std::move( X1 ),
                                                                      std::move( X2 ) );
            }// if
        }// if

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
                            
                            sub_D(i,j,l) = build_hierarchical_tucker( sub_is0[i], sub_is1[j], sub_is2[l], D_sub, acc, apx, ntile );
                            
                            HLR_ASSERT( sub_D(i,j,l).get() != nullptr );
                        }// for
                    }// for
                }// for
            } );

        //
        // look for type of sub blocks
        //

        bool  all_tucker = true;
        bool  all_dense  = true;
        
        for ( uint  l = 0; l < 2; ++l )
            for ( uint  j = 0; j < 2; ++j )
                for ( uint  i = 0; i < 2; ++i )
                {
                    if ( ! is_tucker( *sub_D(i,j,l) ) )
                        all_tucker = false;

                    if ( ! is_dense( *sub_D(i,j,l) ) )
                        all_dense = false;
                }// for
        
        //
        // if all sub blocks are Tucker tensors, try to merge
        //

        if ( all_tucker )
        {
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
        
            if ( std::min({ rank[0], rank[1], rank[2] }) >= std::min({ D.size(0), D.size(1), D.size(2) }) )
            {
                //
                // directly use HOSVD on D as merged ranks are too large
                //
                
                auto        Dc   = blas::copy( D );  // do not modify D (!)
                const auto  lacc = acc( is0, is1, is2 );

                std::tie( G3, Y0, Y1, Y2 ) = std::move( hosvd( Dc, lacc, apx ) );
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

                //
                // compress with respect to local accuracy
                //
            
                const auto  lacc = acc( is0, is1, is2 );

                std::tie( G3, Y0, Y1, Y2 ) = std::move( blas::recompress( G2, X0, X1, X2, lacc, apx, hosvd ) );
            }// if
            
            //
            // return coarse tucker tensor if more memory efficient
            //

            const auto  mem_full   = sizeof(value_t) * ( D.size(0) * D.size(1) * D.size(2) );
            const auto  mem_coarse = sizeof(value_t) * ( G3.size(0) * G3.size(1) * G3.size(2) +
                                                         Y0.nrows() * Y0.ncols() +
                                                         Y1.nrows() * Y1.ncols() +
                                                         Y2.nrows() * Y2.ncols() );

            if ( mem_coarse < std::min( mem_sub, mem_full ) )
            {
                return std::make_unique< tucker_tensor3< value_t > >( is0, is1, is2,
                                                                      std::move( G3 ),
                                                                      std::move( Y0 ),
                                                                      std::move( Y1 ),
                                                                      std::move( Y2 ) );
            }// if
            
            if ( mem_full < mem_sub )
            {
                return std::make_unique< dense_tensor3< value_t > >( is0, is1, is2, std::move( blas::copy( D ) ) );
            }// if
        }// if

        //
        // also return full tensor if all sub blocks are dense
        //

        if ( all_dense )
        {
            return std::make_unique< dense_tensor3< value_t > >( is0, is1, is2, std::move( blas::copy( D ) ) );
        }// if
        
        //
        // as merge was not successful, construct structured tensor
        //

        auto  B = std::make_unique< structured_tensor3< value_t > >( is0, is1, is2 );

        B->set_structure( 2, 2, 2 );
        
        for ( uint  l = 0; l < 2; ++l )
            for ( uint  i = 0; i < 2; ++i )
                for ( uint  j = 0; j < 2; ++j )
                    B->set_block( i, j, l, sub_D(i,j,l).release() );
            
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
build_hierarchical_tucker ( const blas::tensor3< value_t > &  D,
                            const tensor_accuracy &           acc,
                            const approx_t &                  apx,
                            const size_t                      ntile )
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
                                                      D, acc, apx, ntile ) );

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
    else if ( is_tucker( X ) )
    {
        auto  TX    = cptrcast( &X, tucker_tensor3< value_t > );
        auto  T0    = blas::tensor_product( TX->G_decompressed(), TX->X_decompressed(0), 0 );
        auto  T1    = blas::tensor_product( T0,                   TX->X_decompressed(1), 1 );
        auto  DX    = blas::tensor_product( T1,                   TX->X_decompressed(2), 2 );
        auto  D_sub = D( X.is(0), X.is(1), X.is(2) );
        
        blas::copy( DX, D_sub );
    }// if
    else if ( is_dense( X ) )
    {
        auto  DX    = cptrcast( &X, dense_tensor3< value_t > );
        auto  D_sub = D( X.is(0), X.is(1), X.is(2) );

        blas::copy( DX->tensor_decompressed(), D_sub );
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
        
    if ( is_dense( X ) )
    {
        auto  D = X.copy();

        return std::unique_ptr< dense_tensor3< value_t > >( ptrcast( D.release(), dense_tensor3< value_t > ) );
    }// if
    else
    {
        auto  D = std::make_unique< dense_tensor3< value_t > >( X.is(0), X.is(1), X.is(2) );
        
        detail::to_dense( X, D->tensor() );

        return D;
    }// if
}

}}}// namespace hlr::tbb::tensor

#endif // __HLR_TBB_TENSOR_HH

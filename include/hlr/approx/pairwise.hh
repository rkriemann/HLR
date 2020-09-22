#ifndef __HLR_APPROX_PAIRWISE_HH
#define __HLR_APPROX_PAIRWISE_HH
//
// Project     : HLib
// Module      : approx/pairwise
// Description : pair wise approximation handling
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <list>

#include <hlr/arith/blas.hh>

namespace hlr { namespace approx {

namespace hpro = HLIB;

//
// compute low-rank approximation of a sum Σ_i U_i V_i^H
// using pairwise approximation
//
//     (U_0 V_0' + U_1 V_1') + (U_2 V_2' + U_3 V_3') + ...
//                ↘                   ↙
//                U_01 V_01' + U_23 V_23'
//                           ↓
//                     U_0..3 V_0..3'
//
template < typename value_t,
           typename approx_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
pairwise ( const std::list< blas::matrix< value_t > > &  U,
           const std::list< blas::matrix< value_t > > &  V,
           const hpro::TTruncAcc &                       acc,
           const approx_t &                              approx )
{
    //
    // split set into two halves and recurse
    //

    if ( U.size() > 2 )
    {
        size_t                                nfirst = U.size() / 2;
        std::list< blas::matrix< value_t > >  U1, U2, V1, V2;
        auto                                  U_i = U.cbegin();
        auto                                  V_i = V.cbegin();

        do
        {
            if ( nfirst > 0 )
            {
                U1.push_back( *U_i );
                V1.push_back( *V_i );
                --nfirst;
            }// if
            else
            {
                U2.push_back( *U_i );
                V2.push_back( *V_i );
            }// else

            ++U_i;
            ++V_i;
        } while ( U_i != U.cend() );

        auto  [ U1_apx, V1_apx ] = pairwise( U1, V1, acc, approx );
        auto  [ U2_apx, V2_apx ] = pairwise( U2, V2, acc, approx );

        return approx( { U1_apx, U2_apx }, { V1_apx, V2_apx }, acc );
    }// if
    else
    {
        return approx( U, V, acc );
    }// else
}

}}// namespace hlr::approx

#endif // __HLR_APPROX_SVD_HH

#ifndef __HLR_BEM_TRAITS_HH
#define __HLR_BEM_TRAITS_HH
//
// Project     : HLR
// Module      : bem/traits
// Description : type traits for BEM based classes
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/cluster/TBlockCluster.hh>

#include <hlr/utils/traits.hh>
#include <hlr/approx/accuracy.hh>

namespace hlr
{

//
// concept for lowrank approximation types in H context
//
template < typename  T >
concept lowrank_approx_type = requires
{
    requires has_value_type< T >;

    {
        requires ( const T &                    approx,
                   const Hpro::TBlockCluster *  bc,
                   const accuracy &             acc )
        {
            approx.build( & bc, acc );
        }
    };
    
    {
        requires ( const T &                     approx,
                   const Hpro::TBlockIndexSet &  bis,
                   const accuracy &              acc )
        {
            approx.build( & bis, acc );
        }
    };
};

//
// concept for coefficient functions
//
template < typename  T >
concept coefficient_function_type = requires
{
    requires has_value_type< T >;

    {
        requires ( const T &                  coeff,
                   const Hpro::TIndexSet &    rowis,
                   const Hpro::TIndexSet &    colis,
                   Hpro::value_type_t< T > *  matrix )
        {
            coeff.eval( rowis, colis, matrix );
        }
    };
    
    {
        requires ( const T &                           coeff,
                   const std::vector< Hpro::idx_t > &  tau,
                   const std::vector< Hpro::idx_t > &  sigma,
                   Hpro::value_type_t< T > *           matrix )
        {
            coeff.eval( tau, sigma, matrix );
        }
    };
};

}// namespace hlr

#endif // __HLR_BEM_TRAITS_HH

#ifndef __MATRIXBUILD_HH
#define __MATRIXBUILD_HH
//
// Project     : HLib
// File        : matrixbuild.hh
// Description : matrix construction implementation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlib.hh>

namespace SEQ
{

//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< HLIB::TMatrix >
build_matrix ( const HLIB::TBlockCluster *  bct,
               const coeff_t &              coeff,
               const lrapx_t &              lrapx,
               const HLIB::TTruncAcc &      acc );

}// namespace SEQ

namespace TBB
{

//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< HLIB::TMatrix >
build_matrix ( const HLIB::TBlockCluster *  bct,
               const coeff_t &              coeff,
               const lrapx_t &              lrapx,
               const HLIB::TTruncAcc &      acc );

}// namespace TBB

namespace HPX
{

//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< HLIB::TMatrix >
build_matrix ( const HLIB::TBlockCluster *  bct,
               const coeff_t &              coeff,
               const lrapx_t &              lrapx,
               const HLIB::TTruncAcc &      acc );

}// namespace HPX

//
// include implementation
//
#include "matrixbuild.cc"

#endif // __MATRIXBUILD_HH

#ifndef __HLR_TBB_BLAS_HH
#define __HLR_TBB_BLAS_HH
//
// Project     : HLR
// Module      : tbb/blas
// Description : optimized BLAS algorithms using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <functional>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range3d.h>

#include <hlr/arith/blas.hh>

namespace hlr { namespace tbb { namespace blas {

using namespace hlr::blas;

//
// dot product
//
template < tensor_type  tensor1_t,
           tensor_type  tensor2_t >
requires ( std::same_as< typename tensor1_t::value_t,
                         typename tensor2_t::value_t > )
typename tensor1_t::value_t
dot ( const tensor1_t &  t1,
      const tensor2_t &  t2 )
{
    using value_t = typename tensor1_t::value_t;
    
    HLR_DBG_ASSERT( ( t1.size(0) == t2.size(0) ) &&
                    ( t1.size(1) == t2.size(1) ) &&
                    ( t1.size(2) == t2.size(2) ) );

    auto  d = value_t(0);

    ::tbb::parallel_reduce(
        ::tbb::blocked_range3d< size_t >( 0, t1.size(0),
                                          0, t1.size(1),
                                          0, t1.size(2) ),
        [&t1,&t2] ( const auto &  r,
                    const auto    val )
        {
            auto  s = val;
            
            for ( size_t  l = r.pages().begin(); l < r.pages().end(); l++ )
                for ( size_t  j = r.cols().begin(); j < r.cols().end(); j++ )
                    for ( size_t  i = r.rows().begin(); i < r.rows().end(); i++ )
                        s += t1(i,j,l) * t2(i,j,l);

            return s;
        },
        std::plus< value_t >()
    );

    return d;
}

//
// Frobenius norm
//
template < tensor_type  tensor_t >
real_type_t< typename tensor_t::value_t >
norm_F ( const tensor_t &  t )
{
    return std::sqrt( std::abs( dot( t, t ) ) );
}

//
// compute B := α A + β B (element wise)
//
template < typename     alpha_t,
           tensor_type  tensorA_t,
           tensor_type  tensorB_t >
requires ( std::same_as< typename tensorA_t::value_t,
                         typename tensorB_t::value_t > &&
           std::convertible_to< alpha_t, typename tensorA_t::value_t > )
void
add ( const alpha_t      alpha,
      const tensorA_t &  A,
      tensorA_t &        B )
{
    HLR_DBG_ASSERT( ( A.size(0) == B.size(0) ) &&
                    ( A.size(1) == B.size(1) ) &&
                    ( A.size(2) == B.size(2) ) );
    
    ::tbb::parallel_for(
        ::tbb::blocked_range3d< size_t >( 0, A.size(0),
                                          0, A.size(1),
                                          0, A.size(2) ),
        [alpha,&A,&B] ( const auto &  r )
        {
            for ( size_t  l = r.pages().begin(); l < r.pages().end(); l++ )
                for ( size_t  j = r.cols().begin(); j < r.cols().end(); j++ )
                    for ( size_t  i = r.rows().begin(); i < r.rows().end(); i++ )
                        B(i,j,l) += value_t(alpha) * A(i,j,l);
        }
    );
}

}}}// namespace hlr::tbb::blas

#endif // __HLR_TBB_BLAS_HH

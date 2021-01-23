#ifndef __HLR_TF_ARITH_ACCU_HH
#define __HLR_TF_ARITH_ACCU_HH
//
// Project     : HLib
// File        : arith.hh
// Description : arithmetic functions using accumulators implemented with TF
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <taskflow/taskflow.hpp>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/tf/accumulator.hh"
#include "hlr/utils/checks.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/matrix/restrict.hh"

namespace hlr { namespace tf { namespace accu {

namespace hpro = HLIB;

using hlr::tf::matrix::accumulator;

namespace detail
{

//
// compute C = C + α op( A ) op( B ) where A and B are provided as accumulated updates
//
template < typename value_t,
           typename approx_t >
void
multiply ( ::tf::SubflowBuilder &   tf,
           const value_t            alpha,
           hpro::TMatrix &          C,
           accumulator &            accu,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    //
    // first handle all computable updates to C, including if C is non-blocked
    //

    accu.eval( tf, alpha, C, acc, approx );
    
    //
    // now handle recursive updates
    //
    
    if ( is_blocked( C ) )
    {
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );

        //
        // first, split update U into subblock updates
        // (to release U before recursion and by that avoid
        //  memory consumption dependent on hierarchy depth)
        //

        auto  sub_accu = accu.restrict( *BC );

        accu.clear_matrix();

        //
        // now apply recursive multiplications, e.g.,
        // collect all sub-products and recurse
        //
        // TODO: test if creation of sub-accumulators benefits from affinity_partitioner
        //
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                tf.emplace(
                    [=,&sub_accu,&acc,&approx] ( auto &  sf )
                    {
                        multiply< value_t >( sf, alpha, *BC->block(i,j), sub_accu(i,j), acc, approx );
                    } );
            }// for
    }// if
    else 
    {
        // apply accumulated updates
        accu.apply( alpha, C, acc, approx );
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    std::unique_ptr< hpro::TMatrix >  U;

    accumulator::update_list   upd{ { op_A, &A, op_B, &B } };
    accumulator                accu( std::move( U ), std::move( upd ) );

    ::tf::Taskflow  tf;
    
    tf.emplace( [=,&A,&B,&C,&acc,&approx,&accu] ( auto &  sf ) { detail::multiply< value_t >( sf, alpha, C, accu, acc, approx ); } );

    ::tf::Executor  executor;
    
    executor.run( tf ).wait();
}

}}}// namespace hlr::tf::accu

#endif // __HLR_TF_ARITH_ACCU_HH

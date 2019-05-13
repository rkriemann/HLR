#ifndef __HLR_TF_MATRIX_HH
#define __HLR_TF_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <type_traits>

#include <taskflow/taskflow.hpp>

#include <matrix/TMatrix.hh>
#include <matrix/TBlockMatrix.hh>
#include <base/TTruncAcc.hh>

#include "utils/tensor.hh"
#include "utils/log.hh"
#include "seq/matrix.hh"

namespace HLR
{

namespace Matrix
{
    
namespace TF
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
build ( const HLIB::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const HLIB::TTruncAcc &      acc )
{
    static_assert( std::is_same< typename coeff_t::value_t,
                   typename lrapx_t::value_t >::value,
                   "coefficient function and low-rank approximation must have equal value type" );

    assert( bct != nullptr );

    using  value_t = typename coeff_t::value_t;
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< TMatrix >  M;
    const auto                  rowis = bct->is().row_is();
    const auto                  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return Matrix::Seq::build( bct, coeff, lrapx, acc );
        
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M.reset( lrapx.build( bct, acc ) );
        }// if
        else
        {
            M = coeff.build( rowis, colis );
        }// else
    }// if
    else
    {
        auto  B = std::make_unique< TBlockMatrix >( bct );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        auto          nbr = B->nblock_rows();
        auto          nbc = B->nblock_cols();
        tf::Taskflow  tf;

        for ( uint  i = 0; i < nbr; ++i )
        {
            for ( uint  j = 0; j < nbc; ++j )
            {
                if ( bct->son( i, j ) != nullptr )
                {
                    tf.silent_emplace(
                        [bct,i,j,&coeff,&lrapx,&acc,&B] ()
                        {
                            auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc );
                            
                            B->set_block( i, j, B_ij.release() );
                        } );
                }// if
            }// for
        }// for

        tf.wait_for_all();

        M = std::move( B );
    }// else

    // copy properties from the cluster
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    log( 3, HLIB::to_string( "%d", M->id() ) );
    
    return M;
}

}// namespace TF

}// namespace Matrix

}// namespace HLR

#endif // __HLR_TF_MATRIX_HH

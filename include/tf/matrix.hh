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
build_helper ( tf::SubflowBuilder &         tf,
               const HLIB::TBlockCluster *  bct,
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
        auto  nbr = B->nblock_rows();
        auto  nbc = B->nblock_cols();

        for ( uint  i = 0; i < nbr; ++i )
        {
            for ( uint  j = 0; j < nbc; ++j )
            {
                if ( bct->son( i, j ) != nullptr )
                {
                    tf.silent_emplace(
                        [bct,i,j,&coeff,&lrapx,&acc,&B] ( auto &  sf )
                        {
                            auto  B_ij = build_helper( sf, bct->son( i, j ), coeff, lrapx, acc );
                            
                            B->set_block( i, j, B_ij.release() );
                        } );
                }// if
            }// for
        }// for

        M = std::move( B );
    }// else

    // copy properties from the cluster
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );
    
    return M;
}

template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< HLIB::TMatrix >
build ( const HLIB::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const HLIB::TTruncAcc &      acc )
{
    tf::Taskflow                tf;
    std::unique_ptr< TMatrix >  res;
    
    tf.silent_emplace( [&,bct] ( auto &  sf ) { res = build_helper( sf, bct, coeff, lrapx, acc ); } );
    tf.wait_for_all();

    return res;
}

//
// return copy of matrix
// - copy operation is performed in parallel for sub blocks
//
std::unique_ptr< TMatrix >
copy_helper ( tf::SubflowBuilder &  tf,
              const TMatrix &       M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, TBlockMatrix );
        auto  N  = std::make_unique< TBlockMatrix >();
        auto  B  = ptrcast( N.get(), TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    tf.silent_emplace(
                        [B,BM,i,j] ( auto &  sf )
                        {
                            auto  B_ij = copy_helper( sf, * BM->block( i, j ) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        } );
                }// if
            }// for
        }// for

        // tf.wait_for_all();
        
        return N;
    }// if
    else
    {
        // assuming non-structured block and hence no parallel copy needed
        return M.copy();
    }// else
}

std::unique_ptr< TMatrix >
copy ( const TMatrix &  M )
{
    tf::Taskflow                tf;
    std::unique_ptr< TMatrix >  res;
    
    tf.silent_emplace( [&M,&res] ( auto &  sf ) { res = copy_helper( sf, M ); } );
    tf.wait_for_all();

    return res;
}

}// namespace TF

}// namespace Matrix

}// namespace HLR

#endif // __HLR_TF_MATRIX_HH

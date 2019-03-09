#ifndef __HLR_GASPI_MATRIX_HH
#define __HLR_GASPI_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <vector>
#include <list>
#include <unordered_map>
#include <type_traits>

#include <matrix/TMatrix.hh>
#include <matrix/TBlockMatrix.hh>
#include <matrix/TGhostMatrix.hh>
#include <base/TTruncAcc.hh>

#include "utils/tools.hh"
#include "seq/matrix.hh"
#include "tbb/matrix.hh"
#include "gaspi/gaspi.hh"

namespace HLR
{

namespace Matrix
{
    
namespace GASPI
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

    using  value_t = typename coeff_t::value_t;
    
    assert( bct != nullptr );

    //
    // decide upon cluster type, how to construct matrix
    //

    HLR::GASPI::process         proc;
    const auto                  pid   = proc.rank();
    
    std::unique_ptr< TMatrix >  M;
    const auto                  rowis = bct->is().row_is();
    const auto                  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return Matrix::Seq::build( bct, coeff, lrapx, acc );

    if ( ! bct->procs().contains( pid ) )
    {
        M = std::make_unique< TGhostMatrix >( bct->is(), bct->procs(), value_type< value_t >::value );
    }// if
    else if ( bct->is_leaf() )
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
        tbb::blocked_range2d< uint >  r( 0, B->nblock_rows(),
                                         0, B->nblock_cols() );
        
        tbb::parallel_for( r,
            [&,bct] ( const tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( bct->son( i, j ) != nullptr )
                        {
                            auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc );
                            
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );

        M = std::move( B );
    }// else

    // copy properties from the cluster
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

}// namespace GASPI

}// namespace Matrix

}// namespace HLR

#endif // __HLR_MPI_MATRIX_HH

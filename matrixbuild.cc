//
// Project     : HLib
// File        : matrixbuild.cc
// Description : matrix construction implementation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <type_traits>

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
    
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M.reset( lrapx.build( bct, acc ) );
        }// if
        else
        {
            M = coeff.build( bct->is().row_is(), bct->is().col_is() );
        }// else
    }// if
    else
    {
        auto  B = std::make_unique< TBlockMatrix >( bct );

        B->set_id( bct->id() );
        B->set_procs( bct->procs() );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( bct->son( i, j ) != nullptr )
                {
                    auto  B_ij = build_matrix( bct->son( i, j ), coeff, lrapx, acc );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        M = std::move( B );
    }// else

    return M;
}

}// namespace SEQ


#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

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
    if ( std::max( rowis.size(), colis.size() ) <= 1000 )
        return SEQ::build_matrix( bct, coeff, lrapx, acc );
        
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
        tbb::parallel_for(
            tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                          0, B->nblock_cols() ),
            [&,bct] ( const tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( bct->son( i, j ) != nullptr )
                        {
                            auto  B_ij = build_matrix( bct->son( i, j ), coeff, lrapx, acc );
                            
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

}// namespace TBB

#include <hpx/parallel/task_block.hpp>

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
    if ( std::max( rowis.size(), colis.size() ) <= 1000 )
        return SEQ::build_matrix( bct, coeff, lrapx, acc );
        
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
        hpx::parallel::v2::define_task_block(
            [&,bct] ( auto &  tb )
            {
                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < B->nblock_rows(); ++j )
                    {
                        if ( bct->son( i, j ) != nullptr )
                        {
                            std::unique_ptr< TMatrix >  B_ij;

                            tb.run( [&,bct,i,j] { build_matrix( bct->son( i, j ), coeff, lrapx, acc ); } );
                            
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

}// namespace HPX


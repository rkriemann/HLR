#ifndef __HLR_TF_MATRIX_HH
#define __HLR_TF_MATRIX_HH
//
// Project     : HLR
// Module      : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cassert>
#include <type_traits>

#include <taskflow/taskflow.hpp>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/base/TTruncAcc.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/seq/matrix.hh"

#include "hlr/tf/detail/matrix.hh"

namespace hlr { namespace tf { namespace matrix {

namespace hpro = HLIB;

//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
namespace detail
{

template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< hpro::TMatrix >
build_helper ( ::tf::Subflow &              tf,
               const hpro::TBlockCluster *  bct,
               const coeff_t &              coeff,
               const lrapx_t &              lrapx,
               const hpro::TTruncAcc &      acc,
               const size_t                 nseq )
{
    static_assert( std::is_same< typename coeff_t::value_t,
                   typename lrapx_t::value_t >::value,
                   "coefficient function and low-rank approximation must have equal value type" );

    HLR_ASSERT( bct != nullptr );

    //
    // decide upon cluster type, how to construct matrix
    //

    auto        M     = std::unique_ptr< hpro::TMatrix >();
    const auto  rowis = bct->is().row_is();
    const auto  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return seq::matrix::build( bct, coeff, lrapx, acc );
        
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< hpro::TMatrix >( lrapx.build( bct, acc ) );
        }// if
        else
        {
            M = coeff.build( rowis, colis );
        }// else
    }// if
    else if ( std::min( rowis.size(), colis.size() ) <= nseq )
    {
        M = hlr::seq::matrix::build( bct, coeff, lrapx, acc );
    }// if
    else
    {
        M = std::make_unique< hpro::TBlockMatrix >( bct );
        
        auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

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
                    tf.emplace( [=,&coeff,&lrapx,&acc] ( ::tf::Subflow &  sf )
                    {
                        auto  B_ij = build_helper( sf, bct->son( i, j ), coeff, lrapx, acc, nseq );
                        
                        B->set_block( i, j, B_ij.release() );
                    } );
                }// if
            }// for
        }// for

        // make value type consistent in block matrix and sub blocks
        B->adjust_value_type();
    }// else

    // copy properties from the cluster
    // M->set_cluster_force( bct );
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );
    
    return M;
}

}// namespace detail

template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< hpro::TMatrix >
build ( const hpro::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const hpro::TTruncAcc &      acc,
        const size_t                 nseq = hpro::CFG::Arith::max_seq_size )
{
    ::tf::Taskflow                    tf;
    std::unique_ptr< hpro::TMatrix >  res;
    
    tf.emplace( [&,bct,nseq] ( ::tf::Subflow &  sf ) { res = detail::build_helper( sf, bct, coeff, lrapx, acc, nseq ); } );

    ::tf::Executor  executor;
    
    executor.run( tf ).wait();

    return res;
}

//
// build representation of dense matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and low-rank blocks computed by <lrapx>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//
template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hpro::TMatrix > >
build_uniform_lvl ( const hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const hpro::TTruncAcc &      acc,
                    const size_t                 /* nseq */ = hpro::CFG::Arith::max_seq_size ) // ignored
{
    return detail::build_uniform_lvl( bct, coeff, lrapx, basisapx, acc );
}

//
// build representation of dense matrix with matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and low-rank blocks computed by <lrapx>
// - low-rank blocks are converted to uniform low-rank matrices and
//   shared bases are constructed on-the-fly
//
template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::tuple< std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hlr::matrix::cluster_basis< typename coeff_t::value_t > >,
            std::unique_ptr< hpro::TMatrix > >
build_uniform_rec ( const hpro::TBlockCluster *  bct,
                    const coeff_t &              coeff,
                    const lrapx_t &              lrapx,
                    const basisapx_t &           basisapx,
                    const hpro::TTruncAcc &      acc,
                    const size_t                 /* nseq */ = hpro::CFG::Arith::max_seq_size ) // ignored
{
    static_assert( std::is_same_v< typename coeff_t::value_t, typename lrapx_t::value_t >,
                   "coefficient function and low-rank approximation must have equal value type" );
    static_assert( std::is_same_v< typename coeff_t::value_t, typename basisapx_t::value_t >,
                   "coefficient function and basis approximation must have equal value type" );
    
    assert( bct != nullptr );

    using value_t       = typename coeff_t::value_t;
    using cluster_basis = hlr::matrix::cluster_basis< value_t >;

    auto  rowcb  = std::make_unique< cluster_basis >( bct->is().row_is() );
    auto  colcb  = std::make_unique< cluster_basis >( bct->is().col_is() );

    rowcb->set_nsons( bct->rowcl()->nsons() );
    colcb->set_nsons( bct->colcl()->nsons() );

    detail::init_cluster_bases( bct, *rowcb, *colcb );
    
    auto  basis_data = detail::rec_basis_data_t();
    auto  M          = detail::build_uniform_rec( bct, coeff, lrapx, basisapx, acc, *rowcb, *colcb, basis_data );

    return  { std::move( rowcb ), std::move( colcb ), std::move( M ) };
}

//
// assign block cluster to matrix
//
inline
void
assign_cluster ( hpro::TMatrix &              M,
                 const hpro::TBlockCluster &  bc )
{
    hlr::seq::matrix::assign_cluster( M, bc );
}

//
// return copy of matrix
// - copy operation is performed in parallel for sub blocks
//
namespace detail
{

std::unique_ptr< hpro::TMatrix >
copy_helper ( ::tf::Subflow &        tf,
              const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    tf.emplace(
                        [B,BM,i,j] ( ::tf::Subflow &  sf )
                        {
                            auto  B_ij = copy_helper( sf, * BM->block( i, j ) );
                            
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        } );
                }// if
            }// for
        }// for

        return N;
    }// if
    else
    {
        // assuming non-structured block and hence no parallel copy needed
        return M.copy();
    }// else
}

}// namespace detail

std::unique_ptr< hpro::TMatrix >
copy ( const hpro::TMatrix &  M )
{
    ::tf::Taskflow                    tf;
    std::unique_ptr< hpro::TMatrix >  res;
    
    tf.emplace( [&M,&res] ( ::tf::Subflow &  sf ) { res = detail::copy_helper( sf, M ); } );

    ::tf::Executor  executor;
    
    executor.run( tf ).wait();

    return res;
}

//
// copy data of A to matrix B
// - ASSUMPTION: identical matrix structure
//
namespace detail
{

void
copy_to_helper ( ::tf::Subflow &        tf,
                 const hpro::TMatrix &  A,
                 hpro::TMatrix &        B )
{
    HLR_ASSERT( A.type()     == B.type() );
    HLR_ASSERT( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = ptrcast(  &B, hpro::TBlockMatrix );

        HLR_ASSERT( BA->nblock_rows() == BB->nblock_rows() );
        HLR_ASSERT( BA->nblock_cols() == BB->nblock_cols() );
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( BA->block( i, j ) != nullptr )
                {
                    tf.emplace(
                        [BA,BB,i,j] ( ::tf::Subflow &  sf )
                        {
                            HLR_ASSERT( ! is_null( BB->block( i, j ) ) );
                            
                            copy_to_helper( sf, * BA->block( i, j ), * BB->block( i, j ) );
                        } );
                }// if
            }// for
        }// for
    }// if
    else
    {
        A.copy_to( & B );
    }// else
}

}// namespace detail

void
copy_to ( const hpro::TMatrix &  A,
          hpro::TMatrix &        B )
{
    ::tf::Taskflow  tf;
    
    tf.emplace( [&A,&B] ( ::tf::Subflow &  sf ) { detail::copy_to_helper( sf, A, B ); } );

    ::tf::Executor  executor;
    
    executor.run( tf ).wait();
}

//
// reallocate matrix blocks
// - frees old data
// - local operation thereby limiting extra memory usage
//
namespace detail
{

std::unique_ptr< hpro::TMatrix >
realloc_helper ( ::tf::Subflow &  tf,
                 hpro::TMatrix *  A )
{
    if ( is_null( A ) )
        return nullptr;
    
    if ( is_blocked( A ) )
    {
        auto  B  = ptrcast( A, hpro::TBlockMatrix );
        auto  C  = std::make_unique< hpro::TBlockMatrix >();
        auto  BC = ptrcast( C.get(), hpro::TBlockMatrix );

        C->copy_struct_from( B );

        auto  sub_tasks = tf.emplace(
            [B,BC] ( ::tf::Subflow &  sf )
            {
                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < B->nblock_cols(); ++j )
                    {
                        sf.emplace(
                            [B,BC,i,j] ( ::tf::Subflow &  ssf )
                            {
                                auto  C_ij = realloc_helper( ssf, B->block( i, j ) );
                                
                                BC->set_block( i, j, C_ij.release() );
                                B->set_block( i, j, nullptr );
                            } );
                    }// for
                }// for
            } );

        auto  del_task = tf.emplace( [B] () { delete B; } );

        sub_tasks.precede( del_task );

        return C;
    }// if
    else
    {
        auto  C = A->copy();

        delete A;

        return C;
    }// else
}

}// namespace detail

std::unique_ptr< hpro::TMatrix >
realloc ( hpro::TMatrix *  A )
{
    ::tf::Taskflow                    tf;
    std::unique_ptr< hpro::TMatrix >  res;
    
    tf.emplace( [A,&res] ( ::tf::Subflow &  sf ) { res = detail::realloc_helper( sf, A ); } );

    ::tf::Executor  executor;
    
    executor.run( tf ).wait();

    return res;
}

}// namespace matrix

}// namespace tf

}// namespace hlr

#endif // __HLR_TF_MATRIX_HH

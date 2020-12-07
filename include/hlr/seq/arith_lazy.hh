#ifndef __HLR_SEQ_ARITH_LAZY_HH
#define __HLR_SEQ_ARITH_LAZY_HH
//
// Project     : HLib
// Module      : seq/arith_lazy.hh
// Description : sequential arithmetic functions using lazy evaluation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/arith/multiply.hh"

namespace hlr { namespace seq { namespace lazy {

namespace hpro = HLIB;

////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

using  update_t      = std::pair< const hpro::TMatrix *, const hpro::TMatrix * >;
using  update_list_t = std::list< update_t >;

//
// compute C = C + α op( A ) op( B ) where A and B are provided as list of updates
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           hpro::TMatrix &          C,
           update_list_t &          updates,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    if ( is_blocked( C ) )
    {
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );

        //
        // restrict each update to sub indexsets for each subblock of C
        // and recurse
        //
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( BC->block( i, j ) ) );
                
                auto  C_ij       = BC->block( i, j );
                auto  row_i      = C_ij->row_is();
                auto  col_j      = C_ij->col_is();
                auto  created    = std::list< std::unique_ptr< hpro::TMatrix > >();
                auto  updates_ij = update_list_t();
                
                for ( auto  [ A, B ] : updates )
                {
                    //
                    // extract sub-update for C_ij based on types of A/B
                    //

                    if ( is_blocked( A ) )
                    {
                        auto  BA = cptrcast( A, hpro::TBlockMatrix );
                        
                        if ( is_blocked( B ) )
                        {
                            auto  BB = cptrcast( B, hpro::TBlockMatrix );

                            for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                            {
                                HLR_ASSERT( ! is_null_all( BA->block( i, k ), BB->block( k, j ) ) );

                                updates_ij.push_back( std::make_pair( BA->block( i, k ), BB->block( k, j ) ) );
                            }// for
                        }// if
                        else if ( is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, hpro::TRkMatrix );

                            for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BA->block( i, k ) ) );

                                auto  A_ik  = BA->block( i, k );
                                auto  col_k = A_ik->col_is();

                                // restrict B to col_k × col_j
                                auto  U_k = blas::matrix( blas::mat_U< value_t >( RB ), col_k - B->row_ofs(), blas::range::all );
                                auto  V_j = blas::matrix( blas::mat_V< value_t >( RB ), col_j - B->col_ofs(), blas::range::all );
                                auto  B_kj = std::make_unique< hpro::TRkMatrix >( col_k, col_j, U_k, V_j );
                                
                                updates_ij.push_back( std::make_pair( A_ik, B_kj.get() ) );

                                created.push_back( std::move( B_kj ) );
                            }// for
                        }// if
                        else if ( is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, hpro::TDenseMatrix );

                            for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BA->block( i, k ) ) );

                                auto  A_ik  = BA->block( i, k );
                                auto  col_k = A_ik->col_is();

                                // restrict B to col_k × col_j
                                auto  D_kj = blas::matrix( blas::mat< value_t >( DB ), col_k - B->row_ofs(), col_j - B->col_ofs() );
                                auto  B_kj = std::make_unique< hpro::TDenseMatrix >( col_k, col_j, D_kj );
                                
                                updates_ij.push_back( std::make_pair( A_ik, B_kj.get() ) );

                                created.push_back( std::move( B_kj ) );
                            }// for
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type: " + B->typestr() );
                    }// if
                    else if ( is_lowrank( A ) )
                    {
                        auto  RA = cptrcast( A, hpro::TRkMatrix );
                        
                        if ( is_blocked( B ) )
                        {
                            auto  BB = cptrcast( B, hpro::TBlockMatrix );

                            for ( uint  k = 0; k < BB->nblock_rows(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BB->block( k, j ) ) );

                                auto  B_kj  = BB->block( k, j );
                                auto  col_k = B_kj->row_is();
                                
                                // restrict A to row_i × col_k
                                auto  U_i  = blas::matrix( blas::mat_U< value_t >( RA ), row_i - A->row_ofs(), blas::range::all );
                                auto  V_k  = blas::matrix( blas::mat_V< value_t >( RA ), col_k - A->col_ofs(), blas::range::all );
                                auto  A_ik = std::make_unique< hpro::TRkMatrix >( row_i, col_k, U_i, V_k );
                                
                                updates_ij.push_back( std::make_pair( A_ik.get(), B_kj ) );

                                created.push_back( std::move( A_ik ) );
                            }// for
                        }// if
                        else if ( is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, hpro::TRkMatrix );

                            // restrict A to row_i
                            auto  U_i = blas::matrix( blas::mat_U< value_t >( RA ), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< hpro::TRkMatrix >( row_i, A->col_is(), U_i, blas::mat_V< value_t >( RA ) );
                            
                            // restrict B to col_j
                            auto  V_j = blas::matrix( blas::mat_V< value_t >( RB ), col_j - B->col_ofs(), blas::range::all );
                            auto  B_j = std::make_unique< hpro::TRkMatrix >( B->row_is(), col_j, blas::mat_U< value_t >( RB ), V_j );
                                
                            updates_ij.push_back( std::make_pair( A_i.get(), B_j.get() ) );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else if ( is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, hpro::TDenseMatrix );

                            // restrict A to row_i
                            auto  U_i = blas::matrix( blas::mat_U< value_t >( RA ), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< hpro::TRkMatrix >( row_i, A->col_is(), U_i, blas::mat_V< value_t >( RA ) );
                            
                            // restrict B to col_j
                            auto  D_j = blas::matrix( blas::mat< value_t >( DB ), blas::range::all, col_j - B->col_ofs() );
                            auto  B_j = std::make_unique< hpro::TDenseMatrix >( B->row_is(), col_j, D_j );
                                
                            updates_ij.push_back( std::make_pair( A_i.get(), B_j.get() ) );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type: " + B->typestr() );
                    }// if
                    else if ( is_dense( A ) )
                    {
                        auto  DA = cptrcast( A, hpro::TDenseMatrix );
                        
                        if ( is_blocked( B ) )
                        {
                            auto  BB = cptrcast( B, hpro::TBlockMatrix );

                            for ( uint  k = 0; k < BB->nblock_rows(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BB->block( k, j ) ) );

                                auto  B_kj  = BB->block( k, j );
                                auto  col_k = B_kj->row_is();
                                
                                // restrict A to row_i × col_k
                                auto  D_ik = blas::matrix( blas::mat< value_t >( DA ), row_i - A->row_ofs(), col_k - A->col_ofs() );
                                auto  A_ik = std::make_unique< hpro::TDenseMatrix >( row_i, col_k, D_ik );
                                
                                updates_ij.push_back( std::make_pair( A_ik.get(), B_kj ) );

                                created.push_back( std::move( A_ik ) );
                            }// for
                        }// if
                        else if ( is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, hpro::TRkMatrix );
                            
                            // restrict A to row_i
                            auto  D_i = blas::matrix( blas::mat< value_t >( DA ), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< hpro::TDenseMatrix >( row_i, A->col_is(), D_i );
                            
                            // restrict B to col_j
                            auto  V_j = blas::matrix( blas::mat_V< value_t >( RB ), col_j - B->col_ofs(), blas::range::all );
                            auto  B_j = std::make_unique< hpro::TRkMatrix >( B->row_is(), col_j, blas::mat_U< value_t >( RB ), V_j );
                                
                            updates_ij.push_back( std::make_pair( A_i.get(), B_j.get() ) );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else if ( is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, hpro::TDenseMatrix );

                            // restrict A to row_i
                            auto  D_i = blas::matrix( blas::mat< value_t >( DA ), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< hpro::TDenseMatrix >( row_i, A->col_is(), D_i );
                            
                            // restrict B to col_j
                            auto  D_j = blas::matrix( blas::mat< value_t >( DB ), blas::range::all, col_j - B->col_ofs() );
                            auto  B_j = std::make_unique< hpro::TDenseMatrix >( B->row_is(), col_j, D_j );
                                
                            updates_ij.push_back( std::make_pair( A_i.get(), B_j.get() ) );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type: " + B->typestr() );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type: " + A->typestr() );
                }// for

                //
                // recurse
                //
                
                multiply< value_t >( alpha, *BC->block(i,j), updates_ij, acc, approx );
            }// for
        }// for
    }// if
    else 
    {
        // std::cout << C.id() << std::endl;

        // evaluate all updates and apply to C
        for ( auto  [ A, B ] : updates )
        {
            // std::cout << "    " << A->block_is() << " × " << B->block_is() << std::endl;
            
            hlr::multiply( alpha, apply_normal, *A, apply_normal, *B, C, acc, approx );
        }// for
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
    detail::update_list_t  updates{ std::make_pair( &A, &B ) };
    
    detail::multiply< value_t >( alpha, C, updates, acc, approx );
}

}}}// namespace hlr::seq::lazy

#endif // __HLR_SEQ_ARITH_LAZY_HH
#ifndef __HLR_MATRIX_INFO_HH
#define __HLR_MATRIX_INFO_HH
//
// Project     : HLR
// Module      : matrix/info
// Description : functions returning specific information about matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/config.h>

#if defined(HPRO_USE_LIC_CHECK)
#define HLR_HAS_H2
#endif

#include <hpro/matrix/TRkMatrix.hh>

#if defined(HLR_HAS_H2)
#include <hpro/matrix/TUniformMatrix.hh>
#endif

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/h2_lrmatrix.hh>

namespace hlr { namespace matrix {

//
// return min/avg/max rank of given matrix
//
namespace detail
{

template < typename value_t >
std::tuple< uint, size_t, uint, size_t >
rank_info_helper_mat ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto    B        = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        uint    min_rank = 0;
        uint    max_rank = 0;
        size_t  sum_rank = 0;
        size_t  nnodes   = 0;

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto [ min_ij, sum_ij, max_ij, n_ij ] = rank_info_helper_mat( *B->block( i, j ) );
                
                if      ( min_rank == 0 ) min_rank = min_ij;
                else if ( min_ij   != 0 ) min_rank = std::min( min_rank, min_ij );
                
                max_rank  = std::max( max_rank, max_ij );
                sum_rank += sum_ij;
                nnodes   += n_ij;
            }// for
        }// for

        return { min_rank, sum_rank, max_rank, nnodes };
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::lrmatrix< value_t > );

        return { R->rank(), R->rank(), R->rank(), R->rank() > 0 ? 1 : 0 };
    }// if
    else if ( matrix::is_lowrank_sv( M ) )
    {
        auto  R = cptrcast( &M, matrix::lrsvmatrix< value_t > );

        return { R->rank(), R->rank(), R->rank(), R->rank() > 0 ? 1 : 0 };
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        auto  R       = cptrcast( &M, matrix::uniform_lrmatrix< value_t > );
        auto  minrank = std::min( R->row_rank(), R->col_rank() );
        auto  maxrank = std::max( R->row_rank(), R->col_rank() );
        
        return { minrank, ( minrank + maxrank ) / 2, maxrank, minrank > 0 ? 1 : 0 };
    }// if
    else if ( matrix::is_h2_lowrank( &M ) )
    {
        auto  R       = cptrcast( &M, matrix::h2_lrmatrix< value_t > );
        auto  minrank = std::min( R->row_rank(), R->col_rank() );
        auto  maxrank = std::max( R->row_rank(), R->col_rank() );
        
        return { minrank, ( minrank + maxrank ) / 2, maxrank, minrank > 0 ? 1 : 0 };
    }// if
    // else if ( is_generic_lowrank( M ) )
    // {
    //     auto  R = cptrcast( &M, lrmatrix< value_t > );

    //     return { R->rank(), R->rank(), R->rank(), R->rank() > 0 ? 1 : 0 };
    // }// if

    return { 0, 0, 0, 0 };
}

}// namespace detail

template < typename value_t >
std::tuple< uint, uint, uint >
rank_info ( const Hpro::TMatrix< value_t > & M )
{
    auto [ min_rank, sum_rank, max_rank, nnodes ] = detail::rank_info_helper_mat( M );

    return { min_rank, uint( double(sum_rank) / double(nnodes) ), max_rank };
}

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_INFO_HH

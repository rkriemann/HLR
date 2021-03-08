#ifndef __HLR_MATRIX_RESTRICT_HH
#define __HLR_MATRIX_RESTRICT_HH
//
// Project     : HLib
// File        : restrict.hh
// Description : matrix restriction to sub blocks
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>

#include <hlr/arith/blas.hh>

namespace hlr { namespace matrix {

namespace hpro = HLIB;

using namespace hpro;
    
//
// return matrix restricted to subblock
//
inline
std::unique_ptr< TMatrix >
restrict ( TMatrix &               M,
           const TBlockIndexSet &  bis )
{
    HLR_ASSERT( bis.is_subset_of( M.block_is() ) );
    
    if ( is_lowrank( M ) )
    {
        auto  RM = cptrcast( &M, TRkMatrix );
        auto  R  = std::make_unique< TRkMatrix >( bis.row_is(), bis.col_is(), M.value_type() );

        if ( M.is_complex() )
        {
            auto  MU = RM->blas_cmat_A();
            auto  MV = RM->blas_cmat_B();
            auto  RU = blas::matrix< hpro::complex >( MU,
                                                      bis.row_is() - M.row_ofs(),
                                                      blas::range::all,
                                                      copy_value );
            auto  RV = blas::matrix< hpro::complex >( MV,
                                                      bis.col_is() - M.col_ofs(),
                                                      blas::range::all,
                                                      copy_value );

            R->set_lrmat( std::move( RU ), std::move( RV ) );
        }// if
        else
        {
            auto  MU = RM->blas_rmat_A();
            auto  MV = RM->blas_rmat_B();
            auto  RU = blas::matrix< hpro::real >( MU,
                                                   bis.row_is() - M.row_ofs(),
                                                   blas::range::all,
                                                   copy_value );
            auto  RV = blas::matrix< hpro::real >( MV,
                                                   bis.col_is() - M.col_ofs(),
                                                   blas::range::all,
                                                   copy_value );

            R->set_lrmat( std::move( RU ), std::move( RV ) );
        }// else

        return R;
    }// if
    else if ( is_dense( M ) )
    {
        auto  DM = cptrcast( &M, TDenseMatrix );

        if ( M.is_complex() )
        {
            auto  D = blas::matrix< hpro::complex >( DM->blas_cmat(),
                                                     bis.row_is() - M.row_ofs(),
                                                     bis.col_is() - M.col_ofs(),
                                                     copy_value );
                                               
            return std::make_unique< TDenseMatrix >( bis.row_is(), bis.col_is(), std::move( D ) );
        }// if
        else
        {
            auto  D = blas::matrix< hpro::real >( DM->blas_rmat(),
                                                  bis.row_is() - M.row_ofs(),
                                                  bis.col_is() - M.col_ofs(),
                                                  copy_value );
                                               
            return std::make_unique< TDenseMatrix >( bis.row_is(), bis.col_is(), std::move( D ) );
        }// else
    }// if
    else
        HLR_ERROR( "unsupported matrix type " + M.typestr() );
}

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_RESTRICT_HH

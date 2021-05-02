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
#include <hlr/matrix/lrsmatrix.hh>

namespace hlr { namespace matrix {

namespace hpro = HLIB;

template < typename value_t >
std::unique_ptr< hpro::TMatrix >
restrict ( matrix::lrsmatrix< value_t > &  M,
           const hpro::TBlockIndexSet &    bis );

//
// return matrix restricted to subblock
//
inline
std::unique_ptr< hpro::TMatrix >
restrict ( const hpro::TMatrix &         M,
           const hpro::TBlockIndexSet &  bis )
{
    HLR_ASSERT( bis.is_subset_of( M.block_is() ) );
    
    if ( is_lowrank( M ) )
    {
        auto  RM = cptrcast( &M, hpro::TRkMatrix );
        auto  R  = std::make_unique< hpro::TRkMatrix >( bis.row_is(), bis.col_is(), M.value_type() );

        if ( M.is_complex() )
        {
            auto  MU = RM->blas_cmat_A();
            auto  MV = RM->blas_cmat_B();
            auto  RU = blas::matrix< hpro::complex >( MU,
                                                      bis.row_is() - M.row_ofs(),
                                                      blas::range::all,
                                                      hpro::copy_value );
            auto  RV = blas::matrix< hpro::complex >( MV,
                                                      bis.col_is() - M.col_ofs(),
                                                      blas::range::all,
                                                      hpro::copy_value );

            R->set_lrmat( std::move( RU ), std::move( RV ) );
        }// if
        else
        {
            auto  MU = RM->blas_rmat_A();
            auto  MV = RM->blas_rmat_B();
            auto  RU = blas::matrix< hpro::real >( MU,
                                                   bis.row_is() - M.row_ofs(),
                                                   blas::range::all,
                                                   hpro::copy_value );
            auto  RV = blas::matrix< hpro::real >( MV,
                                                   bis.col_is() - M.col_ofs(),
                                                   blas::range::all,
                                                   hpro::copy_value );

            R->set_lrmat( std::move( RU ), std::move( RV ) );
        }// else

        return R;
    }// if
    else if ( is_dense( M ) )
    {
        auto  DM = cptrcast( &M, hpro::TDenseMatrix );

        if ( M.is_complex() )
        {
            auto  D = blas::matrix< hpro::complex >( DM->blas_cmat(),
                                                     bis.row_is() - M.row_ofs(),
                                                     bis.col_is() - M.col_ofs(),
                                                     hpro::copy_value );
                                               
            return std::make_unique< hpro::TDenseMatrix >( bis.row_is(), bis.col_is(), std::move( D ) );
        }// if
        else
        {
            auto  D = blas::matrix< hpro::real >( DM->blas_rmat(),
                                                  bis.row_is() - M.row_ofs(),
                                                  bis.col_is() - M.col_ofs(),
                                                  hpro::copy_value );
                                               
            return std::make_unique< hpro::TDenseMatrix >( bis.row_is(), bis.col_is(), std::move( D ) );
        }// else
    }// if
    else if ( is_sparse( M ) )
    {
        auto  SM = cptrcast( &M, hpro::TSparseMatrix );

        return SM->restrict( bis.row_is(), bis.col_is() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type " + M.typestr() );
}

template < typename value_t >
std::unique_ptr< hpro::TMatrix >
restrict ( matrix::lrsmatrix< value_t > &  M,
           const hpro::TBlockIndexSet &    bis )
{
    auto  RU = blas::matrix< hpro::complex >( M.U(),
                                              bis.row_is() - M.row_ofs(),
                                              blas::range::all,
                                              hpro::copy_value );
    auto  RV = blas::matrix< hpro::complex >( M.V(),
                                              bis.col_is() - M.col_ofs(),
                                              blas::range::all,
                                              hpro::copy_value );

    return std::make_unique< matrix::lrsmatrix< value_t > >( bis.row_is(), bis.col_is(),
                                                             std::move( RU ),
                                                             std::move( blas::copy( M.S() ) ),
                                                             std::move( RV ) );
}
    
}}// namespace hlr::matrix

#endif // __HLR_MATRIX_RESTRICT_HH

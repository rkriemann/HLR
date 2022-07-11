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

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
restrict ( matrix::lrsmatrix< value_t > &  M,
           const Hpro::TBlockIndexSet &    bis );

//
// return matrix restricted to subblock
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
restrict ( const Hpro::TMatrix< value_t > &  M,
           const Hpro::TBlockIndexSet &      bis )
{
    HLR_ASSERT( bis.is_subset_of( M.block_is() ) );
    
    if ( is_lowrank( M ) )
    {
        auto  RM = cptrcast( &M, Hpro::TRkMatrix< value_t > );
        auto  R  = std::make_unique< Hpro::TRkMatrix< value_t > >( bis.row_is(), bis.col_is() );
        auto  MU = RM->blas_mat_A();
        auto  MV = RM->blas_mat_B();
        auto  RU = blas::matrix< value_t >( MU,
                                            bis.row_is() - M.row_ofs(),
                                            blas::range::all,
                                            Hpro::copy_value );
        auto  RV = blas::matrix< value_t >( MV,
                                            bis.col_is() - M.col_ofs(),
                                            blas::range::all,
                                            Hpro::copy_value );
        
        R->set_lrmat( std::move( RU ), std::move( RV ) );

        return R;
    }// if
    else if ( is_dense( M ) )
    {
        auto  DM = cptrcast( &M, Hpro::TDenseMatrix< value_t > );
        auto  D  = blas::matrix< value_t >( DM->blas_mat(),
                                            bis.row_is() - M.row_ofs(),
                                            bis.col_is() - M.col_ofs(),
                                            Hpro::copy_value );
                                               
        return std::make_unique< Hpro::TDenseMatrix< value_t > >( bis.row_is(), bis.col_is(), std::move( D ) );
    }// if
    else if ( is_sparse( M ) )
    {
        auto  SM = cptrcast( &M, Hpro::TSparseMatrix< value_t > );

        return SM->restrict( bis.row_is(), bis.col_is() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type " + M.typestr() );
}

template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
restrict ( matrix::lrsmatrix< value_t > &  M,
           const Hpro::TBlockIndexSet &    bis )
{
    auto  RU = blas::matrix< value_t >( M.U(),
                                        bis.row_is() - M.row_ofs(),
                                        blas::range::all,
                                        Hpro::copy_value );
    auto  RV = blas::matrix< value_t >( M.V(),
                                        bis.col_is() - M.col_ofs(),
                                        blas::range::all,
                                        Hpro::copy_value );

    return std::make_unique< matrix::lrsmatrix< value_t > >( bis.row_is(), bis.col_is(),
                                                             std::move( RU ),
                                                             std::move( blas::copy( M.S() ) ),
                                                             std::move( RV ) );
}
    
}}// namespace hlr::matrix

#endif // __HLR_MATRIX_RESTRICT_HH

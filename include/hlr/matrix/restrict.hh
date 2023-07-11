#ifndef __HLR_MATRIX_RESTRICT_HH
#define __HLR_MATRIX_RESTRICT_HH
//
// Project     : HLR
// Module      : restrict.hh
// Description : matrix restriction to sub blocks
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
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

    if ( compress::is_compressible( M ) )
    {
        if ( matrix::is_lowrank( M ) )
        {
            auto  RM = cptrcast( &M, lrmatrix< value_t > );
            auto  R  = std::make_unique< lrmatrix< value_t > >( bis.row_is(), bis.col_is() );
            auto  MU = RM->U();
            auto  MV = RM->V();
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
        else if ( matrix::is_lowrank_sv( M ) )
        {
            auto  RM = cptrcast( &M, lrsvmatrix< value_t > );
            auto  R  = std::make_unique< lrsvmatrix< value_t > >( bis.row_is(), bis.col_is() );
            auto  MU = RM->U();
            auto  MV = RM->V();
            auto  RU = blas::matrix< value_t >( MU,
                                                bis.row_is() - M.row_ofs(),
                                                blas::range::all,
                                                Hpro::copy_value );
            auto  RS = blas::copy( RM->S() );
            auto  RV = blas::matrix< value_t >( MV,
                                                bis.col_is() - M.col_ofs(),
                                                blas::range::all,
                                                Hpro::copy_value );
        
            R->set_lrmat( std::move( RU ), std::move( RS ), std::move( RV ) );

            return R;
        }// if
        else if ( matrix::is_dense( M ) )
        {
            auto  DM = cptrcast( &M, dense_matrix< value_t > );
            auto  DD = DM->mat();
            auto  D  = blas::matrix< value_t >( DD,
                                                bis.row_is() - M.row_ofs(),
                                                bis.col_is() - M.col_ofs(),
                                                Hpro::copy_value );
                                               
            return std::make_unique< dense_matrix< value_t > >( bis.row_is(), bis.col_is(), std::move( D ) );
        }// if
        else
            HLR_ERROR( "unsupported matrix type " + M.typestr() );
    }// if
    else
    {
        if ( matrix::is_lowrank( M ) )
        {
            auto  RM = cptrcast( &M, lrmatrix< value_t > );
            auto  R  = std::make_unique< lrmatrix< value_t > >( bis.row_is(), bis.col_is() );
            auto  MU = RM->U();
            auto  MV = RM->V();
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
        else if ( matrix::is_dense( M ) )
        {
            auto  DM = cptrcast( &M, dense_matrix< value_t > );
            auto  D  = blas::matrix< value_t >( DM->mat(),
                                                bis.row_is() - M.row_ofs(),
                                                bis.col_is() - M.col_ofs(),
                                                Hpro::copy_value );
                                               
            return std::make_unique< dense_matrix< value_t > >( bis.row_is(), bis.col_is(), std::move( D ) );
        }// if
        else if ( is_sparse( M ) )
        {
            auto  SM = cptrcast( &M, Hpro::TSparseMatrix< value_t > );

            return SM->restrict( bis.row_is(), bis.col_is() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type " + M.typestr() );
    }// else
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

#ifndef __HLR_ARITH_DETAIL_MULTIPLY_COMPRESSED_HH
#define __HLR_ARITH_DETAIL_MULTIPLY_COMPRESSED_HH
//
// Project     : HLib
// Module      : multiply
// Description : matrix multiplication functions with compressed matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

namespace hlr
{

//
// blocked x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

//
// blocked x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

//
// blocked x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// blocked x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// blocked x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

//
// blocked x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TBlockMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

//
// dense x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_MULT_PRINT;
    
    std::scoped_lock  lock( C.mutex() );
    
    // C = C + A B
    auto  DA = A.mat_decompressed();
    auto  DB = B.mat_decompressed();
    
    blas::prod( alpha,
                blas::mat_view( op_A, DA ),
                blas::mat_view( op_B, DB ),
                value_t(1), blas::mat( C ) );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

//
// dense x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// dense x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TDenseMatrix< value_t > &           A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

//
// dense x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::dense_matrix< value_t > &         A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x blocked = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x blocked = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x blocked = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const Hpro::TBlockMatrix< value_t > &           B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x dense = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x dense = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const Hpro::TDenseMatrix< value_t > &           B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x dense = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::dense_matrix< value_t > &         B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x lowrank = blocked
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TBlockMatrix< value_t > &                 C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x lowrank = dense
//
template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::dense_matrix< value_t > &               C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const Hpro::TRkMatrix< value_t > &              A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

template < typename value_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const Hpro::TRkMatrix< value_t > &              B,
           Hpro::TDenseMatrix< value_t > &                 C )
{
    HLR_ERROR( "todo" );
}

//
// lowrank x lowrank = lowrank
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                                   alpha,
           const Hpro::matop_t                             op_A,
           const matrix::lrmatrix< value_t > &             A,
           const Hpro::matop_t                             op_B,
           const matrix::lrmatrix< value_t > &             B,
           matrix::lrmatrix< value_t > &                   C,
           const Hpro::TTruncAcc &                         acc,
           const approx_t &                                approx )
{
    HLR_ERROR( "todo" );
}

}// namespace hlr

#endif // __HLR_ARITH_DETAIL_MULTIPLY_COMPRESSED_HH

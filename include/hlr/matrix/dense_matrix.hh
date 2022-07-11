#ifndef __HLR_MATRIX_DENSE_MATRIX_HH
#define __HLR_MATRIX_DENSE_MATRIX_HH
//
// Project     : HLR
// Module      : dense_matrix
// Description : dense matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <vector>
#include <variant>

#include <hpro/matrix/TMatrix.hh>

#include <hlr/arith/blas.hh>
#include <hlr/utils/compression.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

using indexset = Hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( dense_matrix );

namespace matrix
{

// //
// // Represents a low-rank matrix in factorised form: U·S·V^H
// // with U and V represented as row/column cluster bases for
// // corresponding matrix block (maybe joined by more matrices).
// //
// class dense_matrix : public Hpro::TMatrix
// {
// private:
//     //
//     // compressed storage based on underlying floating point type
//     //
//     #if defined(HAS_SZ)

//     using  compressed_storage = hlr::sz::carray_view;
    
//     #elif defined(HAS_ZFP)
    
//     using  compressed_storage = hlr::zfp::carray;
    
//     #endif

// private:
//     // local index set of matrix
//     indexset              _row_is, _col_is;
    
//     // matrix coefficients
//     blas::generic_matrix  _M;

//     // indicates internal value type
//     // - after initialization identical to _M.index()
//     blas::value_type      _vtype;

//     #if defined(HAS_SZ) || defined(HAS_ZFP)
//     // optional: stores compressed data
//     compressed_storage    _zdata;
//     #endif
    
// public:
//     //
//     // ctors
//     //

//     dense_matrix ()
//             : TMatrix()
//             , _row_is( 0, 0 )
//             , _col_is( 0, 0 )
//             , _vtype( blas::value_type::undefined )
//     {
//     }
    
//     dense_matrix ( const indexset                arow_is,
//                    const indexset                acol_is )
//             : TMatrix()
//             , _row_is( arow_is )
//             , _col_is( acol_is )
//             , _vtype( blas::value_type::undefined )
//     {
//         set_ofs( _row_is.first(), _col_is.first() );
//     }

//     template < typename value_t >
//     dense_matrix ( const indexset                   arow_is,
//                    const indexset                   acol_is,
//                    hlr::blas::matrix< value_t > &   aM )
//             : TMatrix( Hpro::value_type_v< value_t > )
//             , _row_is( arow_is )
//             , _col_is( acol_is )
//             , _M( blas::copy( aM ) )
//             , _vtype( blas::value_type_v< value_t > )
//     {
//         HLR_ASSERT(( _row_is.size() == std::get< blas::matrix< value_t > >( _M ).nrows() ) &&
//                    ( _col_is.size() == std::get< blas::matrix< value_t > >( _M ).ncols() ));

//         set_ofs( _row_is.first(), _col_is.first() );
//     }

//     template < typename value_t >
//     dense_matrix ( const indexset                   arow_is,
//                    const indexset                   acol_is,
//                    hlr::blas::matrix< value_t > &&  aM )
//             : TMatrix( Hpro::value_type_v< value_t > )
//             , _row_is( arow_is )
//             , _col_is( acol_is )
//             , _M( std::move( aM ) )
//             , _vtype( blas::value_type_v< value_t > )
//     {
//         HLR_ASSERT(( _row_is.size() == std::get< blas::matrix< value_t > >( _M ).nrows() ) &&
//                    ( _col_is.size() == std::get< blas::matrix< value_t > >( _M ).ncols() ));

//         set_ofs( _row_is.first(), _col_is.first() );
//     }

//     // dtor
//     virtual ~dense_matrix ()
//     {}
    
//     //
//     // access internal data
//     //

//     // return value type of matrix
//     blas::value_type  value_type () const { return _vtype; }

//     blas::generic_matrix        matrix ()       { return _M; }
//     const blas::generic_matrix  matrix () const { return _M; }
    
//     template < typename value_t > blas::matrix< value_t > &        M ()       { return std::get< blas::matrix< value_t > >( _M ); }
//     template < typename value_t > const blas::matrix< value_t > &  M () const { return std::get< blas::matrix< value_t > >( _M ); }
    
//     template < typename value_t >
//     void
//     set_matrix ( const blas::matrix< value_t > &  aM )
//     {
//         HLR_ASSERT(( nrows() == aM.nrows() ) && ( ncols() == aM.ncols() ));

//         if ( is_compressed() )
//             remove_compressed();
        
//         if ( blas::value_type_v< value_t > == _vtype )
//         {
//             blas::copy( aM, M< value_t >() );
//         }// if
//         else
//         {
//             _M     = std::move( blas::copy( aM ) );
//             _vtype = blas::value_type_v< value_t >;
//         }// else
//     }
    
//     template < typename value_t >
//     void
//     set_lrmat ( blas::matrix< value_t > &&  aM )
//     {
//         HLR_ASSERT(( nrows() == aM.nrows() ) && ( ncols() == aM.ncols() ));

//         if ( is_compressed() )
//             remove_compressed();
        
//         _M     = std::move( aM );
//         _vtype = blas::value_type_v< value_t >;
//     }

//     //
//     // matrix data
//     //
    
//     virtual size_t  nrows     () const { return _row_is.size(); }
//     virtual size_t  ncols     () const { return _col_is.size(); }

//     virtual size_t  rows      () const { return nrows(); }
//     virtual size_t  cols      () const { return ncols(); }

//     // use "op" versions from TMatrix
//     using TMatrix::nrows;
//     using TMatrix::ncols;
    
//     // return true, if matrix is zero
//     virtual bool    is_zero   () const { return false; } // full test too expensive
    
//     virtual void    set_size  ( const size_t  ,
//                                 const size_t   ) {} // ignored
    
//     //
//     // change value type 
//     //
    
//     virtual void  to_real     () { HLR_ASSERT( false ); }
//     virtual void  to_complex  () { HLR_ASSERT( false ); }

//     //
//     // algebra routines
//     //

//     // compute y ≔ β·y + α·op(M)·x, with M = this
//     virtual void mul_vec  ( const Hpro::real       alpha,
//                             const Hpro::TVector *  x,
//                             const Hpro::real       beta,
//                             Hpro::TVector       *  y,
//                             const Hpro::matop_t    op = Hpro::apply_normal ) const;
//     using Hpro::TMatrix::mul_vec;
    
//     // compute y ≔ β·y + α·op(M)·x, with M = this
//     virtual void cmul_vec ( const Hpro::complex    alpha,
//                             const Hpro::TVector *  x,
//                             const Hpro::complex    beta,
//                             Hpro::TVector       *  y,
//                             const Hpro::matop_t    op = Hpro::apply_normal ) const;
    
//     // truncate matrix to accuracy \a acc
//     virtual void truncate ( const Hpro::TTruncAcc & ) {}

//     // scale matrix by alpha
//     virtual void scale    ( const Hpro::real  alpha )
//     {
//         if ( is_compressed() )
//         {
//             HLR_ERROR( "to do" );
//         }// if
//         else
//         {
//             std::visit(
//                 [alpha] ( auto &&  M )
//                 {
//                     using  value_t  = typename std::decay_t< decltype(M) >::value_t;
                    
//                     blas::scale( value_t(alpha), M );
//                 },
//                 _M );
//         }// else
//     }

//     //
//     // RTTI
//     //

//     HLIB_RTTI_DERIVED( dense_matrix, TMatrix )

//     //
//     // virtual constructor
//     //

//     // return matrix of same class (but no content)
//     virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix > { return std::make_unique< dense_matrix >(); }

//     // return copy of matrix
//     virtual auto   copy         () const -> std::unique_ptr< Hpro::TMatrix >;

//     // return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
//     virtual auto   copy         ( const Hpro::TTruncAcc &  acc,
//                                   const bool               do_coarsen = false ) const -> std::unique_ptr< Hpro::TMatrix >;

//     // return structural copy of matrix
//     virtual auto   copy_struct  () const -> std::unique_ptr< Hpro::TMatrix >;

//     // copy matrix data to \a A
//     virtual void   copy_to      ( Hpro::TMatrix *          A ) const;

//     // copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
//     virtual void   copy_to      ( Hpro::TMatrix *          A,
//                                   const Hpro::TTruncAcc &  acc,
//                                   const bool               do_coarsen = false ) const;
    
//     //
//     // misc.
//     //

//     // compress internal data
//     // - may result in non-compression if storage does not decrease
//     virtual void   compress      ( const zconfig_t &  config );

//     // uncompress internal data
//     virtual void   uncompress    ();

//     // return true if data is compressed
//     virtual bool   is_compressed () const
//     {
//         #if defined(HAS_SZ)
//         return ! is_null( _zdata.data() );
//         #elif defined(HAS_ZFP)
//         return ! is_null( _zdata.data() );
//         #else
//         return false;
//         #endif
//     }
    
//     // return size in bytes used by this object
//     virtual size_t byte_size     () const;

// protected:
//     // remove compressed storage (standard storage not restored!)
//     virtual void   remove_compressed ()
//     {
//         #if defined(HAS_SZ)
//         _zdata.free();
//         #elif defined(HAS_ZFP)
//         _zdata = zfp::carray();
//         #endif
//     }
    
// };

// //
// // type test
// //
// inline
// bool
// is_generic_dense ( const Hpro::TMatrix &  M )
// {
//     return IS_TYPE( &M, dense_matrix );
// }

// inline
// bool
// is_generic_dense ( const Hpro::TMatrix *  M )
// {
//     return ! is_null( M ) && IS_TYPE( M, dense_matrix );
// }

// HLR_TEST_ALL( is_generic_dense, Hpro::TMatrix )
// HLR_TEST_ANY( is_generic_dense, Hpro::TMatrix )

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_DENSE_MATRIX_HH

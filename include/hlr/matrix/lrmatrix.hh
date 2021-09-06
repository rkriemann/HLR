#ifndef __HLR_MATRIX_LRMATRIX_HH
#define __HLR_MATRIX_LRMATRIX_HH
//
// Project     : HLR
// Module      : lrmatrix
// Description : low-rank matrix with dynamic value type
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/vector/TScalarVector.hh>

#include <hlr/matrix/cluster_basis.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

namespace hpro = HLIB;

using indexset = hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( lrmatrix );

namespace matrix
{

//
// Represents a low-rank matrix in factorised form: U·S·V^H
// with U and V represented as row/column cluster bases for
// corresponding matrix block (maybe joined by more matrices).
//
class lrmatrix : public hpro::TMatrix
{
private:
    // local index set of matrix
    indexset              _row_is, _col_is;
    
    // lowrank factors
    blas::generic_matrix  _U, _V;

    // indicates internal value type
    blas::value_type      _vtype;
    
public:
    //
    // ctors
    //

    lrmatrix ()
            : TMatrix()
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
            , _vtype( blas::value_type::undefined )
    {
    }
    
    lrmatrix ( const indexset                arow_is,
               const indexset                acol_is )
            : TMatrix()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _vtype( blas::value_type::undefined )
    {
        set_ofs( _row_is.first(), _col_is.first() );
    }

    template < typename value_t >
    lrmatrix ( const indexset                   arow_is,
               const indexset                   acol_is,
               hlr::blas::matrix< value_t > &   aU,
               hlr::blas::matrix< value_t > &   aV )
            : TMatrix( hpro::value_type_v< value_t > )
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( blas::copy( aU ) )
            , _V( blas::copy( aV ) )
            , _vtype( blas::value_type_v< value_t > )
    {
        HLR_ASSERT(( _row_is.size() == std::get< blas::matrix< value_t > >( _U ).nrows() ) &&
                   ( _col_is.size() == std::get< blas::matrix< value_t > >( _V ).nrows() ) &&
                   ( std::get< blas::matrix< value_t > >( _U ).ncols() == std::get< blas::matrix< value_t > >( _V ).ncols() ));

        set_ofs( _row_is.first(), _col_is.first() );
    }

    template < typename value_t >
    lrmatrix ( const indexset                   arow_is,
               const indexset                   acol_is,
               hlr::blas::matrix< value_t > &&  aU,
               hlr::blas::matrix< value_t > &&  aV )
            : TMatrix( hpro::value_type_v< value_t > )
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( std::move( aU ) )
            , _V( std::move( aV ) )
            , _vtype( blas::value_type_v< value_t > )
    {
        HLR_ASSERT(( _row_is.size() == std::get< blas::matrix< value_t > >( _U ).nrows() ) &&
                   ( _col_is.size() == std::get< blas::matrix< value_t > >( _V ).nrows() ) &&
                   ( std::get< blas::matrix< value_t > >( _U ).ncols() == std::get< blas::matrix< value_t > >( _V ).ncols() ));

        set_ofs( _row_is.first(), _col_is.first() );
    }

    // dtor
    virtual ~lrmatrix ()
    {}
    
    //
    // access internal data
    //

    uint
    rank () const
    {
        return std::visit( [] ( auto &&  M ) { return M.ncols(); }, _U );
    }

    // return value type of matrix
    blas::value_type  value_type () const { return _vtype; }

    template < typename value_t > blas::matrix< value_t > &        U ()       { return std::get< blas::matrix< value_t > >( _U ); }
    template < typename value_t > blas::matrix< value_t > &        V ()       { return std::get< blas::matrix< value_t > >( _V ); }
    
    template < typename value_t > const blas::matrix< value_t > &  U () const { return std::get< blas::matrix< value_t > >( _U ); }
    template < typename value_t > const blas::matrix< value_t > &  V () const { return std::get< blas::matrix< value_t > >( _V ); }
    
    template < typename value_t >
    void
    set_lrmat ( const blas::matrix< value_t > &  aU,
                const blas::matrix< value_t > &  aV )
    {
        HLR_ASSERT(( nrows()    == aU.nrows() ) &&
                   ( ncols()    == aV.nrows() ) &&
                   ( aU.ncols() == aV.ncols() ));

        if (( blas::value_type_v< value_t > == _vtype ) && ( aU.ncols() == U< value_t >().ncols() ))
        {
            blas::copy( aU, U< value_t >() );
            blas::copy( aV, V< value_t >() );
        }// if
        else
        {
            _U = blas::copy( aU );
            _V = blas::copy( aV );

            _vtype = blas::value_type_v< value_t >;
        }// else
    }
    
    template < typename value_t >
    void
    set_lrmat ( blas::matrix< value_t > &&  aU,
                blas::matrix< value_t > &&  aV )
    {
        HLR_ASSERT(( nrows()    == aU.nrows() ) &&
                   ( ncols()    == aV.nrows() ) &&
                   ( aU.ncols() == aV.ncols() ));

        _U     = std::move( aU );
        _V     = std::move( aV );
        _vtype = blas::value_type_v< value_t >;
    }

    //
    // matrix data
    //
    
    virtual size_t  nrows     () const { return _row_is.size(); }
    virtual size_t  ncols     () const { return _col_is.size(); }

    virtual size_t  rows      () const { return nrows(); }
    virtual size_t  cols      () const { return ncols(); }

    // use "op" versions from TMatrix
    using TMatrix::nrows;
    using TMatrix::ncols;
    
    // return true, if matrix is zero
    virtual bool    is_zero   () const { return ( rank() == 0 ); }
    
    virtual void    set_size  ( const size_t  ,
                                const size_t   ) {} // ignored
    
    //
    // change value type 
    //
    
    virtual void  to_real     () { HLR_ASSERT( false ); }
    virtual void  to_complex  () { HLR_ASSERT( false ); }

    //
    // algebra routines
    //

    // compute y ≔ β·y + α·op(M)·x, with M = this
    virtual void mul_vec  ( const real             alpha,
                            const hpro::TVector *  x,
                            const real             beta,
                            hpro::TVector       *  y,
                            const hpro::matop_t    op = hpro::apply_normal ) const;
    
    // compute y ≔ β·y + α·op(M)·x, with M = this
    virtual void cmul_vec ( const complex          alpha,
                            const hpro::TVector *  x,
                            const complex          beta,
                            hpro::TVector       *  y,
                            const hpro::matop_t    op = hpro::apply_normal ) const;
    
    // truncate matrix to accuracy \a acc
    virtual void truncate ( const hpro::TTruncAcc & acc );

    // scale matrix by alpha
    virtual void scale    ( const real  alpha )
    {
        std::visit(
            [alpha] ( auto &&  M )
            {
                using  value_t  = typename std::decay_t< decltype(M) >::value_t;
                
                blas::scale( value_t(alpha), M );
            },
            _U );
    }

    //
    // RTTI
    //

    HLIB_RTTI_DERIVED( lrmatrix, TMatrix )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< hpro::TMatrix > { return std::make_unique< lrmatrix >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< hpro::TMatrix >;

    // return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
    virtual auto   copy         ( const hpro::TTruncAcc &  acc,
                                  const bool               do_coarsen = false ) const -> std::unique_ptr< hpro::TMatrix >;

    // return structural copy of matrix
    virtual auto   copy_struct  () const -> std::unique_ptr< hpro::TMatrix >;

    // copy matrix data to \a A
    virtual void   copy_to      ( hpro::TMatrix *          A ) const;

    // copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
    virtual void   copy_to      ( hpro::TMatrix *          A,
                                  const hpro::TTruncAcc &  acc,
                                  const bool               do_coarsen = false ) const;
    
    //
    // misc.
    //

    // return size in bytes used by this object
    virtual size_t byte_size  () const;
};

//
// type test
//
// inline
// bool
// is_lowrank ( const hpro::TMatrix &  M )
// {
//     return IS_TYPE( &M, lrmatrix );
// }

// inline
// bool
// is_lowrank ( const hpro::TMatrix *  M )
// {
//     return ! is_null( M ) && IS_TYPE( M, lrmatrix );
// }

// HLR_TEST_ALL( is_lowrank, hpro::TMatrix )
// HLR_TEST_ANY( is_lowrank, hpro::TMatrix )

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LRMATRIX_HH

#ifndef __HLR_MATRIX_LRSMATRIX_HH
#define __HLR_MATRIX_LRSMATRIX_HH
//
// Project     : HLR
// File        : lrsmatrix.hh
// Description : low-rank matrix with U·S·V' representation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <cassert>
#include <map>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/vector/TScalarVector.hh>

#include <hlr/arith/blas.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr
{ 

using indexset = Hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( lrsmatrix );

namespace matrix
{

//
// Represents a low-rank matrix in factorised form: U·S·V'.
// - U and V do not need to be orthogonal and are not shared
//
template < typename T_value >
class lrsmatrix : public Hpro::TMatrix< T_value >
{
public:
    //
    // export local types
    //

    // value type
    using  value_t = T_value;
    
private:
    // local index set of matrix
    indexset                      _row_is, _col_is;
    
    // low-rank factors
    blas::matrix< value_t >  _U, _S, _V;
    
public:
    //
    // ctors
    //

    lrsmatrix ()
            : Hpro::TMatrix< value_t >()
            , _row_is( 0, 0 )
            , _col_is( 0, 0 )
    {
    }
    
    lrsmatrix ( const indexset                arow_is,
                const indexset                acol_is )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    lrsmatrix ( const indexset                        arow_is,
                const indexset                        acol_is,
                const blas::matrix< value_t > &  aU,
                const blas::matrix< value_t > &  aS,
                const blas::matrix< value_t > &  aV )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( aU )
            , _S( aS )
            , _V( aV )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    lrsmatrix ( const indexset                   arow_is,
                const indexset                   acol_is,
                blas::matrix< value_t > &&  aU,
                blas::matrix< value_t > &&  aS,
                blas::matrix< value_t > &&  aV )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _U( std::move( aU ) )
            , _S( std::move( aS ) )
            , _V( std::move( aV ) )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    // dtor
    virtual ~lrsmatrix ()
    {}
    
    //
    // access internal data
    //

    uint  rank     () const { return std::min( _S.nrows(), _S.ncols() ); }

    uint  row_rank () const { return _S.nrows(); }
    uint  col_rank () const { return _S.ncols(); }

    uint  row_rank ( const Hpro::matop_t  op )       { return op == Hpro::apply_normal ? row_rank() : col_rank(); }
    uint  col_rank ( const Hpro::matop_t  op )       { return op == Hpro::apply_normal ? col_rank() : row_rank(); }

    const blas::matrix< value_t > &  U () const { return _U; }
    const blas::matrix< value_t > &  S () const { return _S; }
    const blas::matrix< value_t > &  V () const { return _V; }
    
    void
    set_lrmat ( const blas::matrix< value_t > &  aU,
                const blas::matrix< value_t > &  aS,
                const blas::matrix< value_t > &  aV )
    {
        HLR_ASSERT( aU.ncols() == aS.nrows() );
        HLR_ASSERT( aV.ncols() == aS.ncols() );
        
        _U = aU;
        _S = aS;
        _V = aV;
    }

    void
    set_lrmat ( blas::matrix< value_t > &&  aU,
                blas::matrix< value_t > &&  aS,
                blas::matrix< value_t > &&  aV )
    {
        HLR_ASSERT( aU.ncols() == aS.nrows() );
        HLR_ASSERT( aV.ncols() == aS.ncols() );
        
        _U = std::move( aU );
        _S = std::move( aS );
        _V = std::move( aV );
    }

    // modify coefficients S even if not consistent with U/V
    void
    set_coeff_unsafe ( const blas::matrix< value_t > &  T )
    {
        blas::copy( T, _S );
    }
    
    void
    set_coeff_unsafe ( blas::matrix< value_t > &&  T )
    {
        if (( _S.nrows() == T.nrows() ) && ( _S.ncols() == T.ncols() ))
            blas::copy( T, _S );
        else
            _S = std::move( T );
    }

    // clear row/column "bases" (HACK for parallel handling!!!)
    void clear_row_basis () { _U = blas::matrix< value_t >(); }
    void clear_col_basis () { _V = blas::matrix< value_t >(); }
    
    //
    // matrix data
    //
    
    virtual size_t  nrows     () const { return _row_is.size(); }
    virtual size_t  ncols     () const { return _col_is.size(); }

    virtual size_t  rows      () const { return nrows(); }
    virtual size_t  cols      () const { return ncols(); }

    // use "op" versions from TMatrix
    using Hpro::TMatrix< value_t >::nrows;
    using Hpro::TMatrix< value_t >::ncols;
    
    // return true, if matrix is zero
    virtual bool    is_zero   () const { return ( rank() == 0 ); }
    
    virtual void    set_size  ( const size_t  ,
                                const size_t   ) {} // ignored
    
    //
    // algebra routines
    //

    // compute y ≔ β·y + α·op(M)·x, with M = this
    virtual void mul_vec ( const value_t                     alpha,
                           const Hpro::TVector< value_t > *  x,
                           const value_t                     beta,
                           Hpro::TVector< value_t > *        y,
                           const Hpro::matop_t               op = Hpro::apply_normal ) const;
    
    // truncate matrix to accuracy \a acc
    virtual void truncate ( const Hpro::TTruncAcc & acc );

    // same as above but only the dimension of the vector spaces is tested,
    // not the corresponding index sets
    virtual void  apply_add   ( const value_t                    alpha,
                                const blas::vector< value_t > &  x,
                                blas::vector< value_t > &        y,
                                const matop_t                    op = apply_normal ) const;

    virtual void  apply_add   ( const value_t                    alpha,
                                const blas::matrix< value_t > &  X,
                                blas::matrix< value_t > &        Y,
                                const matop_t                    op = apply_normal ) const;

    using Hpro::TMatrix< value_t >::apply_add;
    
    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( lrsmatrix, Hpro::TMatrix< value_t > )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< lrsmatrix >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< Hpro::TMatrix< value_t > >;

    // return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
    virtual auto   copy         ( const Hpro::TTruncAcc &  acc,
                                  const bool               do_coarsen = false ) const -> std::unique_ptr< Hpro::TMatrix< value_t > >;

    // return structural copy of matrix
    virtual auto   copy_struct  () const -> std::unique_ptr< Hpro::TMatrix< value_t > >;

    // copy matrix data to \a A
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *          A ) const;

    // copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *          A,
                                  const Hpro::TTruncAcc &  acc,
                                  const bool               do_coarsen = false ) const;
    
    //
    // misc.
    //

    // return size in bytes used by this object
    virtual size_t byte_size  () const;
};

//
// matrix vector multiplication
//
template < typename value_t >
void
lrsmatrix< value_t >::mul_vec ( const value_t                     alpha,
                                const Hpro::TVector< value_t > *  vx,
                                const value_t                     beta,
                                Hpro::TVector< value_t > *        vy,
                                const Hpro::matop_t               op ) const
{
    assert( vx->is() == this->col_is( op ) );
    assert( vy->is() == this->row_is( op ) );
    assert( is_scalar_all( vx, vy ) );

    // exclude complex value and transposed operation for now
    assert( (  op == Hpro::apply_normal     ) ||
            (  op == Hpro::apply_adjoint    ) ||
            (( op == Hpro::apply_transposed ) && ! Hpro::is_complex_type_v< value_t > ) );

    const auto  x = cptrcast( vx, Hpro::TScalarVector< value_t > );
    const auto  y = ptrcast(  vy, Hpro::TScalarVector< value_t > );

    // y := β·y
    if ( beta != value_t(1) )
        blas::scale( value_t(beta), Hpro::blas_vec< value_t >( y ) );
                     
    if ( op == Hpro::apply_normal )
    {
        //
        // y = y + α U·(S·(V' x))
        //
        
        auto  t = blas::mulvec( blas::adjoint( _V ), blas::vec< value_t >( x ) );
        auto  s = blas::mulvec( value_t(1), _S, t );

        blas::mulvec( value_t(alpha), _U, s, value_t(1), blas::vec< value_t >( y ) );
    }// if
    else if ( op == Hpro::apply_transposed )
    {
        //
        // y = y + (U·S·V')^T x
        //   = y + conj(V)·(S^T·(U^T x))
        //
        
        auto  t = blas::mulvec( blas::transposed(_U), blas::vec< value_t >( x ) );
        auto  s = blas::mulvec( blas::transposed(_S), t );

        blas::conj( s );
        
        auto  r = blas::mulvec( _V, s );

        blas::conj( r );
        
        blas::add( value_t(alpha), r, blas::vec< value_t >( y ) );
    }// if
    else if ( op == Hpro::apply_adjoint )
    {
        //
        // y = y + α·(U·S·V')' x
        //   = y + α·V·(S'·(U' x))
        //
        
        auto  t = blas::mulvec( blas::adjoint(_U), blas::vec< value_t >( x ) );
        auto  s = blas::mulvec( blas::adjoint(_S), t );

        blas::mulvec( value_t(alpha), _V, s, value_t(1), blas::vec< value_t >( y ) );
    }// if
}

template < typename value_t >
void
lrsmatrix< value_t >::apply_add  ( const value_t                    alpha,
                                   const blas::vector< value_t > &  x,
                                   blas::vector< value_t > &        y,
                                   const matop_t                    op ) const
{
    switch ( op )
    {
        case apply_normal :
        {
            // y = y + U·(S·(V'·x))
            auto  t1 = blas::mulvec( blas::adjoint( _V ), x );
            auto  t2 = blas::mulvec( _S, t1 );

            blas::mulvec( alpha, _U, t2, value_t(1), y );
        }
        break;

        case apply_conjugate :
        {
            if constexpr ( Hpro::is_complex_type_v< value_t > )
            {
                HLR_ERROR( "not implemented" );
            }// if
            else
            {
                // y = y + U·(S·(V'·x))
                auto  t1 = blas::mulvec( blas::adjoint( _V ), x );
                auto  t2 = blas::mulvec( _S, t1 );

                blas::mulvec( alpha, _U, t2, value_t(1), y );
            }// else
        }
        break;
        
        case apply_transposed :
        {
            if constexpr ( Hpro::is_complex_type_v< value_t > )
            {
                HLR_ERROR( "not implemented" );
            }// if
            else
            {
                // y = y + V·(S'·(U'·x))
                auto  t1 = blas::mulvec( blas::adjoint( _U ), x );
                auto  t2 = blas::mulvec( blas::adjoint( _S ), t1 );

                blas::mulvec( alpha, _V, t2, value_t(1), y );
            }// else
        }
        break;
        
        case apply_adjoint :
        {
            // y = y + V·(S'·(U'·x))
            auto  t1 = blas::mulvec( blas::adjoint( _U ), x );
            auto  t2 = blas::mulvec( blas::adjoint( _S ), t1 );

            blas::mulvec( alpha, _V, t2, value_t(1), y );
        }
        break;
    }// switch
}

template < typename value_t >
void
lrsmatrix< value_t >::apply_add ( const value_t                    alpha,
                                  const blas::matrix< value_t > &  X,
                                  blas::matrix< value_t > &        Y,
                                  const matop_t                    op ) const
{
    switch ( op )
    {
        case apply_normal :
        {
            // Y = Y + U·(S·(V'·X))
            auto  T1 = blas::prod( blas::adjoint( _V ), X );
            auto  T2 = blas::prod( _S, T1 );

            blas::prod( alpha, _U, T2, value_t(1), Y );
        }
        break;

        case apply_conjugate :
        {
            if constexpr ( Hpro::is_complex_type_v< value_t > )
            {
                HLR_ERROR( "not implemented" );
            }// if
            else
            {
                // Y = Y + U·(S·(V'·X))
                auto  T1 = blas::prod( blas::adjoint( _V ), X );
                auto  T2 = blas::prod( _S, T1 );
                
                blas::prod( alpha, _U, T2, value_t(1), Y );
            }// else
        }
        break;
        
        case apply_transposed :
        {
            if constexpr ( Hpro::is_complex_type_v< value_t > )
            {
                HLR_ERROR( "not implemented" );
            }// if
            else
            {
                // Y = Y + V·(S'·(U'·X))
                auto  T1 = blas::prod( blas::adjoint( _U ), X );
                auto  T2 = blas::prod( blas::adjoint( _S ), T1 );
                
                blas::prod( alpha, _V, T2, value_t(1), Y );
            }// else
        }
        break;
        
        case apply_adjoint :
        {
            // Y = Y + V·(S'·(U'·X))
            auto  T1 = blas::prod( blas::adjoint( _U ), X );
            auto  T2 = blas::prod( blas::adjoint( _S ), T1 );

            blas::prod( alpha, _V, T2, value_t(1), Y );
        }
        break;
    }// switch
}

//
// truncate matrix to accuracy <acc>
//
template < typename value_t >
void
lrsmatrix< value_t >::truncate ( const Hpro::TTruncAcc & )
{
}

//
// return copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
lrsmatrix< value_t >::copy () const
{
    auto  M = std::make_unique< lrsmatrix >( _row_is, _col_is,
                                             std::move( blas::copy( _U ) ),
                                             std::move( blas::copy( _S ) ),
                                             std::move( blas::copy( _V ) ) );

    M->copy_struct_from( this );
    
    return M;
}

//
// return copy matrix wrt. given accuracy; if \a do_coarsen is set, perform coarsening
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
lrsmatrix< value_t >::copy ( const Hpro::TTruncAcc &,
                             const bool       ) const
{
    return copy();
}

//
// return structural copy of matrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
lrsmatrix< value_t >::copy_struct  () const
{
    return std::make_unique< lrsmatrix >( _row_is, _col_is );
}

//
// copy matrix data to \a A
//
template < typename value_t >
void
lrsmatrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A ) const
{
    Hpro::TMatrix< value_t >::copy_to( A );
    
    assert( IS_TYPE( A, lrsmatrix ) );

    auto  R = ptrcast( A, lrsmatrix );

    R->_row_is = _row_is;
    R->_col_is = _col_is;
    R->_U      = std::move( blas::copy( _U ) );
    R->_S      = std::move( blas::copy( _S ) );
    R->_V      = std::move( blas::copy( _V ) );
}

//
// copy matrix data to \a A and truncate w.r.t. \acc with optional coarsening
//
template < typename value_t >
void
lrsmatrix< value_t >::copy_to ( Hpro::TMatrix< value_t > *  A,
                                const Hpro::TTruncAcc &,
                                const bool          ) const
{
    return copy_to( A );
}

//
// return size in bytes used by this object
//
template < typename value_t >
size_t
lrsmatrix< value_t >::byte_size () const
{
    size_t  size = Hpro::TMatrix< value_t >::byte_size();

    size += sizeof(_row_is) + sizeof(_col_is);
    size += _U.byte_size() + _S.byte_size() + _V.byte_size();

    return size;
}

//
// type test
//
template < typename value_t >
bool
is_lowrankS ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, lrsmatrix );
}

template < typename value_t >
bool
is_lowrankS ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, lrsmatrix );
}

HLR_TEST_ALL( is_lowrankS, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_lowrankS, Hpro::TMatrix< value_t > )

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LRSMATRIX_HH

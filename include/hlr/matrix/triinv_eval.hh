#ifndef __HLR_MATRIX_TRIINV_EVAL_HH
#define __HLR_MATRIX_TRIINV_EVAL_HH
//
// Project     : HLR
// File        : triinv_eval.hh
// Description : evaluation operator for the inverse of triangular matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <memory>
#include <map>
#include <mutex>

#include <hpro/matrix/TLinearOperator.hh>

#include <hlr/arith/blas.hh>
#include <hlr/dag/graph.hh>
#include <hlr/dag/solve.hh>
#include <hlr/seq/dag.hh>

namespace hlr { namespace matrix {

namespace hpro = HLIB;

// local matrix type
DECLARE_TYPE( triinv_eval );

//
// implements vector solving for LU using DAGs
//
template < typename T_value >
class triinv_eval : public Hpro::TLinearOperator< T_value >
{
public:
    using  value_t = T_value;
    
private:
    // upper or lower triangular
    blas::tri_type_t            _shape;
    
    // diagonal evaluation
    blas::diag_type_t           _diag;
    
    // matrix containing triangular matrix
    Hpro::TMatrix< value_t > &  _mat;

public:
    //
    // ctor
    //

    triinv_eval ( Hpro::TMatrix< value_t > &  M,
                  const blas::tri_type_t      shape,
                  const blas::diag_type_t     diag )
            : _shape( shape )
            , _diag( diag )
            , _mat( M )
    {}
    
    //
    // linear operator properties
    //

    // return true, of operator is self adjoint
    bool  is_self_adjoint () const
    {
        return false;
    }
    
    //
    // linear operator mapping
    //

    //
    // mapping function of linear operator A, e.g. y ≔ A(x).
    // Depending on \a op, either A, A^T or A^H is applied.
    //
    virtual void  apply       ( const Hpro::TVector< value_t > *  x,
                                Hpro::TVector< value_t > *        y,
                                const Hpro::matop_t               op = Hpro::apply_normal ) const;

    //
    // mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
    // Depending on \a op, either A, A^T or A^H is applied.
    //
    virtual void  apply_add   ( const value_t                     alpha,
                                const Hpro::TVector< value_t > *  x,
                                Hpro::TVector< value_t > *        y,
                                const Hpro::matop_t               op = Hpro::apply_normal ) const;

    virtual void  apply_add   ( const value_t                     alpha,
                                const Hpro::TMatrix< value_t > *  X,
                                Hpro::TMatrix< value_t > *        Y,
                                const Hpro::matop_t               op = Hpro::apply_normal ) const;
    
    // same as above but only the dimension of the vector spaces is tested,
    // not the corresponding index sets
    virtual void  apply_add   ( const value_t                     alpha,
                                const blas::vector< value_t > &   x,
                                blas::vector< value_t > &         y,
                                const Hpro::matop_t               op = Hpro::apply_normal ) const;

    virtual void  apply_add   ( const value_t                     alpha,
                                const blas::matrix< value_t > &   X,
                                blas::matrix< value_t > &         Y,
                                const Hpro::matop_t               op = Hpro::apply_normal ) const;

    //
    // access vector space data
    //

    // return dimension of domain
    virtual size_t  domain_dim     () const { return _mat.nrows(); }
    
    // return dimension of range
    virtual size_t  range_dim      () const { return _mat.ncols(); }
    
    // return vector in domain space
    virtual auto    domain_vector  () const -> std::unique_ptr< Hpro::TVector< value_t > > { return _mat.row_vector(); }

    // return vector in range space
    virtual auto    range_vector   () const -> std::unique_ptr< Hpro::TVector< value_t > > { return _mat.col_vector(); }

    //
    // misc.
    //

    // RTTI
    HPRO_RTTI_DERIVED( triinv_eval, Hpro::TLinearOperator< value_t > )
};


}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LEVELMATRIX_HH

#ifndef __HLR_MATRIX_LUINV_EVAL_HH
#define __HLR_MATRIX_LUINV_EVAL_HH
//
// Project     : HLR
// File        : luinv_eval.hh
// Description : evaluation operator for the inverse of LU factorizations
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <memory>
#include <map>
#include <mutex>

#include <hpro/matrix/TLinearOperator.hh>

#include <hlr/arith/blas.hh>

namespace hlr { namespace matrix {

namespace hpro = HLIB;

// local matrix type
DECLARE_TYPE( luinv_eval );

//
// implements vector solving for LU factored matrices
//
class luinv_eval : public hpro::TLinearOperator
{
private:
    // matrix containing LU data
    const hpro::TMatrix &  _mat;

public:
    //
    // ctor
    //

    luinv_eval ( const hpro::TMatrix &  M )
            : _mat( M )
    {}
    
    //
    // linear operator properties
    //

    // return true, if field type is complex
    bool  is_complex      () const
    {
        return _mat.is_complex();
    }
    
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
    virtual void  apply       ( const hpro::TVector *  x,
                                hpro::TVector *        y,
                                const hpro::matop_t    op = hpro::apply_normal ) const;

    //
    // mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
    // Depending on \a op, either A, A^T or A^H is applied.
    //
    virtual void  apply_add   ( const hpro::real       alpha,
                                const hpro::TVector *  x,
                                hpro::TVector *        y,
                                const hpro::matop_t    op = hpro::apply_normal ) const;
    virtual void  capply_add  ( const hpro::complex    alpha,
                                const hpro::TVector *  x,
                                hpro::TVector *        y,
                                const hpro::matop_t    op = hpro::apply_normal ) const;

    virtual void  apply_add   ( const hpro::real       alpha,
                                const hpro::TMatrix *  X,
                                hpro::TMatrix *        Y,
                                const hpro::matop_t    op = hpro::apply_normal ) const;
    
    // same as above but only the dimension of the vector spaces is tested,
    // not the corresponding index sets
    virtual void  apply_add   ( const hpro::real                       alpha,
                                const blas::vector< hpro::real > &     x,
                                blas::vector< hpro::real > &           y,
                                const hpro::matop_t                    op = hpro::apply_normal ) const;
    virtual void  apply_add   ( const hpro::complex                    alpha,
                                const blas::vector< hpro::complex > &  x,
                                blas::vector< hpro::complex > &        y,
                                const hpro::matop_t                    op = hpro::apply_normal ) const;

    virtual void  apply_add   ( const hpro::real                       alpha,
                                const blas::matrix< hpro::real > &     X,
                                blas::matrix< hpro::real > &           Y,
                                const hpro::matop_t                    op = hpro::apply_normal ) const;
    virtual void  apply_add   ( const hpro::complex                    alpha,
                                const blas::matrix< hpro::complex > &  X,
                                blas::matrix< hpro::complex > &        Y,
                                const hpro::matop_t                    op = hpro::apply_normal ) const;

    //
    // access vector space data
    //

    // return dimension of domain
    virtual size_t  domain_dim     () const { return _mat.nrows(); }
    
    // return dimension of range
    virtual size_t  range_dim      () const { return _mat.ncols(); }
    
    // return vector in domain space
    virtual auto    domain_vector  () const -> std::unique_ptr< hpro::TVector > { return _mat.row_vector(); }

    // return vector in range space
    virtual auto    range_vector   () const -> std::unique_ptr< hpro::TVector > { return _mat.col_vector(); }

    //
    // misc.
    //

    // RTTI
    HLIB_RTTI_DERIVED( luinv_eval, hpro::TLinearOperator )
};


}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LEVELMATRIX_HH

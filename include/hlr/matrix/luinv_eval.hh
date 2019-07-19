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

#include <matrix/TLinearOperator.hh>

#include <hlr/dag/graph.hh>
#include <hlr/dag/solve.hh>
#include <hlr/seq/dag.hh>

namespace hlr { namespace matrix {

// local matrix type
DECLARE_TYPE( luinv_eval );

//
// implements vector solving for LU using DAGs
//
class luinv_eval : public HLIB::TLinearOperator
{
private:
    // matrix containing LU data
    std::shared_ptr< HLIB::TMatrix >  _mat;

    // holds pointer to vector to solve
    mutable HLIB::TScalarVector *     _vec;
    
    // DAGs for solving with LU
    mutable hlr::dag::graph           _dag_trsvl, _dag_trsvlt, _dag_trsvlh;
    mutable hlr::dag::graph           _dag_trsvu, _dag_trsvut, _dag_trsvuh;

    // mutex maps for updating vectors
    hlr::dag::mutex_map_t             _map_rows, _map_cols;

    // dag execution function
    hlr::dag::exec_func_t             _exec_func;
    
public:
    //
    // ctor
    //

    luinv_eval ( std::shared_ptr< HLIB::TMatrix > &  M,
                 hlr::dag::refine_func_t             refine_func = seq::dag::refine,
                 hlr::dag::exec_func_t               exec_func   = seq::dag::run );
    
    //
    // linear operator properties
    //

    // return true, if field type is complex
    bool  is_complex      () const
    {
        return _mat->is_complex();
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
    virtual void  apply       ( const HLIB::TVector *  x,
                                HLIB::TVector *        y,
                                const HLIB::matop_t    op = HLIB::apply_normal ) const;

    //
    // mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
    // Depending on \a op, either A, A^T or A^H is applied.
    //
    virtual void  apply_add   ( const HLIB::real       alpha,
                                const HLIB::TVector *  x,
                                HLIB::TVector *        y,
                                const HLIB::matop_t    op = HLIB::apply_normal ) const;
    virtual void  capply_add  ( const HLIB::complex    alpha,
                                const HLIB::TVector *  x,
                                HLIB::TVector *        y,
                                const HLIB::matop_t    op = HLIB::apply_normal ) const;

    virtual void  apply_add   ( const HLIB::real       alpha,
                                const HLIB::TMatrix *  X,
                                HLIB::TMatrix *        Y,
                                const HLIB::matop_t    op = HLIB::apply_normal ) const;
    
    //
    // access to vector space elements
    //

    // return vector in domain space
    virtual auto domain_vector  () const -> std::unique_ptr< HLIB::TVector > { return _mat->row_vector(); }

    // return vector in range space
    virtual auto range_vector   () const -> std::unique_ptr< HLIB::TVector > { return _mat->col_vector(); }

    //
    // misc.
    //

    // RTTI
    HLIB_RTTI_DERIVED( luinv_eval, HLIB::TLinearOperator )
};


}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LEVELMATRIX_HH

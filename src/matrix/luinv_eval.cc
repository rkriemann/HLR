//
// Project     : HLR
// File        : luinv_eval.cc
// Description : evaluation operator for the inverse of LU factorizations
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlr/matrix/luinv_eval.hh>

namespace hlr { namespace matrix {

using namespace HLIB;

//
// ctor
//

luinv_eval::luinv_eval ( std::shared_ptr< TMatrix > &  M )
        : _mat( M )
{
    assert( _mat.get() != nullptr );
}
    
//
// linear operator mapping
//

//
// mapping function of linear operator A, e.g. y ≔ A(x).
// Depending on \a op, either A, A^T or A^H is applied.
//
void
luinv_eval::apply  ( const TVector *  x,
                     TVector *        y,
                     const matop_t    op ) const
{
}

//
// mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
// Depending on \a op, either A, A^T or A^H is applied.
//
void
luinv_eval::apply_add  ( const real       alpha,
                         const TVector *  x,
                         TVector *        y,
                         const matop_t    op ) const
{
}

void
luinv_eval::capply_add  ( const complex    alpha,
                          const TVector *  x,
                          TVector *        y,
                          const matop_t    op ) const
{
}

void
luinv_eval::apply_add  ( const real       alpha,
                         const TMatrix *  X,
                         TMatrix *        Y,
                         const matop_t    op ) const
{
}

}} // namespace hlr::matrix

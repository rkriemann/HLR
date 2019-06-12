#ifndef __HLR_MATRIX_LEVELMATRIX_HH
#define __HLR_MATRIX_LEVELMATRIX_HH
//
// Project     : HLR
// File        : level_matrix.hh
// Description : block matrix for full level of H-matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <memory>
#include <vector>

#include <matrix/TBlockMatrix.hh>

namespace hlr { namespace matrix {

// local matrix type
DECLARE_TYPE( level_matrix );

//
// block matrix representing a single, global level
// in the H hierarchy
//
class level_matrix : public HLIB::TBlockMatrix
{
private:
    // pointers to level matrices above and below
    std::shared_ptr< level_matrix >  _above;
    std::shared_ptr< level_matrix >  _below;

public:
    //
    // ctor
    //

    level_matrix ( const uint               nrows,
                   const uint               ncols,
                   const HLIB::TIndexSet &  rowis,
                   const HLIB::TIndexSet &  colis );

    //
    // give access to level hierarchy
    //

    level_matrix *  above () { return  _above.get(); }
    level_matrix *  below () { return  _below.get(); }

    void  set_above ( std::shared_ptr< level_matrix > &  M ) { _above = M; }
    void  set_below ( std::shared_ptr< level_matrix > &  M ) { _below = M; }

    //
    // RTTI
    //

    HLIB_RTTI_DERIVED( level_matrix, TBlockMatrix )

};

//
// construct set of level matrices for given H-matrix
//
std::vector< std::shared_ptr< level_matrix > >
construct_lvlhier ( HLIB::TMatrix &  A );

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LEVELMATRIX_HH

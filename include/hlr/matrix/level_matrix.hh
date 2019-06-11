#ifndef __HLR_MATRIX_LEVELMATRIX_HH
#define __HLR_MATRIX_LEVELMATRIX_HH
//
// Project     : HLR
// File        : level_matrix.hh
// Description : block matrix for full level of H-matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <matrix/TBlockMatrix.hh>

namespace hlr { namespace matrix {

class level_matrix : public HLIB::TBlockMatrix
{
private:
    // pointers to level matrices above and below
    level_matrix *  _above;
    level_matrix *  _below;

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

    level_matrix *  above () { return  _above; }
    level_matrix *  below () { return  _below; }
};

//
// construct set of level matrices for given H-matrix
//
std::unique_ptr< level_matrix >
construct_lvlhier ( HLIB::TMatrix &  A );

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LEVELMATRIX_HH

#ifndef __HLR_MATRIX_PRINT_HH
#define __HLR_MATRIX_PRINT_HH
//
// Project     : HLR
// Module      : matrix/print
// Description : printing functions for matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <string>

#include <hpro/matrix/TMatrix.hh>

namespace hlr { namespace matrix {

namespace hpro = HLIB;

//
// print matrix <M> to file <filename>
//
void
print_eps ( const hpro::TMatrix &  M,
            const std::string &    filename );

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_PRINT_HH

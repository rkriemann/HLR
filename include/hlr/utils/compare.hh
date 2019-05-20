#ifndef __HLR_COMPARE_HH
#define __HLR_COMPARE_HH
//
// Project     : HLib
// File        : compare.hh
// Description : comparison function for matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <string>

#include <matrix/TMatrix.hh>

namespace hlr
{

//
// compare <A> with reference read from file <filename>
//
void
compare_ref_file ( HLIB::TMatrix *      A,
                   const std::string &  filename );

}// namespace hlr

#endif // __HLR_COMPARE_HH

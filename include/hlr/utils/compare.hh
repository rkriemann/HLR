#ifndef __HLR_COMPARE_HH
#define __HLR_COMPARE_HH
//
// Project     : HLR
// Module      : compare.hh
// Description : comparison function for matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <string>

#include <hpro/matrix/TMatrix.hh>

namespace hlr
{

//
// compare blocks of <A> with reference read from file <filename>
// and output difference if larger then <error>
//
template < typename value_t >
void
compare_ref_file ( Hpro::TMatrix< value_t > *  A,
                   const std::string &         filename,
                   const double                error );

}// namespace hlr

#endif // __HLR_COMPARE_HH

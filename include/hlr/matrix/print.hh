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

#include <hlr/matrix/cluster_basis.hh>

namespace hlr { namespace matrix {

//
// print matrix <M> to file <filename>
//
template < typename value_t >
void
print_eps ( const Hpro::TMatrix< value_t > &  M,
            const std::string &               filename,
            const std::string &               options = "default" );

//
// print matrix <M> level wise to the files
// <basename><lvl>.eps, e.g., each level (excluding root)
// will be written to a separate file
//
template < typename value_t >
void
print_lvl_eps ( const Hpro::TMatrix< value_t > &  M,
                const std::string &               basename,
                const std::string &               options = "default" );

//
// colorize matrix blocks in <M> according to rank
//
template < typename value_t >
void
print_mem_eps ( const Hpro::TMatrix< value_t > &  M,
                const std::string &               filename,
                const std::string &               options = "default" );

//
// print cluster basis <cl> to file <filename>
//
template < typename value_t >
void
print_eps ( const cluster_basis< value_t > &  cb,
            const std::string &               filename,
            const std::string &               options = "default" );

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_PRINT_HH

#ifndef __HLR_UTILS_IO_HH
#define __HLR_UTILS_IO_HH
//
// Project     : HLR
// Module      : utils/io
// Description : IO related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <string>

#include <hpro/io/TMatrixIO.hh>

#include <hlr/arith/blas.hh>
#include <hlr/utils/checks.hh>

namespace hlr { namespace io {

//
// write blas matrix/vector in Matlab format with given name
// - if filename is empty, the matrix/vector name is used
//
template < typename value_t >
void
write_matlab ( const blas::matrix< value_t > &  M,
               const std::string &              matname,
               const std::string &              filename = "" )
{
    if ( filename == "" )
        hpro::DBG::write( M, matname + ".mat", matname );
    else
        hpro::DBG::write( M, filename, matname );
}

template < typename value_t >
void
write_matlab ( const blas::vector< value_t > &  v,
               const std::string &              vecname,
               const std::string &              filename = "" )
{
    if ( filename == "" )
        hpro::DBG::write( v, vecname + ".mat", vecname );
    else
        hpro::DBG::write( v, filename, vecname );
}

//
// read matrix with given matrix name from given Matlab file
// - if matrix name is empty, first matrix in file is returned
//
template < typename value_t >
blas::matrix< value_t >
read_matlab ( const std::string &  filename,
              const std::string &  matname = "" )
{
    hpro::TMatlabMatrixIO  mio;
    auto                   D = mio.read( filename, matname );

    HLR_ASSERT( is_dense( *D ) );
    
    return std::move( blas::mat< value_t >( ptrcast( D.get(), hpro::TDenseMatrix ) ) );
}

}}// namespace hlr::io

#endif // __HLR_UTILS_IO_HH

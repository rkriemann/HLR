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
#include <hpro/io/TClusterVis.hh>
#include <hpro/io/TCoordVis.hh>

#include <hlr/arith/blas.hh>
#include <hlr/utils/checks.hh>
#include <hlr/matrix/print.hh>

namespace hlr { namespace io {

//////////////////////////////////////////////////////////////////////
//
// HLIBpro format
//
//////////////////////////////////////////////////////////////////////

namespace hpro
{

//
// write matrix M to file <filename>
//
inline
void
write ( const HLIB::TMatrix &  M,
        const std::string &    filename )
{
    HLIB::THLibMatrixIO  mio;

    mio.write( &M, filename );
}

//
// read matrix M from file <filename>
//
inline
std::unique_ptr< HLIB::TMatrix >
read ( const std::string &  filename )
{
    HLIB::THLibMatrixIO  mio;

    return mio.read( filename );
}

}// namespace hpro

//////////////////////////////////////////////////////////////////////
//
// Matlab format
//
//////////////////////////////////////////////////////////////////////

namespace matlab
{

//
// write blas matrix/vector in Matlab format with given name
// - if filename is empty, the matrix/vector name is used
//
template < typename value_t >
void
write ( const blas::matrix< value_t > &  M,
        const std::string &              matname,
        const std::string &              filename = "" )
{
    if ( filename == "" )
        HLIB::DBG::write( M, matname + ".mat", matname );
    else
        HLIB::DBG::write( M, filename, matname );
}

inline
void
write ( const HLIB::TMatrix &  M,
        const std::string &    matname,
        const std::string &    filename = "" )
{
    if ( filename == "" )
        HLIB::DBG::write( M, matname + ".mat", matname );
    else
        HLIB::DBG::write( M, filename, matname );
}

inline
void
write ( const HLIB::TMatrix *  M,
        const std::string &    matname,
        const std::string &    filename = "" )
{
    HLR_ASSERT( ! is_null( M ) );

    write( *M, matname, filename );
}

inline
void
write ( const HLIB::TVector &  v,
        const std::string &    vecname,
        const std::string &    filename = "" )
{
    if ( filename == "" )
        HLIB::DBG::write( &v, vecname + ".mat", vecname );
    else
        HLIB::DBG::write( &v, filename, vecname );
}

template < typename value_t >
void
write ( const blas::vector< value_t > &  v,
        const std::string &              vecname,
        const std::string &              filename = "" )
{
    if ( filename == "" )
        HLIB::DBG::write( v, vecname + ".mat", vecname );
    else
        HLIB::DBG::write( v, filename, vecname );
}

//
// read matrix with given matrix name from given Matlab file
// - if matrix name is empty, first matrix in file is returned
//
template < typename value_t >
blas::matrix< value_t >
read ( const std::string &  filename,
       const std::string &  matname = "" )
{
    HLIB::TMatlabMatrixIO  mio;
    auto                   D = mio.read( filename, matname );

    HLR_ASSERT( is_dense( *D ) );
    
    return std::move( blas::mat< value_t >( ptrcast( D.get(), HLIB::TDenseMatrix ) ) );
}

}// namespace matlab

//////////////////////////////////////////////////////////////////////
//
// HDF5 format
//
//////////////////////////////////////////////////////////////////////

namespace hdf5
{

//
// write blas matrix/vector in Matlab format with given name
// - if filename is empty, the matrix/vector name is used
//
template < typename value_t >
void
write ( const blas::matrix< value_t > &  M,
        const std::string &              matname,
        const std::string &              filename = "" )
{
    HLIB::THDF5MatrixIO  mio;
    
    if ( filename == "" )
        mio.write( M, matname + ".hdf5", matname );
    else
        mio.write( M, filename, matname );
}

inline
void
write ( const HLIB::TMatrix &  M,
        const std::string &    matname,
        const std::string &    filename = "" )
{
    HLIB::THDF5MatrixIO  mio;
    
    if ( filename == "" )
        mio.write( &M, matname + ".hdf5", matname );
    else
        mio.write( &M, filename, matname );
}

inline
void
write ( const HLIB::TMatrix *  M,
        const std::string &    matname,
        const std::string &    filename = "" )
{
    HLR_ASSERT( ! is_null( M ) );

    write( *M, matname, filename );
}

//
// read matrix with given matrix name from given Matlab file
// - if matrix name is empty, first matrix in file is returned
//
template < typename value_t >
blas::matrix< value_t >
read ( const std::string &  filename )
{
    HLIB::THDF5MatrixIO  mio;
    auto                 D = mio.read( filename );

    HLR_ASSERT( is_dense( *D ) );
    
    return std::move( blas::mat< value_t >( ptrcast( D.get(), HLIB::TDenseMatrix ) ) );
}

}// namespace hdf5

//////////////////////////////////////////////////////////////////////
//
// PostScript format
//
//////////////////////////////////////////////////////////////////////

namespace eps
{

//
// print cluster <cl> to file <filename>
//
inline
void
print ( const HLIB::TCluster &  cl,
        const std::string &     filename )
{
    HLIB::TPSClusterVis  vis;

    vis.print( & cl, filename );
}

//
// print blockcluster <cl> to file <filename>
//
inline
void
print ( const HLIB::TBlockCluster &  cl,
        const std::string &          filename )
{
    HLIB::TPSBlockClusterVis  vis;

    vis.print( & cl, filename );
}

//
// print matrix <M> to file <filename>
//
inline
void
print ( const HLIB::TMatrix &  M,
        const std::string &    filename,
        const std::string &    options = "default" )
{
    matrix::print_eps( M, filename, options );
}

//
// print matrix <M> level-wise to files with filename <basename><lvl>
//
inline
void
print_lvl ( const HLIB::TMatrix &  M,
            const std::string &    basename,
            const std::string &    options = "default" )
{
    matrix::print_lvl_eps( M, basename, options );
}

//
// print matrix <M> with blocks coloured according to memory consumption
//
inline
void
print_mem ( const HLIB::TMatrix &  M,
            const std::string &    filename,
            const std::string &    options = "default" )
{
    matrix::print_mem_eps( M, filename, options );
}

//
// print cluster basis <cl> to file <filename>
//
template < typename cluster_basis_t >
void
print ( const cluster_basis_t &  cb,
        const std::string &      filename,
        const std::string &      options = "default" )
{
    hlr::matrix::print_eps( cb, filename, options );
}

}// namespace eps

//////////////////////////////////////////////////////////////////////
//
// VTK format
//
//////////////////////////////////////////////////////////////////////

namespace vtk
{

//
// print coordinates
//
inline
void
print ( const HLIB::TCoordinate &  coord,
        const std::string &        filename )
{
    HLIB::TVTKCoordVis  vis;

    vis.print( & coord, filename );
}

}// namespace vtk

}}// namespace hlr::io

#endif // __HLR_UTILS_IO_HH

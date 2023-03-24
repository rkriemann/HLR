#ifndef __HLR_UTILS_IO_HH
#define __HLR_UTILS_IO_HH
//
// Project     : HLR
// Module      : utils/io
// Description : IO related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <string>
#include <cstring>

#if defined(HLR_USE_HDF5)
#  include <H5Cpp.h>
#endif

#include <hpro/config.h>

#if defined(HPRO_USE_LIC_CHECK)  // hack to test for full HLIBpro
#  define HLR_USE_H2
#endif

#include <hpro/io/TMatrixIO.hh>
#include <hpro/io/TClusterVis.hh>
#include <hpro/io/TCoordVis.hh>

#if defined(HLR_USE_H2)
#  include <hpro/io/TClusterBasisVis.hh>
#  include <hpro/matrix/TUniformMatrix.hh>
#endif

#include <hlr/arith/blas.hh>
#include <hlr/utils/checks.hh>
#include <hlr/matrix/print.hh>
#include <hlr/dag/graph.hh>
#include <hlr/tensor/base_tensor.hh>
#include <hlr/tensor/dense_tensor.hh>

#include <hlr/utils/detail/io.hh>

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
template < typename value_t >
void
write ( const Hpro::TMatrix< value_t > &  M,
        const std::string &               filename )
{
    Hpro::THproMatrixIO  mio;

    mio.write( &M, filename );
}

//
// read matrix M from file <filename>
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
read ( const std::string &  filename )
{
    Hpro::THproMatrixIO  mio;

    return mio.read< value_t >( filename );
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
        Hpro::DBG::write( M, matname + ".mat", matname );
    else
        Hpro::DBG::write( M, filename, matname );
}

template < typename value_t >
void
write ( const Hpro::TMatrix< value_t > &  M,
        const std::string &               matname,
        const std::string &               filename = "" )
{
    if ( filename == "" )
        Hpro::DBG::write( M, matname + ".mat", matname );
    else
        Hpro::DBG::write( M, filename, matname );
}

template < typename value_t >
void
write ( const Hpro::TMatrix< value_t > *  M,
        const std::string &               matname,
        const std::string &               filename = "" )
{
    HLR_ASSERT( ! is_null( M ) );

    write( *M, matname, filename );
}

template < typename value_t >
void
write ( const Hpro::TVector< value_t > &  v,
        const std::string &               vecname,
        const std::string &               filename = "" )
{
    if ( filename == "" )
        Hpro::DBG::write( &v, vecname + ".mat", vecname );
    else
        Hpro::DBG::write( &v, filename, vecname );
}

template < typename value_t >
void
write ( const blas::vector< value_t > &  v,
        const std::string &              vecname,
        const std::string &              filename = "" )
{
    if ( filename == "" )
        Hpro::DBG::write( v, vecname + ".mat", vecname );
    else
        Hpro::DBG::write( v, filename, vecname );
}

//
// read matrix with given matrix name from given file
// - if matrix name is empty, first matrix in file is returned
//
template < typename value_t >
blas::matrix< value_t >
read ( const std::string &  filename,
       const std::string &  matname = "" )
{
    Hpro::TMatlabMatrixIO  mio;
    auto                   D = mio.read< value_t >( filename, matname );

    HLR_ASSERT( is_dense( *D ) );
    
    return std::move( blas::mat( ptrcast( D.get(), Hpro::TDenseMatrix< value_t > ) ) );
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
// write blas matrix/vector in HDF5 format with given name
// - if filename is empty, the matrix/vector name is used
//
template < typename value_t >
void
write ( const blas::matrix< value_t > &  M,
        const std::string &              matname,
        const std::string &              filename = "" )
{
    Hpro::THDF5MatrixIO  mio;
    
    if ( filename == "" )
        mio.write( M, matname + ".hdf5", matname );
    else
        mio.write( M, filename, matname );
}

template < typename value_t >
void
write ( const Hpro::TMatrix< value_t > &  M,
        const std::string &               matname,
        const std::string &               filename = "" )
{
    Hpro::THDF5MatrixIO  mio;
    
    if ( filename == "" )
        mio.write( &M, matname + ".hdf5", matname );
    else
        mio.write( &M, filename, matname );
}

template < typename value_t >
void
write ( const Hpro::TMatrix< value_t > *  M,
        const std::string &               matname,
        const std::string &               filename = "" )
{
    HLR_ASSERT( ! is_null( M ) );

    write( *M, matname, filename );
}

template < typename value_t >
void
write ( const tensor::dense_tensor3< value_t > &  t,
        const std::string &                       tname,
        const std::string &                       fname = "" )
{
    #if defined(HLR_USE_HDF5)

    const std::string  filename = ( fname == "" ? tname + ".h5" : fname );
    auto               file     = H5::H5File( filename, H5F_ACC_TRUNC );
    
    detail::h5_write_tensor( file, "/" + tname, t );

    #endif
}

template < typename value_t >
void
write ( const blas::tensor3< value_t > &  t,
        const std::string &               tname,
        const std::string &               fname = "" )
{
    #if defined(HLR_USE_HDF5)

    const std::string  filename = ( fname == "" ? tname + ".h5" : fname );
    auto               file     = H5::H5File( filename, H5F_ACC_TRUNC );
    
    detail::h5_write_tensor( file, tname, t );

    #endif
}

template < blas::matrix_type  T >
T
read ( const std::string &  filename )
{
    using  value_t = typename T::value_t;
    
    Hpro::THDF5MatrixIO  mio;
    auto                 D = mio.read< value_t >( filename );

    HLR_ASSERT( is_dense( *D ) );
    
    return std::move( blas::mat( ptrcast( D.get(), Hpro::TDenseMatrix< value_t > ) ) );
}

template < blas::tensor_type  T >
T
read ( const std::string &  filename = "" )
{
    #if defined(HLR_USE_HDF5)

    using  value_t = typename T::value_t;
    
    auto  file = H5::H5File( filename, H5F_ACC_RDONLY );

    return  detail::h5_read_blas_tensor< value_t >( file, "" );

    #else

    return T();
    
    #endif
}

template < typename value_t >
tensor::dense_tensor3< value_t >
read ( const std::string &  filename = "" )
{
    #if defined(HLR_USE_HDF5)

    auto  file = H5::H5File( filename, H5F_ACC_RDONLY );

    return  detail::h5_read_tensor< value_t >( file, "" );

    #else

    return tensor::dense_tensor3< value_t >();
    
    #endif
}

}// namespace hdf5

//////////////////////////////////////////////////////////////////////
//
// NetCDF format
//
//////////////////////////////////////////////////////////////////////

namespace h2lib
{

//
// read matrix with given matrix name from given file
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
read ( const std::string &  filename )
{
    Hpro::TH2LibMatrixIO  mio;

    return mio.read< value_t >( filename );
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
print ( const Hpro::TCluster &  cl,
        const std::string &     filename )
{
    Hpro::TPSClusterVis  vis;

    vis.print( & cl, filename );
}

//
// print blockcluster <cl> to file <filename>
//
inline
void
print ( const Hpro::TBlockCluster &  cl,
        const std::string &          filename )
{
    Hpro::TPSBlockClusterVis  vis;

    vis.print( & cl, filename );
}

//
// print matrix <M> to file <filename>
//
template < typename value_t >
void
print ( const Hpro::TMatrix< value_t > &  M,
        const std::string &               filename,
        const std::string &               options = "default" )
{
    matrix::print_eps( M, filename, options );
}

//
// print matrix <M> level-wise to files with filename <basename><lvl>
//
template < typename value_t >
void
print_lvl ( const Hpro::TMatrix< value_t > &  M,
            const std::string &               basename,
            const std::string &               options = "default" )
{
    matrix::print_lvl_eps( M, basename, options );
}

//
// print matrix <M> with blocks coloured according to memory consumption
//
template < typename value_t >
void
print_mem ( const Hpro::TMatrix< value_t > &  M,
            const std::string &               filename,
            const std::string &               options = "default" )
{
    matrix::print_mem_eps( M, filename, options );
}

//
// print cluster basis <cl> to file <filename>
//
template < typename value_t >
void
print ( const hlr::matrix::cluster_basis< value_t > &  cb,
        const std::string &                            filename,
        const std::string &                            options = "default" )
{
    hlr::matrix::print_eps( cb, filename, options );
}

#if defined(HLR_USE_H2)
template < typename value_t >
void
print ( const Hpro::TClusterBasis< value_t > &  cb,
        const std::string &                     filename )
{
    Hpro::TPSClusterBasisVis< value_t >  cbvis;

    cbvis.colourise( false );
    cbvis.print( &cb, filename );
}
#endif

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
print ( const Hpro::TCoordinate &  coord,
        const std::string &        filename )
{
    Hpro::TVTKCoordVis  vis;

    vis.print( & coord, filename );
}

//
// print tensor
//
template < typename value_t >
void
print ( const tensor::base_tensor3< value_t > &  t,
        const std::string &                      filename )
{
    if ( tensor::is_dense( t ) )
        detail::vtk_print_full_tensor( cptrcast( &t, tensor::dense_tensor3< value_t > )->tensor(), filename );
    else
        detail::vtk_print_tensor( t, filename );
}

template < typename value_t >
void
print ( const blas::tensor3< value_t > &  t,
        const std::string &               filename )
{
    detail::vtk_print_full_tensor( t, filename );
}

}// namespace vtk

//////////////////////////////////////////////////////////////////////
//
// GraphViz format
//
//////////////////////////////////////////////////////////////////////

namespace dot
{

//
// print coordinates
//
inline
void
print ( const dag::graph &   graph,
        const std::string &  filename )
{
    graph.print_dot( filename );
}

}// namespace vtk

}}// namespace hlr::io

#endif // __HLR_UTILS_IO_HH

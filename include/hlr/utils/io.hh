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

#if defined(HAS_HDF5)
#  include "H5Cpp.h"
#endif

#include <hpro/config.h>

#if defined(USE_LIC_CHECK)  // hack to test for full HLIBpro
#  define HAS_H2
#endif

#include <hpro/io/TMatrixIO.hh>
#include <hpro/io/TClusterVis.hh>
#include <hpro/io/TCoordVis.hh>

#if defined(HAS_H2)
#  include <hpro/io/TClusterBasisVis.hh>
#  include <hpro/matrix/TUniformMatrix.hh>
#endif

#include <hlr/arith/blas.hh>
#include <hlr/utils/checks.hh>
#include <hlr/matrix/print.hh>
#include <hlr/dag/graph.hh>
#include <hlr/tensor/dense_tensor.hh>

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

namespace detail
{

#if defined(USE_HDF5)

template < typename value_t, int dim >
void
h5_write_tensor ( H5::H5File &                                  file,
                  const std::string &                           gname,
                  const tensor::dense_tensor< value_t, dim > &  t )
{
    auto  group = std::make_unique< H5::Group >( file.createGroup( gname ) );

    if ( Hpro::is_complex_type< value_t >::value )
        HLR_ERROR( "complex not yet supported" );
    
    hsize_t    dims[ dim ];

    for ( uint  i = 0; i < dim; ++i )
        dims[i] = t.dim(i);
    
    H5::DataSpace  dataspace( dim, dims );

    // define datatype
    std::unique_ptr< H5::FloatType >  datatype;
    
    if ( Hpro::is_single_prec_v< value_t > )
        datatype = std::make_unique< H5::FloatType >( H5::PredType::NATIVE_FLOAT );
    else
        datatype = std::make_unique< H5::FloatType >( H5::PredType::NATIVE_DOUBLE );

    // create dataset for tensor data
    H5::DataSet dataset = file.createDataSet( gname + "/" + tname, *datatype, dataspace );
            
    // write the data to the dataset using default memory space, file
    // space, and transfer properties.
    if ( Hpro::is_single_prec_v< value_t > )
        dataset.write( t.data(), H5::PredType::NATIVE_FLOAT );
    else
        dataset.write( t.data(), H5::PredType::NATIVE_DOUBLE );
}

#endif

}// namespace detail

template < typename value_t, int dim >
void
write ( const tensor::dense_tensor< value_t, dim > &  t,
        const std::string &                           tname,
        const std::string &                           fname = "" )
{
    #if defined(USE_HDF5)

    const std::string  filename = ( fname == "" ? tname + ".h5" : fname );
    H5::H5File         file( filename, H5F_ACC_TRUNC );
    
    detail::h5_write_tensor( file, "/" + tname, t );

    #endif
}

//
// read matrix from given file
//
template < typename value_t >
blas::matrix< value_t >
read ( const std::string &  filename )
{
    Hpro::THDF5MatrixIO  mio;
    auto                 D = mio.read< value_t >( filename );

    HLR_ASSERT( is_dense( *D ) );
    
    return std::move( blas::mat( ptrcast( D.get(), Hpro::TDenseMatrix< value_t > ) ) );
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

#if defined(HAS_H2)
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

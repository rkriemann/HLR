#ifndef __HLR_UTILS_IO_HH
#define __HLR_UTILS_IO_HH
//
// Project     : HLR
// Module      : utils/io
// Description : IO related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
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
#include <hpro/io/TGridVis.hh>
#include <hpro/io/TGridIO.hh>

#if defined(HLR_USE_H2)
#  include <hpro/io/TClusterBasisVis.hh>
#  include <hpro/matrix/TUniformMatrix.hh>
#endif

#include <hlr/arith/blas.hh>
#include <hlr/utils/checks.hh>
#include <hlr/matrix/print.hh>
#include <hlr/matrix/shared_cluster_basis.hh>
#include <hlr/matrix/nested_cluster_basis.hh>
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

//
// write matrix M to file <filename>
//
inline
void
write ( const Hpro::TGrid &  grid,
        const std::string &  filename )
{
    Hpro::THproGridIO  gio;

    gio.write( &grid, filename );
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
        mio.write( M, matname + ".h5", matname );
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
        mio.write( &M, matname + ".h5", matname );
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

    #else

    HLR_ERROR( "no HDF5 support available" );
    
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

    #else

    HLR_ERROR( "no HDF5 support available" );
    
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

    HLR_ERROR( "no HDF5 support available" );
    
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

    HLR_ERROR( "no HDF5 support available" );
    
    return tensor::dense_tensor3< value_t >();
    
    #endif
}

}// namespace hdf5

//////////////////////////////////////////////////////////////////////
//
// raw data
//
//////////////////////////////////////////////////////////////////////

namespace raw
{

template < typename value_t >
blas::tensor3< value_t >
read ( const std::string &  filename,
       const size_t         size0,
       const size_t         size1,
       const size_t         size2 )
{
    auto  file = std::ifstream( filename, std::ios::in | std::ios::binary );
    auto  t    = blas::tensor3< value_t >( size0, size1, size2 );

    file.read( reinterpret_cast< char * >( t.data() ), sizeof(value_t) * size0 * size1 * size2 );

    return  t;
}

template < typename value_t >
void
write ( const blas::tensor3< value_t > &  t,
        const std::string &               filename = "" )
{
    auto  file = std::ofstream( filename, std::ios::out | std::ios::binary );

    file.write( reinterpret_cast< const char * >( t.data() ), sizeof(value_t) * t.size(0) * t.size(1) * t.size(2) );
}

}// namespace raw

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
void
print ( const Hpro::TBlockCluster &  cl,
        const std::string &          filename,
        const std::string &          options = "default" );

//
// print matrix <M> to file <filename>
//
// options:  nosize    - do not print size information in blocks
//           pattern   - print non-zero coefficient pattern
//           norank    - do not print rank of blocks
//           nonempty  - only print nonempty blocks
//           indexset  - print index set data per block
//           noid      - do not print id of block
//           noinnerid - do not print id of non-leaf blocks
//
// Multiple options need to be comma separated.
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
print ( const hlr::matrix::shared_cluster_basis< value_t > &  cb,
        const std::string &                                   filename,
        const std::string &                                   options = "default" )
{
    hlr::matrix::print_eps( cb, filename, options );
}

template < typename value_t >
void
print ( const hlr::matrix::nested_cluster_basis< value_t > &  cb,
        const std::string &                                   filename,
        const std::string &                                   options = "default" )
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

inline
void
print ( const Hpro::TGrid &  grid,
        const std::string &  filename )
{
    Hpro::TVTKGridVis  vis;

    vis.print( & grid, filename );
}

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

inline
void
print ( const Hpro::TCoordinate &    coord,
        const std::vector< uint > &  label,
        const std::string &          filename )
{
    Hpro::TVTKCoordVis  vis;

    vis.print( & coord, label, filename );
}

//
// print (bounding boxes of geometric) clusters
// - up to <lvl> sub levels are printed
//
inline
void
print ( const Hpro::TCluster &  cl,
        const uint              lvl,
        const std::string &     filename )
{
    detail::vtk_print_cluster( cl, lvl, filename );
}

//
// print coordinates labeled according to clusters
// in cluster tree on a particular level
//
inline
void
print ( const Hpro::TCoordinate &   coord,
        const Hpro::TCluster &      ct,
        const Hpro::TPermutation &  pi2e,
        const uint                  lvl,
        const std::string &         filename )
{
    detail::vtk_print_cluster( coord, ct, pi2e, lvl, filename );
}

//
// print bounding box and coordinates of given block cluster
//
inline
void
print ( const Hpro::TBlockCluster &  bc,
        const Hpro::TCoordinate &    coord,
        const Hpro::TPermutation &   pi2e,
        const std::string &          filename )
{
    detail::vtk_print_cluster( bc, coord, pi2e, filename );
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

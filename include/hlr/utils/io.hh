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

#if defined(HAS_HDF5)
#  include <H5Cpp.h>
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

template < typename value_t, uint dim >
void
h5_write_tensor ( H5::H5File &                                  file,
                  const std::string &                           gname,
                  const tensor::dense_tensor< value_t, dim > &  t )
{
    auto  group = std::make_unique< H5::Group >( file.createGroup( gname ) );

    if ( Hpro::is_complex_type< value_t >::value )
        HLR_ERROR( "complex not yet supported" );

    //
    // indexset info
    //

    {
        auto     buf         = std::vector< char >( sizeof(int) + sizeof(int) + dim * 2 * sizeof(long) );
        hsize_t  data_dims[] = { 1 };
        auto     data_space  = H5::DataSpace( 1, data_dims );
        auto     data_type   = H5::CompType( buf.size() );
        size_t   ofs         = 0;
        int      id          = t.id();
        int      bdim        = dim;
        
        data_type.insertMember( "id", ofs, H5::PredType::NATIVE_INT );
        std::memcpy( buf.data() + ofs,  & id, sizeof(id) );
        ofs += sizeof(id);

        data_type.insertMember( "dim", ofs, H5::PredType::NATIVE_INT );
        std::memcpy( buf.data() + ofs,  & bdim, sizeof(bdim) );
        ofs += sizeof(bdim);

        hsize_t  is_dims = 2*dim;
        auto     is_type = H5::ArrayType( H5::PredType::NATIVE_LONG, 1, & is_dims );
        
        data_type.insertMember( "indexsets", ofs, is_type );

        for ( uint  d = 0; d < dim; ++d )
        {
            const auto  is         = t.is( d );
            const long  is_data[2] = { is.first(), is.last() };
        
            std::memcpy( buf.data() + ofs, is_data, sizeof(is_data) );
            ofs += sizeof(is_data);
        }// for
    
        auto  data_set = file.createDataSet( gname + "/structure", data_type, data_space );

        data_set.write( buf.data(), data_type );

        data_space.close();
        data_type.close();
        data_set.close();
    }

    //
    // tensor data
    //

    {
        hsize_t  data_dims[ dim ];

        for ( uint  i = 0; i < dim; ++i )
            data_dims[i] = t.dim(i);
    
        auto  data_space = H5::DataSpace( dim, data_dims );

        // define datatype
        auto  data_type = ( Hpro::is_single_prec_v< value_t > ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE );

        // create dataset for tensor data
        auto  data_set = file.createDataSet( gname + "/data", data_type, data_space );
            
        data_set.write( t.data(), data_type );

        data_space.close();
        data_type.close();
        data_set.close();
    }
}

herr_t
visit_func ( hid_t               /* loc_id */,
             const char *        name,
             const H5O_info_t *  info,
             void *              operator_data )
{
    std::string *  dname = static_cast< std::string * >( operator_data );
    std::string    oname = name;

    if ( oname[0] != '.' )
    {
        if ( info->type == H5O_TYPE_GROUP )
        {
            // use first name encountered
            if ( *dname == "" )
                *dname = oname;
        }// if
        else if ( info->type == H5O_TYPE_DATASET )
        {
            if ( *dname != "" )
            {
                if ( oname == *dname + "/data" )     // actual dataset
                    *dname = *dname;
                else if ( oname == *dname + "/structure" )
                    *dname = *dname;
                else
                    *dname = "";
            }// if
            else
            {
                *dname = oname;                       // directly use dataset
            }// else
        }// if
    }// if
    
    return 0;
}

template < typename value_t, uint dim >
tensor::dense_tensor< value_t, dim >
h5_read_tensor ( H5::H5File &         file,
                 const std::string &  gname )
{
    auto  data_name = std::string( "" );
    auto  status    = H5Ovisit( file.getId(), H5_INDEX_NAME, H5_ITER_INC, visit_func, & data_name );

    HLR_ASSERT( status == 0 );

    // check if nothing compatible was found
    if ( data_name == "" )
        return tensor::dense_tensor< value_t, dim >();

    //
    // read structural info
    //

    int   id = -1;
    auto  is = std::array< indexset, dim >();
    
    {
        auto    data_set   = file.openDataSet( data_name + "/structure" );
        auto    data_space = data_set.getSpace();

        auto    buf        = std::vector< char >( sizeof(int) + sizeof(int) + dim * 2 * sizeof(long) );
        auto    data_type  = H5::CompType( buf.size() );
        size_t  ofs        = 0;

        data_type.insertMember( "id",  ofs, H5::PredType::NATIVE_INT ); ofs += sizeof(int);
        data_type.insertMember( "dim", ofs, H5::PredType::NATIVE_INT ); ofs += sizeof(int);

        hsize_t  is_dims = 2*dim;
        auto     is_type = H5::ArrayType( H5::PredType::NATIVE_LONG, 1, & is_dims );
        
        data_type.insertMember( "indexsets", ofs, is_type );

        data_set.read( buf.data(), data_type );

        // extract data
        int  ddim = 0;
        
        ofs = 0;
        std::memcpy( &id,   buf.data() + ofs, sizeof(int) ); ofs += sizeof(int);
        std::memcpy( &ddim, buf.data() + ofs, sizeof(int) ); ofs += sizeof(int);

        HLR_ASSERT( dim == ddim );

        long  is_data[2*dim];
        
        std::memcpy( is_data, buf.data() + ofs, sizeof(is_data) ); ofs += sizeof(is_data);
        
        for ( uint  d = 0; d < dim; ++d )
            is[d] = indexset( is_data[2*d], is_data[2*d+1] );
    }

    //
    // read tensor data
    //

    auto  t = tensor::dense_tensor< value_t, dim >( is );
    
    {
        auto  data_set   = file.openDataSet( data_name + "/data" );
        auto  data_space = data_set.getSpace();
        auto  data_type  = ( Hpro::is_single_prec_v< value_t > ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE );
        auto  dims       = std::vector< hsize_t >( dim );

        HLR_ASSERT( dim == data_space.getSimpleExtentNdims() );

        data_space.getSimpleExtentDims( dims.data() );

        for ( uint  d = 0; d < dim; ++d )
        {
            HLR_ASSERT( dims[d] == is[d].size() );
        }// for
        
        data_set.read( t.data(), data_type );
    }
    
    return t;
}

#endif

}// namespace detail

template < typename value_t, uint dim >
void
write ( const tensor::dense_tensor< value_t, dim > &  t,
        const std::string &                           tname,
        const std::string &                           fname = "" )
{
    #if defined(USE_HDF5)

    const std::string  filename = ( fname == "" ? tname + ".h5" : fname );
    auto               file     = H5::H5File( filename, H5F_ACC_TRUNC );
    
    detail::h5_write_tensor( file, "/" + tname, t );

    #endif
}

template < typename value_t, uint dim >
tensor::dense_tensor< value_t, dim >
read_tensor ( const std::string &  filename = "" )
{
    #if defined(USE_HDF5)

    auto  file = H5::H5File( filename, H5F_ACC_RDONLY );

    return  detail::h5_read_tensor< value_t, dim >( file, "" );

    #else

    return tensor::dense_tensor< value_t, dim >();
    
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

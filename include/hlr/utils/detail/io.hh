#ifndef __HLR_UTILS_DETAIL_IO_HH
#define __HLR_UTILS_DETAIL_IO_HH
//
// Project     : HLR
// Module      : utils/io
// Description : IO related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <fstream>
#include <filesystem>
#include <array>

#include <hpro/cluster/TGeomCluster.hh>

#include <hlr/tensor/structured_tensor.hh>
#include <hlr/tensor/tucker_tensor.hh>
#include <hlr/tensor/dense_tensor.hh>

#include <hlr/utils/tools.hh>

namespace hlr { namespace io {

namespace detail
{

//////////////////////////////////////////////////////////////////////
//
// HDF5 format
//
//////////////////////////////////////////////////////////////////////

#if defined(HLR_USE_HDF5)

template < typename value_t >
void
h5_write_tensor ( H5::H5File &                              file,
                  const std::string &                       gname,
                  const tensor::dense_tensor3< value_t > &  t )
{
    constexpr uint  dim = 3;

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
            data_dims[i] = t.dim(2-i);
    
        auto  data_space = H5::DataSpace( dim, data_dims );

        // define datatype
        auto  data_type = ( Hpro::is_single_prec_v< value_t > ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE );

        // create dataset for tensor data
        auto  data_set = file.createDataSet( gname + "/data", data_type, data_space );
            
        data_set.write( t.tensor().data(), data_type );

        data_space.close();
        data_type.close();
        data_set.close();
    }
}

template < typename value_t >
void
h5_write_tensor ( H5::H5File &                      file,
                  const std::string &               tname,
                  const blas::tensor3< value_t > &  t )
{
    constexpr uint  dim = 3;

    if ( Hpro::is_complex_type< value_t >::value )
        HLR_ERROR( "complex not yet supported" );

    //
    // just tensor data
    //

    hsize_t  data_dims[ dim ];

    for ( uint  i = 0; i < dim; ++i )
        data_dims[i] = t.size(2-i);
    
    auto  data_space = H5::DataSpace( dim, data_dims );
    auto  data_type  = ( Hpro::is_single_prec_v< value_t > ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE );
    auto  data_set   = file.createDataSet( "/" + tname, data_type, data_space );
    
    data_set.write( t.data(), data_type );

    data_space.close();
    data_type.close();
    data_set.close();
}

inline
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

template < typename value_t >
tensor::dense_tensor3< value_t >
h5_read_tensor ( H5::H5File &         file,
                 const std::string &  gname )
{
    constexpr uint  dim = 3;

    auto  data_name = std::string( "" );
    auto  status    = H5Ovisit( file.getId(), H5_INDEX_NAME, H5_ITER_INC, visit_func, & data_name );

    HLR_ASSERT( status == 0 );

    // check if nothing compatible was found
    if ( data_name == "" )
        return tensor::dense_tensor3< value_t >();

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

    auto  t = tensor::dense_tensor3< value_t >( is );
    
    {
        auto  data_set   = file.openDataSet( data_name + "/data" );
        auto  data_space = data_set.getSpace();
        auto  data_type  = ( Hpro::is_single_prec_v< value_t > ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE );
        auto  dims       = std::vector< hsize_t >( dim );

        HLR_ASSERT( dim == data_space.getSimpleExtentNdims() );

        data_space.getSimpleExtentDims( dims.data() );

        for ( uint  d = 0; d < dim; ++d )
        {
            HLR_ASSERT( dims[2-d] == is[d].size() );
        }// for
        
        data_set.read( t.tensor().data(), data_type );
    }
    
    return t;
}

template < typename value_t >
blas::tensor3< value_t >
h5_read_blas_tensor ( H5::H5File &         file,
                      const std::string &  tname )
{
    constexpr uint  dim = 3;

    auto  data_name = std::string( "" );
    auto  status    = H5Ovisit( file.getId(), H5_INDEX_NAME, H5_ITER_INC, visit_func, & data_name );

    HLR_ASSERT( status == 0 );
    
    // check if nothing compatible was found
    if ( data_name == "" )
        return blas::tensor3< value_t >();

    //
    // read tensor data
    //

    auto  data_set   = file.openDataSet( data_name );
    auto  data_space = data_set.getSpace();
    auto  data_type  = ( Hpro::is_single_prec_v< value_t > ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE );
    auto  dims       = std::vector< hsize_t >( dim );

    HLR_ASSERT( dim == data_space.getSimpleExtentNdims() );

    data_space.getSimpleExtentDims( dims.data() );
    
    auto  t = blas::tensor3< value_t >( dims[2], dims[1], dims[0] );
    
    data_set.read( t.data(), data_type );
    
    return t;
}

#endif // HLR_USE_HDF5

//////////////////////////////////////////////////////////////////////
//
// VTK format
//
//////////////////////////////////////////////////////////////////////

inline
void
vtk_print_cluster ( const Hpro::TCluster &  cl,
                    const uint              nlvl,
                    const std::string &     filename )
{
    //
    // collecting clusters for visualization
    //

    auto  current = std::list< const Hpro::TGeomCluster * >();
    uint  lvl     = 0;

    HLR_ASSERT( Hpro::is_geom_cluster( cl ) );
    
    current.push_back( cptrcast( &cl, Hpro::TGeomCluster ) );
    
    while ( lvl < nlvl )
    {
        auto  sons = std::list< const Hpro::TGeomCluster * >();

        for ( auto  cluster : current )
        {
            for ( uint  i = 0; i < cluster->nsons(); ++i )
            {
                auto  son_i = cluster->son(i);

                if ( is_null( son_i ) )
                    continue;

                HLR_ASSERT( Hpro::is_geom_cluster( son_i ) );
                    
                sons.push_back( cptrcast( son_i, Hpro::TGeomCluster ) );
            }// for
        }// for

        current = std::move( sons );
        ++lvl;
    }// while

    auto  clusters = std::move( current );
    
    std::cout << clusters.size() << std::endl;
    
    //
    // print bounding boxes
    //
    // vertex order:
    //
    //     6───────7
    //    ╱│      ╱│
    //   4─┼─────5 │
    //   │ 2─────┼─3
    //   │╱      │╱
    //   0───────1
    //
    
    auto  outname = std::filesystem::path( filename );
    auto  out     = std::ofstream( outname.has_extension() ? filename : filename + ".vtk", std::ios::binary );
    
    out << "# vtk DataFile Version 2.0" << std::endl
        << "HLR cluster tree" << std::endl
        << "ASCII" << std::endl
        << "DATASET UNSTRUCTURED_GRID" << std::endl;

    const size_t  nc = clusters.size();

    if constexpr ( std::same_as< Hpro::TBoundingVolume, Hpro::TBBox > )
    {
        out << "POINTS " << 8*nc << " FLOAT" << std::endl;

        for ( auto  cluster : clusters )
        {
            auto  bbmin = cluster->bvol().min();
            auto  bbmax = cluster->bvol().max();
            
            HLR_ASSERT(( bbmin.dim() == 3 ) && ( bbmax.dim() == 3 ));
        
            out << bbmin[0] << ' ' << bbmin[1] << ' ' << bbmin[2] << std::endl
                << bbmax[0] << ' ' << bbmin[1] << ' ' << bbmin[2] << std::endl
                << bbmin[0] << ' ' << bbmax[1] << ' ' << bbmin[2] << std::endl
                << bbmax[0] << ' ' << bbmax[1] << ' ' << bbmin[2] << std::endl
                << bbmin[0] << ' ' << bbmin[1] << ' ' << bbmax[2] << std::endl
                << bbmax[0] << ' ' << bbmin[1] << ' ' << bbmax[2] << std::endl
                << bbmin[0] << ' ' << bbmax[1] << ' ' << bbmax[2] << std::endl
                << bbmax[0] << ' ' << bbmax[1] << ' ' << bbmax[2] << std::endl;
        }// for
        
        out << "CELLS " << nc << ' ' << 9 * nc << std::endl;
        
        for ( size_t  i = 0; i < nc; ++i )
            out << "8 "
                << 8*i   << ' '
                << 8*i+1 << ' '
                << 8*i+2 << ' '
                << 8*i+3 << ' '
                << 8*i+4 << ' '
                << 8*i+5 << ' '
                << 8*i+6 << ' '
                << 8*i+7 << std::endl;
        
        out << "CELL_TYPES " << nc << std::endl;
        
        for ( size_t  i = 0; i < nc; ++i )
            out << "11 ";
        out << std::endl;
        
        out << "CELL_DATA " << nc << std::endl
            << "COLOR_SCALARS label 1" << std::endl;
    }// if
    
    uint  label = 1;

    for ( auto  cluster : clusters )
    {
        out << label << " ";
        ++label;
    }// for
    out << std::endl;
}

inline
void
vtk_print_cluster ( const Hpro::TCoordinate &   coord,
                    const Hpro::TCluster &      cl,
                    const Hpro::TPermutation &  i2e,
                    const uint                  nlvl,
                    const std::string &         filename )
{
    //
    // collecting clusters
    //

    auto  current  = std::list< const Hpro::TGeomCluster * >();
    uint  lvl      = 0;

    HLR_ASSERT( Hpro::is_geom_cluster( cl ) );
    
    current.push_back( cptrcast( &cl, Hpro::TGeomCluster ) );
    
    while ( lvl < nlvl )
    {
        auto  sons = std::list< const Hpro::TGeomCluster * >();

        for ( auto  cluster : current )
        {
            for ( uint  i = 0; i < cluster->nsons(); ++i )
            {
                auto  son_i = cluster->son(i);

                if ( is_null( son_i ) )
                    continue;

                HLR_ASSERT( Hpro::is_geom_cluster( son_i ) );
                    
                sons.push_back( cptrcast( son_i, Hpro::TGeomCluster ) );
            }// for
        }// for

        current = std::move( sons );
        ++lvl;
    }// while

    auto    clusters = std::move( current );
    size_t  ncoord   = 0;
    
    for ( auto  cluster : clusters )
        ncoord += cluster->size();

    //
    // print (labeled) coordinates
    //

    auto  outname = std::filesystem::path( filename );
    auto  out     = std::ofstream( outname.has_extension() ? filename : filename + ".vtk", std::ios::binary );
    
    out << "# vtk DataFile Version 2.0" << std::endl
        << "HLR coordinates" << std::endl
        << "ASCII" << std::endl
        << "DATASET UNSTRUCTURED_GRID" << std::endl
        << "POINTS " << ncoord << " FLOAT" << std::endl;

    for ( auto  cluster : clusters )
    {
        for ( idx_t  idx = cluster->first(); idx <= cluster->last(); ++idx )
        {
            const auto  pidx = i2e.permute( idx );
            const auto  vtx  = coord.coord( pidx );

            out << vtx[0] << " " << vtx[1] << " " << vtx[2] << std::endl;
        }// for
    }// for

    out << "CELLS " << ncoord << " " << 2 * ncoord << std::endl;
    
    for ( size_t  i = 0; i < ncoord; ++i )
        out << "1 " << i << " ";

    out << std::endl;

    out << "CELL_TYPES " << ncoord << std::endl;

    for ( size_t  i = 0; i < ncoord; ++i )
        out << "1 ";

    out << std::endl;

    out << "CELL_DATA " << ncoord << std::endl
        << "COLOR_SCALARS label 1" << std::endl;
    
    uint  label = 1;

    for ( auto  cluster : clusters )
    {
        for ( idx_t  idx = cluster->first(); idx <= cluster->last(); ++idx )
            out << label << " ";
        out << std::endl;

        ++label;
    }// for
}

inline
void
vtk_print_cluster ( const Hpro::TBlockCluster &  bc,
                    const Hpro::TCoordinate &    coord,
                    const Hpro::TPermutation &   pi2e,
                    const std::string &          filename )
{
    //
    // compute bounding boxes of row/column clusters
    //

    auto  dim   = coord.dim();
    auto  rowcl = bc.rowcl();
    auto  colcl = bc.colcl();
    auto  rmin  = Hpro::TPoint( dim, coord.coord( pi2e.permute( rowcl->first() ) ) );
    auto  rmax  = Hpro::TPoint( dim, coord.coord( pi2e.permute( rowcl->first() ) ) );
    auto  cmin  = Hpro::TPoint( dim, coord.coord( pi2e.permute( colcl->first() ) ) );
    auto  cmax  = Hpro::TPoint( dim, coord.coord( pi2e.permute( colcl->first() ) ) );

    for ( idx_t  i = rowcl->first(); i <= rowcl->last(); ++i )
    {
        auto  c_i = coord.coord( pi2e.permute( i ) );

        for ( uint  j = 0; j < dim; ++j )
        {
            rmin[j] = std::min( rmin[j], c_i[j] );
            rmax[j] = std::max( rmax[j], c_i[j] );
        }// for
    }// for
    
    for ( idx_t  i = colcl->first(); i <= colcl->last(); ++i )
    {
        auto  c_i = coord.coord( pi2e.permute( i ) );

        for ( uint  j = 0; j < dim; ++j )
        {
            cmin[j] = std::min( cmin[j], c_i[j] );
            cmax[j] = std::max( cmax[j], c_i[j] );
        }// for
    }// for
    
    //
    // print (labeled) coordinates
    //

    auto  outname = std::filesystem::path( filename );
    auto  out     = std::ofstream( outname.has_extension() ? filename : filename + ".vtk", std::ios::binary );
    auto  ncoord  = rowcl->size() + colcl->size();
    
    out << "# vtk DataFile Version 2.0" << std::endl
        << "HLR coordinates" << std::endl
        << "ASCII" << std::endl
        << "DATASET UNSTRUCTURED_GRID" << std::endl
        << "POINTS " << ncoord + 2*8 << " FLOAT" << std::endl;

    // first the bounding boxes
    if ( dim == 3 )
    {
        out << rmin[0] << ' ' << rmin[1] << ' ' << rmin[2] << std::endl
            << rmax[0] << ' ' << rmin[1] << ' ' << rmin[2] << std::endl
            << rmin[0] << ' ' << rmax[1] << ' ' << rmin[2] << std::endl
            << rmax[0] << ' ' << rmax[1] << ' ' << rmin[2] << std::endl
            << rmin[0] << ' ' << rmin[1] << ' ' << rmax[2] << std::endl
            << rmax[0] << ' ' << rmin[1] << ' ' << rmax[2] << std::endl
            << rmin[0] << ' ' << rmax[1] << ' ' << rmax[2] << std::endl
            << rmax[0] << ' ' << rmax[1] << ' ' << rmax[2] << std::endl;

        out << cmin[0] << ' ' << cmin[1] << ' ' << cmin[2] << std::endl
            << cmax[0] << ' ' << cmin[1] << ' ' << cmin[2] << std::endl
            << cmin[0] << ' ' << cmax[1] << ' ' << cmin[2] << std::endl
            << cmax[0] << ' ' << cmax[1] << ' ' << cmin[2] << std::endl
            << cmin[0] << ' ' << cmin[1] << ' ' << cmax[2] << std::endl
            << cmax[0] << ' ' << cmin[1] << ' ' << cmax[2] << std::endl
            << cmin[0] << ' ' << cmax[1] << ' ' << cmax[2] << std::endl
            << cmax[0] << ' ' << cmax[1] << ' ' << cmax[2] << std::endl;
    }// if

    // now the coordinates in the clusters
    for ( idx_t  i = rowcl->first(); i <= rowcl->last(); ++i )
    {
        auto  vtx = coord.coord( pi2e.permute( i ) );

        if ( dim == 3 )
            out << vtx[0] << " " << vtx[1] << " " << vtx[2] << std::endl;
    }// for
    
    for ( idx_t  i = colcl->first(); i <= colcl->last(); ++i )
    {
        auto  vtx = coord.coord( pi2e.permute( i ) );

        if ( dim == 3 )
            out << vtx[0] << " " << vtx[1] << " " << vtx[2] << std::endl;
    }// for

    out << "CELLS " << ncoord + 2 << " " << 2 * ncoord + 2*9 << std::endl;
    
    out << "8 0 1 2 3 4 5 6 7" << std::endl;
    out << "8 8 9 10 11 12 13 14 15" << std::endl;
    
    for ( size_t  i = 0; i < ncoord; ++i )
        out << "1 " << i << " ";

    out << std::endl;

    out << "CELL_TYPES " << ncoord + 2 << std::endl;

    out << "11 11 ";
    
    for ( size_t  i = 0; i < ncoord; ++i )
        out << "1 ";

    out << std::endl;

    out << "CELL_DATA " << ncoord + 2 << std::endl
        << "COLOR_SCALARS label 1" << std::endl;
    
    out << 1 << ' ' << 2 << ' ';
    
    for ( idx_t  i = rowcl->first(); i <= rowcl->last(); ++i )
        out << 1 << ' ';
    
    for ( idx_t  i = colcl->first(); i <= colcl->last(); ++i )
        out << 2 << ' ';

    out << std::endl;
}

template < typename value_t >
void
vtk_print_full_tensor ( const blas::tensor3< value_t > &  t,
                        const std::string &               filename )
{
    auto  outname = std::filesystem::path( filename );
    auto  out     = std::ofstream( outname.has_extension() ? filename : filename + ".vtk", std::ios::binary );

    #if 1
    
    //
    // assuming tensor grid in equal step width in all dimensions
    //

    const double  h = 1.0 / ( std::max({ t.size(0), t.size(1), t.size(2) }) - 1 );

    // out << "# vtk DataFile Version 2.0" << std::endl
    //     << "HLR full tensor" << std::endl
    //     << "ASCII" << std::endl
    //     << "DATASET STRUCTURED_POINTS" << std::endl
    //     << "DIMENSIONS " << t.size(0) << ' ' << t.size(1) << ' ' << t.size(2) << std::endl
    //     << "SPACING " << h << ' ' << h << ' ' << h << std::endl
    //     << "ORIGIN 0 0 0" << std::endl
    //     << "POINT_DATA " << t.size(0) * t.size(1) * t.size(2) << std::endl
    //     << "SCALARS v float 1" << std::endl
    //     << "LOOKUP_TABLE default" << std::endl;

    // constexpr size_t  bufsize = 65536;
    // std::string       buffer;

    // buffer.reserve( bufsize );
    
    // for ( size_t  l = 0; l < t.size(2); ++l )
    //     for ( size_t  j = 0; j < t.size(1); ++j )
    //         for ( size_t  i = 0; i < t.size(0); ++i )
    //         {
    //             std::ostringstream  oss;
                
    //             oss << t( i, j, l ) << ' ';

    //             if ( buffer.length() + oss.str().length() >= bufsize )
    //             {
    //                 out << buffer;
    //                 buffer.resize( 0 );
    //             }// if

    //             buffer.append( oss.str() );
    //         }// for

    // out << buffer << std::endl;
                                                                         
    out << "# vtk DataFile Version 2.0" << std::endl
        << "HLR full tensor" << std::endl
        << "BINARY" << std::endl
        << "DATASET STRUCTURED_POINTS" << std::endl
        << "DIMENSIONS " << t.size(0) << ' ' << t.size(1) << ' ' << t.size(2) << std::endl
        << "ORIGIN 0 0 0" << std::endl
        << "SPACING " << h << ' ' << h << ' ' << h << std::endl
        << "POINT_DATA " << t.size(0) * t.size(1) * t.size(2) << std::endl
        << "SCALARS v double 1" << std::endl
        << "LOOKUP_TABLE default" << std::endl;

    //
    // test endianess (VTK needs big endian)
    //

    bool  is_big_endian = false;

    {
        int16_t  endian = 0x1234;
        char *   ptr    = reinterpret_cast< char * >( &endian );
        
        if (( ptr[0] == 0x12 ) && ( ptr[1] == 0x34 ))
            is_big_endian = true;
    }

    if ( is_big_endian )
    {
        //
        // write all directly
        //
        
        out.write( reinterpret_cast< const char * >( t.data() ), t.size(0) * t.size(1) * t.size(2) * sizeof(value_t) );
    }// if
    else
    {
        //
        // change endianess and write row by row
        //
        
        auto  buf           = std::vector< double >( t.size(0) );
        auto  change_endian = [] ( const double  d )
        {
            double  f = d;
            char *  p = reinterpret_cast< char * >( &f );

            std::swap( p[0], p[7] );
            std::swap( p[1], p[6] );
            std::swap( p[2], p[5] );
            std::swap( p[3], p[4] );

            return f;
        };
    
        for ( size_t  l = 0; l < t.size(2); ++l )
        {
            for ( size_t  j = 0; j < t.size(1); ++j )
            {
                for ( size_t  i = 0; i < t.size(0); ++i )
                    buf[i] = change_endian( t(i,j,l) );

                out.write( reinterpret_cast< const char * >( buf.data() ), t.size(0) * sizeof(value_t) );
            }// for
        }// for
    }// else
    
    #else
    
    out << "# vtk DataFile Version 2.0" << std::endl
        << "HLR full tensor" << std::endl
        << "ASCII" << std::endl
        << "DATASET UNSTRUCTURED_GRID" << std::endl;

    //
    // assuming tensor grid in equal step width in all dimensions
    //

    const size_t  nc  = t.size(0) * t.size(1) * t.size(2);
    const size_t  nv0 = t.size(0)+1;
    const size_t  nv1 = t.size(1)+1;
    const size_t  nv2 = t.size(2)+1;
    const size_t  nv  = nv0 * nv1 * nv2;
    const double  h   = 1.0 / std::min({ t.size(0), t.size(1), t.size(2) });

    out << "POINTS " << nv << " FLOAT" << std::endl;

    for ( size_t  l = 0; l < nv2; ++l )
        for ( size_t  j = 0; j < nv1; ++j )
            for ( size_t  i = 0; i < nv0; ++i )
                out << i * h << ' ' << j * h << ' ' << l * h << std::endl;

    //
    //     6───────7
    //    ╱│      ╱│
    //   4─┼─────5 │
    //   │ 2─────┼─3
    //   │╱      │╱
    //   0───────1
    //
            
    out << "CELLS " << nc << ' ' << 9 * nc << std::endl;

    for ( size_t  l = 0; l < t.size(2); ++l )
        for ( size_t  j = 0; j < t.size(1); ++j )
            for ( size_t  i = 0; i < t.size(0); ++i )
                out << "8 "
                    << ( l * nv1 + j ) * nv0 + i
                    << ' '
                    << ( l * nv1 + j ) * nv0 + (i + 1)
                    << ' '
                    << ( l * nv1 + (j+1) ) * nv0 + i
                    << ' '
                    << ( l * nv1 + (j+1) ) * nv0 + (i+1)
                    << ' '
                    << ( (l+1) * nv1 + j ) * nv0 + i
                    << ' '
                    << ( (l+1) * nv1 + j ) * nv0 + (i+1)
                    << ' '
                    << ( (l+1) * nv1 + (j+1) ) * nv0 + i
                    << ' '
                    << ( (l+1) * nv1 + (j+1) ) * nv0 + (i+1)
                    << std::endl;
        
    out << "CELL_TYPES " << nc << std::endl;
        
    for ( size_t  i = 0; i < nc; ++i )
        out << "11 ";
    out << std::endl;

    //
    // associate entry in tensor with cell in grid
    //

    out << "CELL_DATA " << nc << std::endl
        << "COLOR_SCALARS v" << " 1" << std::endl;
        
    for ( size_t  l = 0; l < t.size(2); ++l )
        for ( size_t  j = 0; j < t.size(1); ++j )
            for ( size_t  i = 0; i < t.size(0); ++i )
                out << t( i, j, l ) << ' ';
    out << std::endl;

    #endif
}

template < typename value_t >
void
vtk_print_tensor ( const tensor::base_tensor3< value_t > &  t,
                   const std::string &                      filename )
{
    //
    // go through tensor and collect vertices and cell indices
    //

    using  coord_t  = std::array< double, 3 >;
    using  voxel_t  = std::array< size_t, 8 >;
    using  tensor_t = tensor::base_tensor3< value_t >;
    
    const size_t  n = t.dim(0) * t.dim(1) * t.dim(2);
    const double  h = 1.0 / std::min({ t.dim(0), t.dim(1), t.dim(2) });
    
    auto  coords = std::deque< coord_t >();
    auto  voxels = std::deque< voxel_t >();
    auto  colors = std::deque< int >();
    
    size_t  ncoord = 0;
    size_t  nvoxel = 0;

    auto  tensors = std::list< const tensor_t * >{ &t };

    while ( ! tensors.empty() )
    {
        auto  T = behead( tensors );
        
        if ( tensor::is_structured( *T ) )
        {
            auto  B = cptrcast( T, tensor::structured_tensor3< value_t > );

            for ( uint  l = 0; l < B->nblocks(2); ++l )
                for ( uint  j = 0; j < B->nblocks(1); ++j )
                    for ( uint  i = 0; i < B->nblocks(0); ++i )
                        if ( ! is_null( B->block(i,j,l) ) )
                            tensors.push_back( B->block(i,j,l) );
        }// if
        else
        {
            //
            //     6-------7
            //    /|      /|
            //   4-------5 |
            //   | 2-----|-3
            //   |/      |/
            //   0 ----- 1
            //
            
            coords.push_back( { T->is(0).first() * h, T->is(1).first() * h, T->is(2).first() * h } );
            coords.push_back( { T->is(0).last()  * h, T->is(1).first() * h, T->is(2).first() * h } );
            coords.push_back( { T->is(0).first() * h, T->is(1).last()  * h, T->is(2).first() * h } );
            coords.push_back( { T->is(0).last()  * h, T->is(1).last()  * h, T->is(2).first() * h } );
            coords.push_back( { T->is(0).first() * h, T->is(1).first() * h, T->is(2).last()  * h } );
            coords.push_back( { T->is(0).last()  * h, T->is(1).first() * h, T->is(2).last()  * h } );
            coords.push_back( { T->is(0).first() * h, T->is(1).last()  * h, T->is(2).last()  * h } );
            coords.push_back( { T->is(0).last()  * h, T->is(1).last()  * h, T->is(2).last()  * h } );

            voxels.push_back({ nvoxel, nvoxel+1, nvoxel+2, nvoxel+3, nvoxel+4, nvoxel+5, nvoxel+6, nvoxel+7 });
            nvoxel += 8;

            if      ( tensor::is_tucker( *T ) ) colors.push_back( 1 );
            else if ( tensor::is_dense(  *T ) ) colors.push_back( 2 );
            else                                colors.push_back( 0 );
        }// else
    }// while

    //
    // write VTK file
    //

    auto  outname = std::filesystem::path( filename );
    auto  out     = std::ofstream( outname.has_extension() ? filename : filename + ".vtk" );
    
    out << "# vtk DataFile Version 2.0" << std::endl
        << "HLR hierarchical tensor" << std::endl
        << "ASCII" << std::endl
        << "DATASET UNSTRUCTURED_GRID" << std::endl;

    out << "POINTS " << coords.size() << " FLOAT" << std::endl;

    for ( auto  c : coords )
        out << c[0] << ' ' << c[1] << ' ' << c[2] << std::endl;

    out << "CELLS " << voxels.size() << ' ' << 9 * voxels.size() << std::endl;

    for ( auto  v : voxels )
        out << "8 " << v[0] << ' ' << v[1] << ' ' << v[2] << ' ' << v[3] << ' ' << v[4] << ' ' << v[5] << ' ' << v[6] << ' ' << v[7] << std::endl;
        
    out << "CELL_TYPES " << voxels.size() << std::endl;
        
    for ( size_t  i = 0; i < voxels.size(); ++i )
        out << "11 ";
    out << std::endl;

    //
    // cell colour
    //

    out << "CELL_DATA " << voxels.size() << std::endl
        << "COLOR_SCALARS cellcolour 1" << std::endl;

    for ( auto  c : colors )
        out << c << ' ';
    out << std::endl;
}

}// namespace detail

}}// namespace hlr::io

#endif // __HLR_UTILS_DETAIL_IO_HH

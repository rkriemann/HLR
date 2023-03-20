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

namespace hlr { namespace io {

namespace detail
{

//////////////////////////////////////////////////////////////////////
//
// HDF5 format
//
//////////////////////////////////////////////////////////////////////

#if defined(USE_HDF5)

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
            data_dims[i] = t.dim(i);
    
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
                  const std::string &               gname,
                  const blas::tensor3< value_t > &  t )
{
    constexpr uint  dim = 3;

    auto  group = std::make_unique< H5::Group >( file.createGroup( gname ) );

    if ( Hpro::is_complex_type< value_t >::value )
        HLR_ERROR( "complex not yet supported" );

    //
    // just tensor data
    //

    {
        hsize_t  data_dims[ dim ];

        for ( uint  i = 0; i < dim; ++i )
            data_dims[i] = t.size(i);
    
        auto  data_space = H5::DataSpace( dim, data_dims );

        // define datatype
        auto  data_type = ( Hpro::is_single_prec_v< value_t > ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE );

        // create dataset for tensor data
        auto  data_set = file.createDataSet( gname + "/values", data_type, data_space );
            
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
            HLR_ASSERT( dims[d] == is[d].size() );
        }// for
        
        data_set.read( t.tensor().data(), data_type );
    }
    
    return t;
}

#endif

//////////////////////////////////////////////////////////////////////
//
// VTK format
//
//////////////////////////////////////////////////////////////////////

template < typename value_t >
void
vtk_print_tensor ( const blas::tensor3< value_t > &  t,
                   const std::string &               filename )
{
    auto  outname = std::filesystem::path( filename );
    auto  out     = std::ofstream( outname.has_extension() ? filename : filename + ".vtk" );
    
    out << "# vtk DataFile Version 2.0" << std::endl
        << "HLIBpro coordinates" << std::endl
        << "ASCII" << std::endl
        << "DATASET UNSTRUCTURED_GRID" << std::endl;

    //
    // assuming tensor grid in equal step width in all dimensions
    //

    const size_t  n = t.size(0) * t.size(1) * t.size(2);
    const double  h = 1.0 / std::min( t.size(0), std::min( t.size(1), t.size(2) ) );

    out << "POINTS " << n << " FLOAT" << std::endl;

    for ( size_t  l = 0; l < t.size(2); ++l )
        for ( size_t  j = 0; j < t.size(1); ++j )
            for ( size_t  i = 0; i < t.size(0); ++i )
                out << i * h << ' ' << j * h << ' ' << l * h << std::endl;

    out << "CELLS " << n << ' ' << 2 * n << std::endl;

    for ( size_t  i = 0; i < n; ++i )
        out << "1 " << i << ' ';
    out << std::endl;
        
    out << "CELL_TYPES " << n << std::endl;
        
    for ( size_t  i = 0; i < n; ++i )
        out << "1 ";
    out << std::endl;

    out << "CELL_DATA " << n << std::endl
        << "COLOR_SCALARS v" << " 1" << std::endl;
        
    for ( size_t  i = 0; i < n; ++i )
        out << t.data()[i] << ' ';
    out << std::endl;
}

}// namespace detail

}}// namespace hlr::io

#endif // __HLR_UTILS_DETAIL_IO_HH

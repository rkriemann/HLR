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

#include <hlr/tensor/structured_tensor.hh>

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
        data_dims[i] = t.size(i);
    
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
            HLR_ASSERT( dims[d] == is[d].size() );
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
    
    auto  t = blas::tensor3< value_t >( dims[0], dims[1], dims[2] );
    
    data_set.read( t.data(), data_type );
    
    return t;
}

#endif // HLR_USE_HDF5

//////////////////////////////////////////////////////////////////////
//
// VTK format
//
//////////////////////////////////////////////////////////////////////

template < typename value_t >
void
vtk_print_full_tensor ( const blas::tensor3< value_t > &  t,
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

    const size_t  nc = t.size(0) * t.size(1) * t.size(2);
    const size_t  nv = (t.size(0)+1) * (t.size(1)+1) * (t.size(2)+1);
    const double  h  = 1.0 / ( std::min({ t.size(0), t.size(1), t.size(2) }) + 1 );

    out << "POINTS " << nv << " FLOAT" << std::endl;

    for ( size_t  l = 0; l <= t.size(2); ++l )
        for ( size_t  j = 0; j <= t.size(1); ++j )
            for ( size_t  i = 0; i <= t.size(0); ++i )
                out << i * h << ' ' << j * h << ' ' << l * h << std::endl;

    //
    //     6-------7
    //    /|      /|
    //   4-------5 |
    //   | 2-----|-3
    //   |/      |/
    //   0 ----- 1
    //
            
    out << "CELLS " << nc << ' ' << 9 * nc << std::endl;

    for ( size_t  l = 0; l < t.size(2); ++l )
        for ( size_t  j = 0; j < t.size(1); ++j )
            for ( size_t  i = 0; i < t.size(0); ++i )
                out << "8 "
                    << ( l * t.size(1) + j ) * t.size(0) + i
                    << ' '
                    << ( l * t.size(1) + j ) * t.size(0) + (i + 1)
                    << ' '
                    << ( l * t.size(1) + (j+1) ) * t.size(0) + i
                    << ' '
                    << ( l * t.size(1) + (j+1) ) * t.size(0) + (i+1)
                    << ' '
                    << ( (l+1) * t.size(1) + j ) * t.size(0) + i
                    << ' '
                    << ( (l+1) * t.size(1) + j ) * t.size(0) + (i+1)
                    << ' '
                    << ( (l+1) * t.size(1) + (j+1) ) * t.size(0) + i
                    << ' '
                    << ( (l+1) * t.size(1) + (j+1) ) * t.size(0) + (i+1)
                    << std::endl;
    out << std::endl;
        
    out << "CELL_TYPES " << nc << std::endl;
        
    for ( size_t  i = 0; i < nc; ++i )
        out << "11 ";
    out << std::endl;

    out << "CELL_DATA " << nc << std::endl
        << "COLOR_SCALARS v" << " 1" << std::endl;
        
    for ( size_t  i = 0; i < n; ++i )
    {
        //
        // average of vertex values
        //
        
        auto  v = ( t(   i,   j,   l ) +
                    t( i+1,   j,   l ) +
                    t(   i, j+1,   l ) +
                    t( i+1, j+1,   l ) +
                    t(   i,   j, l+1 ) +
                    t( i+1,   j, l+1 ) +
                    t(   i, j+1, l+1 ) +
                    t( i+1, j+1, l+1 ) );
            
        out << v / 8.0 << ' ';
    }// for
    
    out << std::endl;
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
        }// else
    }// while

    //
    // write VTK file
    //

    auto  outname = std::filesystem::path( filename );
    auto  out     = std::ofstream( outname.has_extension() ? filename : filename + ".vtk" );
    
    out << "# vtk DataFile Version 2.0" << std::endl
        << "HLIBpro coordinates" << std::endl
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
}

}// namespace detail

}}// namespace hlr::io

#endif // __HLR_UTILS_DETAIL_IO_HH

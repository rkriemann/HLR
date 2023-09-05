//
// Project     : HLR
// Module      : matrix/print
// Description : printing functions for matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <list>
#include <fstream>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <hpro/config.h>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>

#if defined(USE_LIC_CHECK)  // hack to test for full HLIBpro
#include <hpro/matrix/TUniformMatrix.hh>
#endif

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/lrsvmatrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/h2_lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/matrix/sparse_matrix.hh>
#include <hlr/utils/eps_printer.hh>
#include <hlr/utils/tools.hh>

#include <hlr/matrix/print.hh>

namespace hlr { namespace matrix {

namespace
{

//
// color indices
//
enum {
    HLR_COLOR_BG_DEFAULT,
    HLR_COLOR_BG_BLOCKED,
    HLR_COLOR_BG_DENSE,
    HLR_COLOR_BG_DENSE_COMPRESSED,
    HLR_COLOR_BG_LOWRANK,
    HLR_COLOR_BG_LOWRANK_COMPRESSED,
    HLR_COLOR_BG_UNIFORM,
    HLR_COLOR_BG_UNIFORM_COMPRESSED,
    HLR_COLOR_BG_H2,
    HLR_COLOR_BG_H2_COMPRESSED,
    HLR_COLOR_BG_SPARSE,
    HLR_COLOR_BG_SPARSE_COMPRESSED,
    HLR_COLOR_FG_DEFAULT,
    HLR_COLOR_FG_BORDER,
    HLR_COLOR_FG_DIM,
    HLR_COLOR_FG_RANK,
    HLR_COLOR_FG_INDEXSET,
    HLR_COLOR_FG_ID,
    HLR_COLOR_FG_PATTERN,
};

//
// color palette for matrix printing
//
const uint colors[] = {
    // background
    0xFFFFFF,   // default
    0xA0A0A0,   // blocked
    0xF35E5E,   // dense (ScarletRed1!75!White)
    0xF79494,   // compressed dense (ScarletRed1!50!White)
    0xC4F099,   // lowrank (Chameleon1!50!White)
    0xE1F7CC,   // compressed lowrank (Chameleon1!25!White)
    0xB8CFE7,   // uniform (SkyBlue1!50!White)
    0xDBE7F3,   // compressed uniform (SkyBlue1)
    0xB8CFE7,   // H2 (SkyBlue1!50!White)
    0xDBE7F3,   // compressed H2 (SkyBlue1!25!White)
    0xFDF4A7,   // sparse (Butter!50!White)
    0xFEF9D3,   // compressed sparse (Butter!25!White)
    
    // foreground
    0x000000,   // default
    0x000000,   // border
    0x000000,   // dimensions
    0x000000,   // rank
    0x5C3566,   // indexset (Plum3)
    0x204A87,   // id (SkyBlue3)
    0x000000    // pattern
};

//
// print singular values
//
template < typename value_t >
void
print_sv ( eps_printer &                                    prn,
           const Hpro::TMatrix< value_t > &                 M,
           const blas::vector< real_type_t< value_t > > &   S )
{
    const auto  k = S.length();

    if ( k == 0 )
        return;
    
    const auto  max_k  = std::log10( S(0) );
    const auto  min_k  = std::log10( S(k-1) / 5.0 );
    const auto  rbrd   = std::min( 75.0, M.nrows() * 0.1 );
    const auto  cbrd   = std::min( 75.0, M.ncols() * 0.1 );
    
    prn.save();
    
    prn.translate( M.col_ofs()             + cbrd,
                   M.row_ofs() + M.nrows() - rbrd );

    prn.scale(  ( M.ncols() - 2 * cbrd ) / double(k),
               -( M.nrows() - 2 * rbrd ) / ( max_k - min_k + 1.0 ) );
        
    for ( uint  i = 0; i < k; i++ )
    {
        const auto  y = std::log10( S(i) ) - min_k + 1;

        if ( y > 0.0 )
            prn.fill_rect( 0, 0, i+1, y );
    }// for

    prn.restore();
}

//
// actual print function
//
template < typename value_t >
void
print_eps ( const Hpro::TMatrix< value_t > &    M,
            eps_printer &                       prn,
            const bool                          recurse,
            const std::vector< std::string > &  options )
{
    if ( is_blocked( M ) && recurse )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    print_eps( * B->block( i, j ), prn, recurse, options );
            }// for
        }// for
    }// if
    else
    {
        if ( matrix::is_dense( M ) || Hpro::is_dense( M ) )
        {
            if ( matrix::is_dense( M ) && cptrcast( &M, matrix::dense_matrix< value_t > )->is_compressed() )
                prn.set_rgb( colors[HLR_COLOR_BG_DENSE_COMPRESSED] );
            else
                prn.set_rgb( colors[HLR_COLOR_BG_DENSE] );
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );

            if ( ! contains( options, "nosize" ) )
            {
                prn.save();
                prn.set_font( "Helvetica", std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 4.0 ) );
                
                prn.set_rgb( colors[HLR_COLOR_FG_DIM] );
                if ( M.nrows() != M.ncols() )
                    prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
                                   double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
                                   Hpro::to_string( "%dx%d", M.nrows(), M.ncols() ) );
                else
                    prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
                                   double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
                                   Hpro::to_string( "%d", M.nrows() ) );
                
                prn.restore();
            }// if

            if ( contains( options, "pattern" ) )
            {
                auto  D = blas::matrix< value_t >();

                if ( matrix::is_dense( M ) ) D = cptrcast( &M, matrix::dense_matrix< value_t > )->mat();
                else                         D = cptrcast( &M, Hpro::TDenseMatrix< value_t > )->blas_mat();

                prn.set_gray( 0 );
                
                for ( uint  i = 0; i < D.nrows(); ++i )
                    for ( uint  j = 0; j < D.ncols(); ++j )
                        if ( D(i,j) != value_t(0) )
                            prn.fill_rect( j     + M.col_ofs(), i     + M.row_ofs(),
                                           j + 1 + M.col_ofs(), i + 1 + M.row_ofs() );
            }// if
        }// if
        // else if ( is_mixedprec_lowrank( M ) && is_compressed( M ) )
        // {
        //     auto        R          = cptrcast( &M, matrix::lrsvmatrix< value_t > );
        //     const auto  rank       = R->rank();
        //     auto        mpdata     = R->mp_data();
        //     uint        col_bg[3]  = { 0, 0, 0 };

        //     const uint  col_mp3[3] = { 252, 233,  79 }; // yellow
        //     const uint  col_mp2[3] = { 114, 159, 207 }; // blue
        //     const uint  col_mp1[3] = { 239,  41,  41 }; // red

        //     const uint  nmp1       = mpdata.U1.size() / R->nrows();
        //     const uint  nmp2       = mpdata.U2.size() / R->nrows();
        //     const uint  nmp3       = mpdata.U3.size() / R->nrows();

        //     for ( uint  c = 0; c < 3; c++ )
        //         col_bg[c] = std::min< uint >( 255, uint( ( nmp3 * col_mp3[c] +
        //                                                    nmp2 * col_mp2[c] +
        //                                                    nmp1 * col_mp1[c] ) / double(rank) ) );
            
        //     prn.set_rgb( (col_bg[0] << 16) + (col_bg[1] << 8) + col_bg[2] );
        //     prn.fill_rect( M.col_ofs(),
        //                    M.row_ofs(),
        //                    M.col_ofs() + M.ncols(),
        //                    M.row_ofs() + M.nrows() );

        //     if ( ! contains( options, "norank" ) )
        //     {
        //         prn.save();
        //         prn.set_font( "Helvetica", std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 4.0 ) );
                
        //         prn.set_rgb( colors[HLR_COLOR_FG_RANK] );
        //         prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
        //                        double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
        //                        Hpro::to_string( "%d", rank ) );
                
        //         prn.restore();
        //     }// if
        // }// if
        else if ( matrix::is_lowrank( M ) )
        {
            auto  R    = cptrcast( &M, matrix::lrmatrix< value_t > );
            auto  rank = R->rank();

            if ( ! contains( options, "nonempty" ) || ( rank > 0 ))
            {
                if ( R->is_compressed() ) prn.set_rgb( colors[HLR_COLOR_BG_LOWRANK_COMPRESSED] );
                else                      prn.set_rgb( colors[HLR_COLOR_BG_LOWRANK] );

                prn.fill_rect( M.col_ofs(),
                               M.row_ofs(),
                               M.col_ofs() + M.ncols(),
                               M.row_ofs() + M.nrows() );

                if ( contains( options, "sv" ) )
                {
                    auto  U = R->U();
                    auto  V = R->V();

                    if ( U.ncols() > 0 )
                    {
                        auto  S = blas::sv( U, V );
                    
                        prn.set_gray( 128 );
                        print_sv( prn, M, S );
                    }// if
                }// if

                if ( ! contains( options, "norank" ) )
                {
                    prn.save();
                    prn.set_font( "Helvetica", std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 4.0 ) );
                    
                    prn.set_rgb( colors[HLR_COLOR_FG_RANK] );
                    prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
                                   double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
                                   Hpro::to_string( "%d", rank ) );
                    
                    prn.restore();
                }// if

                if ( contains( options, "pattern" ) )
                {
                    auto  D = blas::prod( R->U(), blas::adjoint( R->V() ) );
                    
                    prn.set_gray( 0 );
                
                    for ( uint  i = 0; i < D.nrows(); ++i )
                        for ( uint  j = 0; j < D.ncols(); ++j )
                            if ( D(i,j) != value_t(0) )
                                prn.fill_rect( j     + M.col_ofs(), i     + M.row_ofs(),
                                               j + 1 + M.col_ofs(), i + 1 + M.row_ofs() );
                }// if
            }// if
        }// if
        else if ( matrix::is_uniform_lowrank( M ) )
        {
            auto  R = cptrcast( &M, matrix::uniform_lrmatrix< value_t > );

            // background
            prn.set_rgb( colors[HLR_COLOR_BG_UNIFORM] );
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );

            if ( ! contains( options, "norank" ) )
            {
                prn.save();
                prn.set_font( "Helvetica", std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 4.0 ) );
                
                prn.set_rgb( colors[HLR_COLOR_FG_RANK] );
                prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
                               double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
                               Hpro::to_string( "%dx%d", R->row_rank(), R->col_rank() ) );
                
                prn.restore();
            }// if
        }// if
        else if ( matrix::is_h2_lowrank( &M ) )
        {
            auto  R = cptrcast( &M, matrix::h2_lrmatrix< value_t > );

            prn.set_rgb( colors[HLR_COLOR_BG_H2] );
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );

            if ( ! contains( options, "norank" ) )
            {
                prn.save();
                prn.set_font( "Helvetica", std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 4.0 ) );
                
                prn.set_rgb( colors[HLR_COLOR_FG_RANK] );
                prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
                               double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
                               Hpro::to_string( "%dx%d", R->row_rank(), R->col_rank() ) );
                
                prn.restore();
            }// if
        }// if
        #if defined(USE_LIC_CHECK)
        else if ( Hpro::is_uniform( &M ) )
        {
            auto  R = cptrcast( &M, Hpro::TUniformMatrix< value_t > );

            prn.set_rgb( colors[HLR_COLOR_BG_H2] );
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );

            if ( ! contains( options, "norank" ) )
            {
                prn.save();
                prn.set_font( "Helvetica", std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 4.0 ) );
                
                prn.set_rgb( colors[HLR_COLOR_FG_RANK] );
                prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
                               double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
                               Hpro::to_string( "%dx%d", R->row_rank(), R->col_rank() ) );
                
                prn.restore();
            }// if
        }// if
        #endif
        else if ( is_sparse( M ) )
        {
            auto  S = cptrcast( &M, Hpro::TSparseMatrix< value_t > );
            
            // background
            prn.set_rgb( colors[HLR_COLOR_BG_SPARSE] );
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );

            if ( contains( options, "pattern" ) )
            {
                prn.set_rgb( colors[HLR_COLOR_FG_PATTERN] );

                for ( uint  i = 0; i < S->nrows(); ++i )
                {
                    const auto  lb = S->rowptr( i );
                    const auto  ub = S->rowptr( i+1 );
                    
                    for ( uint  l = lb; l < ub; ++l )
                        prn.fill_rect( S->colind(l)     + S->col_ofs(), i     + S->row_ofs(),
                                       S->colind(l) + 1 + S->col_ofs(), i + 1 + S->row_ofs() );
                }// for
            }// if
        }// if
        else if ( is_sparse_eigen( M ) )
        {
            auto  S = cptrcast( &M, sparse_matrix< value_t > );
            
            // background
            prn.set_rgb( colors[HLR_COLOR_BG_SPARSE] );
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );

            if ( contains( options, "pattern" ) )
            {
                #if defined(HAS_EIGEN)
                using  iter_t = typename sparse_matrix< value_t >::spmat_t::InnerIterator;
                
                prn.set_rgb( colors[HLR_COLOR_FG_PATTERN] );

                for ( int k = 0; k < S->spmat().outerSize(); ++k )
                {
                    for ( iter_t  it( S->spmat(), k ); it; ++it )
                    {
                        const auto  val = it.value();
                        const auto  row = it.row() + S->row_ofs();
                        const auto  col = it.col() + S->col_ofs();
                        
                        prn.fill_rect( col,   row,
                                       col+1, row+1 );
                    }// for
                }// for
                #endif
            }// if
        }// else

        //
        // draw index set data 
        //

        if ( contains( options, "indexset" ) )
        {
            const auto  fn_size = std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 16.0 );
            
            prn.save();
            prn.set_font( "Helvetica", fn_size );
                
            prn.set_rgb( colors[HLR_COLOR_FG_INDEXSET] );
            prn.draw_text( double(M.col_ofs()) + double(M.ncols()) / 2.0,
                           double(M.row_ofs()) + fn_size,
                           Hpro::to_string( "%d ... %d", M.col_ofs(), M.col_ofs() + M.ncols() - 1 ), 'c' );

            prn.save();
            prn.translate( double(M.col_ofs()) + fn_size, double(M.row_ofs()) + double(M.nrows()) / 2.0 );
            prn.rotate( -90 );
            prn.draw_text( 0, 0, Hpro::to_string( "%d ... %d", M.row_ofs(), M.row_ofs() + M.nrows() - 1 ), 'c' );
            prn.restore();
            
            prn.restore();
        }// if

        //
        // draw frame
        //
        prn.set_rgb( colors[HLR_COLOR_FG_BORDER] );
        prn.draw_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );
                       
    }// else

    // draw ID
    if ( ! ( contains( options, "noid" ) || ( contains( options, "noinnerid" ) && is_blocked( M ) )) )
    {
        prn.save();

        const auto  fn_size = std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 4.0 );
        const auto  text    = Hpro::to_string( "%d", M.id() );
    
        prn.set_font( "Helvetica", fn_size );
        prn.set_rgb( colors[HLR_COLOR_FG_ID] );
        prn.draw_text( double(M.col_ofs()) + double(M.cols()) / 2.0 - fn_size * text.length() / 4.0,
                       double(M.row_ofs()) + double(M.rows()) / 2.0 + fn_size / 3.0,
                       text );
        prn.restore();
    }// if
}

//
// actual print function
//
template < typename value_t >
void
print_mem ( const Hpro::TMatrix< value_t > &    M,
            eps_printer &                       prn,
            const std::vector< std::string > &  options )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    print_mem( * B->block( i, j ), prn, options );
            }// for
        }// for
    }// if
    else
    {
        if ( matrix::is_dense( M ) )
        {
            // background
            prn.set_gray( 0 );
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );
        }// if
        else if ( matrix::is_lowrank( M ) )
        {
            auto  R     = cptrcast( &M, matrix::lrmatrix< value_t > );
            auto  rank  = R->rank();
            auto  ratio = ( rank * double( R->nrows() + R->ncols() ) ) / ( double(R->nrows()) * double(R->ncols()) );
            
            // background
            prn.set_gray( 255 - int( ratio * 255.0 ) );
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );
        }// if
        else if ( matrix::is_uniform_lowrank( M ) )
        {
            auto  R        = cptrcast( &M, matrix::uniform_lrmatrix< value_t > );
            auto  row_rank = R->row_rank();
            auto  col_rank = R->row_rank();
            auto  ratio    = ( ( row_rank * double( R->nrows() ) +
                                 col_rank * double( R->ncols() ) +
                                 double(row_rank * col_rank) ) /
                               ( double(R->nrows()) * double(R->ncols()) ) );
            
            // background
            prn.set_gray( 255 - int( ratio * 255.0 ) );
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type: " + M.typestr() );

        // draw frame
        prn.set_gray( 0 );
        prn.draw_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );
    }// else
}

}// namespace anonymous

//
// print matrix <M> to file <filename>
//
template < typename value_t >
void
print_eps ( const Hpro::TMatrix< value_t > &  M,
            const std::string &               filename,
            const std::string &               options )
{
    const boost::filesystem::path  filepath( filename );
    std::string                    suffix;

    if ( ! filepath.has_extension() )
        suffix = ".eps";

    std::vector< std::string >  optarr;

    boost::split( optarr, options, [] ( char c ) { return c == ','; } );
    
    std::ofstream  out( filename + suffix );
    eps_printer    prn( out );

    const auto   max_size = std::max( std::max( M.nrows(), M.ncols() ), size_t(1) );
    const auto   min_size = std::max( std::min( M.nrows(), M.ncols() ), size_t(1) );
    const auto   width    = ( M.ncols() == max_size ? 500 : 500 * double(min_size) / double(max_size) );
    const auto   height   = ( M.nrows() == max_size ? 500 : 500 * double(min_size) / double(max_size) );
    
    prn.begin( width, height );

    prn.scale( double(width)  / double(M.ncols()),
               double(height) / double(M.nrows()) );
    
    prn.translate( - double(M.col_ofs()),
                   - double(M.row_ofs()) );
    
    prn.set_font( "Courier", 0.3 );

    prn.set_line_width( 0.1 );

    print_eps( M, prn, true, optarr );

    prn.end();
}

//
// print matrix <M> to file <filename>
//
template < typename value_t >
void
print_lvl_eps ( const Hpro::TMatrix< value_t > &  M,
                const std::string &               basename,
                const std::string &               options )
{
    std::vector< std::string >  optarr;

    boost::split( optarr, options, [] ( char c ) { return c == ','; } );

    //
    // common settings for all files
    //
    
    const auto   max_size = std::max( std::max( M.nrows(), M.ncols() ), size_t(1) );
    const auto   min_size = std::max( std::min( M.nrows(), M.ncols() ), size_t(1) );
    const auto   width    = ( M.ncols() == max_size ? 500 : 500 * double(min_size) / double(max_size) );
    const auto   height   = ( M.nrows() == max_size ? 500 : 500 * double(min_size) / double(max_size) );
    
    //
    // go BFS style through matrix and print each level separately
    //

    auto  parents = std::list< const Hpro::TMatrix< value_t > * >();
    auto  blocks  = std::list< const Hpro::TMatrix< value_t > * >{ & M };
    uint  lvl     = 0;

    while ( ! blocks.empty() )
    {
        if ( lvl > 0 )
        {
            std::ostringstream  filename;

            filename << basename << lvl << ".eps";
            
            std::ofstream  out( filename.str() );
            eps_printer    prn( out );
            
            prn.begin( width, height );

            prn.scale( double(width)  / double(M.ncols()),
                       double(height) / double(M.nrows()) );
    
            prn.translate( - double(M.col_ofs()),
                           - double(M.row_ofs()) );
    
            prn.set_font( "Courier", 0.3 );

            prn.set_line_width( 0.1 );

            for ( auto  M_i : blocks )
            {
                print_eps( * M_i, prn, false, optarr );
            }// for

            // draw thicker frame around parent blocks to show block structure
            if ( ! parents.empty() )
            {
                prn.save();
                prn.set_gray( 0 );
                prn.set_line_width( std::max( std::min( M.rows(), M.cols() ) / 500.0, 1.0 ) );
                    
                for ( auto  P_i : parents )
                {
                    prn.draw_rect( P_i->col_ofs(),
                                   P_i->row_ofs(),
                                   P_i->col_ofs() + P_i->ncols(),
                                   P_i->row_ofs() + P_i->nrows() );
                }// for

                prn.restore();
            }// if
            
            prn.end();
        }// if

        //
        // next level
        //

        auto  sons = std::list< const Hpro::TMatrix< value_t > * >();

        for ( auto  M_i : blocks )
        {
            if ( is_blocked( M_i ) )
            {
                auto  B_i = cptrcast( M_i, Hpro::TBlockMatrix< value_t > );

                for ( uint  i = 0; i < B_i->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < B_i->nblock_cols(); ++j )
                    {
                        if ( ! is_null( B_i->block( i, j ) ) )
                            sons.push_back( B_i->block( i, j ) );
                    }// for
                }// for
            }// if
        }// for

        parents = std::move( blocks );
        blocks  = std::move( sons );
        lvl++;
    }// while
}

//
// colorize matrix blocks in <M> according to rank
//
template < typename value_t >
void
print_mem_eps ( const Hpro::TMatrix< value_t > &  M,
                const std::string &               filename,
                const std::string &               options )
{
    const boost::filesystem::path  filepath( filename );
    std::string                    suffix;

    if ( ! filepath.has_extension() )
        suffix = ".eps";

    std::vector< std::string >  optarr;

    boost::split( optarr, options, [] ( char c ) { return c == ','; } );
    
    std::ofstream  out( filename + suffix );
    eps_printer    prn( out );

    const auto   max_size = std::max( std::max( M.nrows(), M.ncols() ), size_t(1) );
    const auto   min_size = std::max( std::min( M.nrows(), M.ncols() ), size_t(1) );
    const auto   width    = ( M.ncols() == max_size ? 500 : 500 * double(min_size) / double(max_size) );
    const auto   height   = ( M.nrows() == max_size ? 500 : 500 * double(min_size) / double(max_size) );
    
    prn.begin( width, height );

    prn.scale( double(width)  / double(M.ncols()),
               double(height) / double(M.nrows()) );
    
    prn.translate( - double(M.col_ofs()),
                   - double(M.row_ofs()) );
    
    prn.set_font( "Courier", 0.3 );

    prn.set_line_width( 0.1 );

    print_mem( M, prn, optarr );

    prn.end();
}

//
// print cluster basis <cl> to file <filename>
//
template < typename value_t >
void
print_eps ( const shared_cluster_basis< value_t > &  cb,
            const std::string &                      filename,
            const std::string &                      options )
{
    using real_t = Hpro::real_type_t< value_t >;
    
    const boost::filesystem::path  filepath( filename );
    std::string                    suffix;

    if ( ! filepath.has_extension() )
        suffix = ".eps";

    std::vector< std::string >  optarr;

    boost::split( optarr, options, [] ( char c ) { return c == ','; } );
    
    std::ofstream  out( filename + suffix );
    eps_printer    prn( out );

    const uint    depth     = uint( cb.depth() );
    const uint    size      = uint( cb.is().size() );
    const double  max_x     = 500.0;
    const double  scale_x   = max_x / double(size);
    const double  scale_y   = std::min( 500.0 / double(depth+1), 20.0 );
    const double  prn_width = max_x;
    auto          bases     = std::list< const shared_cluster_basis< value_t > * >{ & cb };
    uint          level     = 0;

    prn.begin( uint(prn_width), uint(( depth ) * scale_y) );
    prn.set_line_width( 1.0 / std::max( scale_x, scale_y ) );

    while ( ! bases.empty() )
    {
        decltype( bases )  sons;

        for ( auto  clb : bases )
        {
            const auto    is      = clb->is();
            const double  fn_size = std::min( 5.0, scale_x * double(is.size()) / 2.0 );

            if ( clb->rank() != 0 )
            {
                uint  text_col = 0;
                
                if ( contains( optarr, "mem" ) )
                {
                    uint  bg_col = std::max< int >( 0, 255 - std::floor( 255.0 * double( clb->basis().ncols() ) / double( clb->basis().nrows() ) ) );

                    if ( bg_col < 96 )
                        text_col = 255;
                    
                    prn.save();

                    prn.set_gray( bg_col );
                    prn.fill_rect( scale_x * is.first(),        scale_y * level,
                                   scale_x * ( is.last() + 1 ), scale_y * ( level + 1 ) );
                    
                    prn.restore();
                }// if
                
                if ( contains( optarr, "svd" ) )
                {
                    auto  V = blas::copy( clb->basis() );
                    auto  S = blas::vector< real_t >();

                    blas::sv( V, S );

                    const auto  max_k = math::log10( S(0) );
                    const auto  min_k = math::log10( S(S.length()-1) );
                    const auto  h_gap = scale_x * is.size() * 0.05;
                    const auto  v_gap = scale_y * 0.05;

                    prn.save();

                    prn.translate( scale_x * is.first() + h_gap, scale_y * (level+1) - v_gap );
                    prn.scale( ( scale_x * is.size() - 2.0 * h_gap ) / double( S.length() ),
                               -( scale_y - 2.0 * v_gap ) / ( max_k - min_k + 1.0 ) );

                    prn.set_rgb( 32,74,135 ); // SkyBlue3
                    
                    auto  y_last = real_t(-1);
                    
                    for ( idx_t  i = 0; i < idx_t( S.length() ); i++ )
                    {
                        const auto  y = math::log10( S(i) ) - min_k + 1;

                        if ( y > 0.0 )
                        {
                            if ( y_last >= real_t(0) )
                                prn.draw_line( double(i)-0.5, y_last, double(i)+0.5, y );
                        }// if

                        y_last = y;
                    }// for

                    prn.restore();
                }// if
                
                if ( ! contains( optarr, "norank" ) )
                {
                    std::ostringstream  srank;

                    srank << clb->rank();
                
                    prn.save();

                    if ( text_col != 0 )
                        prn.set_gray( text_col );
                    
                    prn.set_font( "Helvetica", fn_size );
                    prn.draw_text( scale_x * (is.first() + is.last() + 1) / 2.0,
                                   scale_y * (level + 0.5) + fn_size * 0.4,
                                   srank.str(),
                                   'c' );
                
                    prn.restore();
                }// if
            }// if
            
            prn.draw_rect( scale_x * is.first(),    scale_y * level,
                           scale_x * (is.last()+1), scale_y * ( level+1 ) );

            for ( uint  i = 0; i < clb->nsons(); ++i )
            {
                if ( ! is_null( clb->son( i ) ) )
                    sons.push_back( clb->son( i ) );
            }// for
        }// while

        bases = std::move( sons );

        ++level;
    }// while
    
    prn.end();
}

template < typename value_t >
void
print_eps ( const nested_cluster_basis< value_t > &  cb,
            const std::string &                      filename,
            const std::string &                      options )
{
    using real_t = Hpro::real_type_t< value_t >;
    
    const boost::filesystem::path  filepath( filename );
    std::string                    suffix;

    if ( ! filepath.has_extension() )
        suffix = ".eps";

    std::vector< std::string >  optarr;

    boost::split( optarr, options, [] ( char c ) { return c == ','; } );
    
    std::ofstream  out( filename + suffix );
    eps_printer    prn( out );

    const uint    depth     = uint( cb.depth() );
    const uint    size      = uint( cb.is().size() );
    const double  max_x     = 500.0;
    const double  scale_x   = max_x / double(size);
    const double  scale_y   = std::min( 500.0 / double(depth+1), 20.0 );
    const double  prn_width = max_x;
    auto          bases     = std::list< const nested_cluster_basis< value_t > * >{ & cb };
    uint          level     = 0;

    prn.begin( uint(prn_width), uint(( depth ) * scale_y) );
    prn.set_line_width( 1.0 / std::max( scale_x, scale_y ) );

    while ( ! bases.empty() )
    {
        decltype( bases )  sons;

        for ( auto  clb : bases )
        {
            const auto    is      = clb->is();
            const double  fn_size = std::min( 5.0, scale_x * double(is.size()) / 2.0 );

            if ( clb->rank() != 0 )
            {
                uint  text_col = 0;
                
                if ( ! contains( optarr, "norank" ) )
                {
                    std::ostringstream  srank;

                    srank << clb->rank();
                
                    prn.save();

                    if ( text_col != 0 )
                        prn.set_gray( text_col );
                    
                    prn.set_font( "Helvetica", fn_size );
                    prn.draw_text( scale_x * (is.first() + is.last() + 1) / 2.0,
                                   scale_y * (level + 0.5) + fn_size * 0.4,
                                   srank.str(),
                                   'c' );
                
                    prn.restore();
                }// if
            }// if
            
            prn.draw_rect( scale_x * is.first(),    scale_y * level,
                           scale_x * (is.last()+1), scale_y * ( level+1 ) );

            for ( uint  i = 0; i < clb->nsons(); ++i )
            {
                if ( ! is_null( clb->son( i ) ) )
                    sons.push_back( clb->son( i ) );
            }// for
        }// while

        bases = std::move( sons );

        ++level;
    }// while
    
    prn.end();
}

#define INST_PRINT( type )                                              \
    template void print_eps< type >     ( const Hpro::TMatrix< type > &, \
                                          const std::string &          , \
                                          const std::string &          ); \
    template void print_lvl_eps< type > ( const Hpro::TMatrix< type > &, \
                                          const std::string &          , \
                                          const std::string &          ); \
    template void print_mem_eps< type > ( const Hpro::TMatrix< type > &, \
                                          const std::string &          , \
                                          const std::string &          ); \
    template void print_eps< type >     ( const shared_cluster_basis< type > & , \
                                          const std::string &,          \
                                          const std::string & );        \
    template void print_eps< type >     ( const nested_cluster_basis< type > & , \
                                          const std::string &,          \
                                          const std::string & );        \

INST_PRINT( float )
INST_PRINT( double )
INST_PRINT( std::complex< float > )
INST_PRINT( std::complex< double > )

}}// namespace hlr::matrix

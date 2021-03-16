//
// Project     : HLR
// Module      : matrix/print
// Description : printing functions for matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <list>
#include <fstream>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>

#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/utils/eps_printer.hh>

#include <hlr/matrix/print.hh>

namespace hlr { namespace matrix {

namespace
{

//
// actual print function
//
void
print_eps ( const hpro::TMatrix &               M,
            eps_printer &                       prn,
            const bool                          recurse,
            const std::vector< std::string > &  options )
{
    if ( is_blocked( M ) && recurse )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix );

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
        if ( is_dense( M ) )
        {
            // background
            prn.set_rgb( 243, 94, 94 ); // ScarletRed1!75!White
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );

            if ( std::find( options.cbegin(), options.cend(), "nosize" ) == options.end() )
            {
                prn.save();
                prn.set_font( "Helvetica", std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 4.0 ) );
                
                prn.set_gray( 0 );
                if ( M.nrows() != M.ncols() )
                    prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
                                   double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
                                   hpro::to_string( "%dx%d", M.nrows(), M.ncols() ) );
                else
                    prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
                                   double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
                                   hpro::to_string( "%d", M.nrows() ) );
                
                prn.restore();
            }// if
        }// if
        else if ( is_lowrank( M ) )
        {
            auto  R = cptrcast( &M, hpro::TRkMatrix );

            // background
            prn.set_rgb( 225, 247, 204 ); // Chameleon1!25!White
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );

            if ( std::find( options.cbegin(), options.cend(), "norank" ) == options.end() )
            {
                prn.save();
                prn.set_font( "Helvetica", std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 4.0 ) );
                
                prn.set_gray( 0 );
                prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
                               double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
                               hpro::to_string( "%d", R->rank() ) );
                
                prn.restore();
            }// if
        }// if
        else if ( is_uniform_lowrank( M ) )
        {
            auto  R = cptrcast( &M, matrix::uniform_lrmatrix< hpro::real > );

            // background
            prn.set_rgb( 219, 231, 243 ); // SkyBlue1!25!White
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );

            if ( std::find( options.cbegin(), options.cend(), "norank" ) == options.end() )
            {
                prn.save();
                prn.set_font( "Helvetica", std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 4.0 ) );
                
                prn.set_gray( 0 );
                prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
                               double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
                               hpro::to_string( "%dx%d", R->row_rank(), R->col_rank() ) );
                
                prn.restore();
            }// if
        }// if

        // draw frame
        prn.set_gray( 0 );
        prn.draw_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );
                       
    }// else

    // draw ID
    if ( std::find( options.cbegin(), options.cend(), "noid" ) == options.end() )
    {
        prn.save();

        const auto  fn_size = std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 8.0 );
        const auto  text    = hpro::to_string( "%d", M.id() );
    
        prn.set_font( "Helvetica", fn_size );
        prn.set_rgb( 32, 74, 135 ); // SkyBlue3
        prn.draw_text( double(M.col_ofs()) + double(M.cols()) / 2.0 - fn_size * text.length() / 4.0,
                       double(M.row_ofs()) + double(M.rows()) / 2.0 + fn_size / 3.0,
                       text );
        prn.restore();
    }// if
}

//
// actual print function
//
void
print_mem ( const hpro::TMatrix &               M,
            eps_printer &                       prn,
            const std::vector< std::string > &  options )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix );

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
        if ( is_dense( M ) )
        {
            // background
            prn.set_gray( 0 );
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );
        }// if
        else if ( is_lowrank( M ) )
        {
            auto  R     = cptrcast( &M, hpro::TRkMatrix );
            auto  rank  = R->rank();
            auto  ratio = ( rank * double( R->nrows() + R->ncols() ) ) / ( double(R->nrows()) * double(R->ncols()) );
            
            // background
            prn.set_gray( 255 - int( ratio * 255.0 ) );
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );
        }// if
        else if ( is_uniform_lowrank( M ) )
        {
            auto  R        = cptrcast( &M, matrix::uniform_lrmatrix< hpro::real > );
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
void
print_eps ( const hpro::TMatrix &  M,
            const std::string &    filename,
            const std::string &    options )
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
void
print_lvl_eps ( const hpro::TMatrix &  M,
                const std::string &    basename,
                const std::string &    options )
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

    auto  parents = std::list< const hpro::TMatrix * >();
    auto  blocks  = std::list< const hpro::TMatrix * >{ & M };
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

        auto  sons = std::list< const hpro::TMatrix * >();

        for ( auto  M_i : blocks )
        {
            if ( is_blocked( M_i ) )
            {
                auto  B_i = cptrcast( M_i, hpro::TBlockMatrix );

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
void
print_mem_eps ( const hpro::TMatrix &  M,
                const std::string &    filename,
                const std::string &    options )
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

}}// namespace hlr::matrix

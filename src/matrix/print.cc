//
// Project     : HLR
// Module      : matrix/print
// Description : printing functions for matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <fstream>

#include <boost/filesystem.hpp>

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
// colors
//
void
print_eps ( const hpro::TMatrix &  M,
            eps_printer &          prn )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    print_eps( * B->block( i, j ), prn );
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

            {
                prn.save();
                prn.set_font( "Helvetica", std::max( 1.0, double( std::min(M.nrows(),M.ncols()) ) / 4.0 ) );
                
                prn.set_gray( 0 );
                prn.draw_text( double(M.col_ofs()) + (double(M.cols()) / 14.0),
                               double(M.row_ofs() + M.rows()) - (double(M.rows()) / 14.0),
                               hpro::to_string( "%dx%d", M.nrows(), M.ncols() ) );
                
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

            if ( R->rank() > 0 )
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

            if ( R->rank() > 0 )
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
}

}// namespace anonymous

//
// print matrix <M> to file <filename>
//
void
print_eps ( const hpro::TMatrix &  M,
            const std::string &    filename )
{
    const boost::filesystem::path  filepath( filename );
    std::string                    suffix;

    if ( ! filepath.has_extension() )
        suffix = ".eps";
    
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

    print_eps( M, prn );

    prn.end();
}

}}// namespace hlr::matrix

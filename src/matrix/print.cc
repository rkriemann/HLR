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
#include <hpro/io/TClusterBasisVis.hh>

#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/utils/eps_printer.hh>
#include <hlr/utils/tools.hh>

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

            if ( ! contains( options, "nosize" ) )
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

            if ( ! contains( options, "norank" ) )
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

            if ( ! contains( options, "norank" ) )
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
        else if ( is_sparse( M ) )
        {
            auto  S = cptrcast( &M, hpro::TSparseMatrix );
            
            // background
            prn.set_rgb( 253, 250, 167 ); // Butter1!50!White 242,229,188
            prn.fill_rect( M.col_ofs(),
                           M.row_ofs(),
                           M.col_ofs() + M.ncols(),
                           M.row_ofs() + M.nrows() );

            if ( contains( options, "pattern" ) )
            {
                prn.set_gray( 0 );

                for ( uint  i = 0; i < S->nrows(); ++i )
                {
                    const auto  lb = S->rowptr( i );
                    const auto  ub = S->rowptr( i+1 );
                    
                    for ( uint  l = lb; l < ub; ++l )
                        prn.fill_rect( S->colind(l),   i,
                                       S->colind(l)+1, i+1 );
                }// for
            }// if
        }// else
        
        // draw frame
        prn.set_gray( 0 );
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

//
// print cluster basis <cl> to file <filename>
//
template < typename cluster_basis_t >
void
print_eps ( const cluster_basis_t &  cb,
            const std::string &      filename,
            const std::string &      options )
{
    using value_t = typename cluster_basis_t::value_t;
    using real_t  = hpro::real_type_t< value_t >;
    
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
    auto          bases     = std::list< const cluster_basis_t * >{ & cb };
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
                if ( clb->son( i ) != nullptr )
                    sons.push_back( clb->son( i ) );
            }// for
        }// while

        bases = std::move( sons );

        ++level;
    }// while
    
    prn.end();
}

template void print_eps< cluster_basis< float > >                  ( const cluster_basis< float > & ,
                                                                     const std::string &,
                                                                     const std::string & );

template void print_eps< cluster_basis< double > >                 ( const cluster_basis< double > & ,
                                                                     const std::string &,
                                                                     const std::string & );

template void print_eps< cluster_basis< std::complex< float > > >  ( const cluster_basis< std::complex< float > > & ,
                                                                     const std::string &,
                                                                     const std::string & );

template void print_eps< cluster_basis< std::complex< double > > > ( const cluster_basis< std::complex< double > > & ,
                                                                     const std::string &,
                                                                     const std::string & );

template <>
void
print_eps< hpro::TClusterBasis< hpro::real > > ( const hpro::TClusterBasis< hpro::real > &  cb,
                                                 const std::string &                        filename,
                                                 const std::string &                        /* options */ )
{
    hpro::TPSClusterBasisVis< hpro::real >  cbvis;

    cbvis.print( &cb, filename );
}

template <>
void
print_eps< hpro::TClusterBasis< hpro::complex > > ( const hpro::TClusterBasis< hpro::complex > &  cb,
                                                    const std::string &                           filename,
                                                    const std::string &                           /* options */ )
{
    hpro::TPSClusterBasisVis< hpro::complex >  cbvis;

    cbvis.print( &cb, filename );
}

}}// namespace hlr::matrix

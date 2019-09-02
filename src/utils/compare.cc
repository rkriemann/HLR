//
// Project     : HLib
// File        : compare.cc
// Description : comparison function for matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <iostream>

// #include <experimental/filesystem>

// namespace fs = std::experimental::filesystem;

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#include <algebra/mat_norm.hh>
#include <matrix/TBlockMatrix.hh>
#include <matrix/structure.hh>
#include <io/TMatrixIO.hh>

#include "hlr/utils/term.hh"
#include "hlr/utils/compare.hh"

namespace hlr
{

using namespace HLIB;

//
// compare <A> with reference read from file <filename>
//
void
compare_ref_file ( TMatrix *            A,
                   const std::string &  filename,
                   const double         error )
{
    if ( fs::exists( filename ) )
    {
        auto  D  = read_matrix( filename );
        auto  BA = ptrcast( A,       TBlockMatrix );
        auto  BD = ptrcast( D.get(), TBlockMatrix );

        if (( A->block_is() == D->block_is() ) &&
            ( BA->nrows() == BD->nrows() ) &&
            ( BA->ncols() == BD->ncols() ) &&
            ( BA->nblock_rows() == BD->nblock_rows() ) &&
            ( BA->nblock_cols() == BD->nblock_cols() ))
        {
            bool  correct = true;
                
            // D->set_procs( ps_single( pid ), recursive );
        
            for ( uint i = 0; i < BA->nblock_rows(); ++i )
            {
                for ( uint j = 0; j < BA->nblock_cols(); ++j )
                {
                    if ( ! is_ghost( BA->block( i, j ) ) )
                    {
                        const auto  f = diff_norm_F( BD->block( i, j ), BA->block( i, j ) );

                        if ( f > error )
                        {
                            std::cout << term::ltred
                                      << HLIB::to_string( "%2d,%2d : %.6e", i, j, diff_norm_F( BD->block( i, j ), BA->block( i, j ) ) )
                                      << term::reset << std::endl;
                            correct = false;
                        }// if
                    }// if
                }// for
            }// for

            if ( correct )
                std::cout << term::ltgreen << "no error" << term::reset << std::endl;
        }// if
        else
        {
            std::cout << term::ltred << "different block structure" << term::reset << std::endl;

            std::cout << BA->block_is().to_string() << std::endl
                      << BA->nblock_rows() << " x " << BA->nblock_cols() << std::endl;
            std::cout << BD->block_is().to_string() << std::endl
                      << BD->nblock_rows() << " x " << BD->nblock_cols() << std::endl;
        }// else
    }// if
    else
        std::cout << "no reference matrix found" << std::endl;
}

}// namespace hlr

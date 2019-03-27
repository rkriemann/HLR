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

#include "utils/compare.hh"

namespace HLR
{

using namespace HLIB;

//
// compare <A> with reference read from file <filename>
//
void
compare_ref_file ( TMatrix *            A,
                   const std::string &  filename )
{
    // mpi::communicator  world;
    // const auto         pid = world.rank();

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

                        if ( f > 1e-10 )
                        {
                            DBG::printf( "%2d,%2d : %.6e", i, j, diff_norm_F( BD->block( i, j ), BA->block( i, j ) ) );
                            correct = false;
                        }// if
                    }// if
                }// for
            }// for

            if ( correct )
                std::cout << "    no error" << std::endl;
        }// if
    }// if
    else
        std::cout << "    no reference matrix found" << std::endl;
}

}// namespace HLR

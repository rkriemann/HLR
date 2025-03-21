//
// Project     : HLR
// Module      : compare.cc
// Description : comparison function for matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <iostream>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#include <hpro/io/TMatrixIO.hh>

#include "hlr/arith/norm.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/term.hh"
#include "hlr/utils/compare.hh"

namespace hlr
{

namespace
{

template < typename value_t >
bool
compare ( const Hpro::TMatrix< value_t > &  A,
          const Hpro::TMatrix< value_t > &  B,
          const double                      error )
{
    if ( is_blocked_all( A, B ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );

        if (( BA->block_is()    == BB->block_is() ) &&
            ( BA->nblock_rows() == BB->nblock_rows() ) &&
            ( BA->nblock_cols() == BB->nblock_cols() ))
        {
            bool  correct = true;
                
            for ( uint i = 0; i < BA->nblock_rows(); ++i )
            {
                for ( uint j = 0; j < BA->nblock_cols(); ++j )
                {
                    if ( ! is_ghost( BA->block( i, j ) ) )
                    {
                        if ( ! compare( *BA->block( i, j ), *BB->block( i, j ), error ) )
                            correct = false;
                    }// if
                }// for
            }// for
            
            return correct;
        }// if
        else
        {
            std::cout << term::ltred << "different block structure" << term::reset << std::endl;

            std::cout << BA->block_is().to_string() << std::endl
                      << BA->nblock_rows() << " x " << BA->nblock_cols() << std::endl;
            std::cout << BB->block_is().to_string() << std::endl
                      << BB->nblock_rows() << " x " << BB->nblock_cols() << std::endl;

            return false;
        }// else
    }// if
    else
    {
        const auto  norm_diff = hlr::norm::frobenius( 1.0, B, -1.0, A );
        const auto  norm_B    = hlr::norm::frobenius( B );

        if ( norm_diff / norm_B > error )
        {
            std::cout << term::ltred << HLIB::to_string( "%2d : %.6e", A.id(), norm_diff / norm_B ) << term::reset << std::endl;

            return false;
        }// if
        else
            return true;
    }// else
}

}// namespace anonymous

//
// compare <A> with reference read from file <filename>
//
template < typename value_t >
void
compare_ref_file ( Hpro::TMatrix< value_t > *  A,
                   const std::string &         filename,
                   const double                error )
{
    if ( fs::exists( filename ) )
    {
        auto  D = Hpro::read_matrix< value_t >( filename );

        if ( compare( *A, *D, error ) )
            std::cout << term::ltgreen << "no error" << term::reset << std::endl;
    }// if
    else
        std::cout << "no reference matrix found" << std::endl;
}

}// namespace hlr

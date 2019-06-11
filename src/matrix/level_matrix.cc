//
// Project     : HLR
// File        : level_matrix.cc
// Description : block matrix for full level of H-matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <iostream>
#include <list>
#include <unordered_set>

#include <matrix/structure.hh>

#include "hlr/utils/tools.hh"

#include "hlr/matrix/level_matrix.hh"

//
// hash function for HLIB::TIndexSet
//
namespace std
{

template <>
struct hash< HLIB::TIndexSet >
{
    size_t operator () ( const HLIB::TIndexSet &  is ) const
    {
        return ( std::hash< HLIB::idx_t >()( is.first() ) +
                 std::hash< HLIB::idx_t >()( is.last()  ) );
    }
};

}// namespace std

namespace hlr { namespace matrix {

using std::list;
using std::unordered_set;

using namespace HLIB;

namespace
{

//
// compare function for TIndexSet
// - only works for sets of disjoint index sets
//
bool
cmp_is ( const TIndexSet &  is1,
         const TIndexSet &  is2 )
{
    return is1.is_strictly_left_of( is2 );
}

}// namespace anonymous

//
// ctor
//

level_matrix::level_matrix ( const uint         nrows,
                             const uint         ncols,
                             const TIndexSet &  rowis,
                             const TIndexSet &  colis )
        : TBlockMatrix( bis( rowis, colis ) )
        , _above( nullptr )
        , _below( nullptr )
{
    set_block_struct( nrows, ncols );
}

//
// construct set of level matrices for given H-matrix
// - do BFS in A and create level matrices starting at first level with leaves
//
std::unique_ptr< level_matrix >
construct_lvlhier ( TMatrix &  A )
{
    list< TMatrix * >  matrices{ & A };
    bool               reached_leaves = ! is_blocked( A );

    while ( ! matrices.empty() )
    {
        ////////////////////////////////////////////////////
        //
        // construct level matrix for current level
        //
        ////////////////////////////////////////////////////

        if ( reached_leaves )
        {
            //
            // collect all row/column indexsets to determine number of rows/columns
            //

            list< TIndexSet >  rowis, colis;
            uint               nrowis, ncolis;

            for ( auto  M : matrices )
            {
                if ( ! contains( rowis, M->row_is() ) )
                {
                    rowis.push_back( M->row_is() );
                    nrowis++;
                }// if
                
                if ( ! contains( colis, M->col_is() ) )
                {
                    colis.push_back( M->col_is() );
                    ncolis++;
                }// if
            }// for

            rowis.sort( cmp_is );
            colis.sort( cmp_is );

            std::cout << "rowis : " << to_string( rowis ) << std::endl;
            std::cout << "colis : " << to_string( colis ) << std::endl;
        }// if
        
        ////////////////////////////////////////////////////
        //
        // collect matrix blocks on next level
        //
        ////////////////////////////////////////////////////
        
        list< TMatrix * >  subs;

        while ( ! matrices.empty() )
        {
            auto  M = behead( matrices );
            
            if ( is_blocked( M ) )
            {
                auto  B = ptrcast( M, TBlockMatrix );

                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < B->nblock_cols(); ++j )
                    {
                        auto  B_ij = B->block( i, j );

                        if ( B_ij != nullptr )
                        {
                            subs.push_back( B_ij );

                            if ( is_leaf( B_ij ) )
                                reached_leaves = true;
                        }// if
                    }// for
                }// for
            }// if
        }// while

        matrices = std::move( subs );
    }// while

    return nullptr;
}

}}// namespace hlr::matrix

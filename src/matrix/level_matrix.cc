//
// Project     : HLR
// File        : level_matrix.cc
// Description : block matrix for full level of H-matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <iostream>
#include <list>
#include <unordered_map>

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
using std::unordered_map;

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
// - do BFS in A and create level matrices (INACTIVE: starting at first level with leaves)
//
std::vector< std::shared_ptr< level_matrix > >
construct_lvlhier ( TMatrix &  A )
{
    list< TMatrix * >  matrices{ & A };
    // bool               reached_leaves = ! is_blocked( A );
    auto               L_hier = list< std::shared_ptr< level_matrix > >{};
    auto               L_prev = std::shared_ptr< level_matrix >{};

    while ( ! matrices.empty() )
    {
        ////////////////////////////////////////////////////
        //
        // construct level matrix for current level
        //
        ////////////////////////////////////////////////////

        // if ( reached_leaves )
        {
            //
            // collect all row/column indexsets to determine number of rows/columns
            //

            list< TIndexSet >  rowis, colis;
            uint               nrowis = 0, ncolis = 0;

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

            //
            // set up index positions of row/col indexsets
            //
            
            unordered_map< TIndexSet, uint >  rowmap, colmap;
            
            rowis.sort( cmp_is );
            colis.sort( cmp_is );

            // std::cout << "rowis : " << to_string( rowis ) << std::endl;
            // std::cout << "colis : " << to_string( colis ) << std::endl;
            
            uint  pos = 0;
            
            for ( auto  is : rowis )
                rowmap[ is ] = pos++;

            pos = 0;
            
            for ( auto  is : colis )
                colmap[ is ] = pos++;

            //
            // construct level matrix
            //

            auto  L = std::make_shared< level_matrix >( nrowis, ncolis, A.row_is(), A.col_is() );

            for ( auto  M : matrices )
            {
                const auto  i = rowmap[ M->row_is() ];
                const auto  j = colmap[ M->col_is() ];

                L->set_block( i, j, M );
            }// for

            L_hier.push_back( L );
            
            L->set_above( L_prev );
            
            if ( L_prev.get() != nullptr )
                L_prev->set_below( L );

            L_prev = L;
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

                            // if ( is_leaf( B_ij ) )
                            //     reached_leaves = true;
                        }// if
                    }// for
                }// for
            }// if
        }// while

        matrices = std::move( subs );
    }// while

    //
    // convert list to vector
    //

    std::vector< std::shared_ptr< level_matrix > >  L_vec;

    L_vec.reserve( L_hier.size() );

    for ( auto  L : L_hier )
        L_vec.push_back( L );
    
    return L_vec;
}

}}// namespace hlr::matrix

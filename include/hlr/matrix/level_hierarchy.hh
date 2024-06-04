#ifndef __HLR_MATRIX_LEVELHIERARCHY_HH
#define __HLR_MATRIX_LEVELHIERARCHY_HH
//
// Project     : HLR
// Module      : level_hierarchy.hh
// Description : block special hierarchy representation for H-matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>

namespace hlr { namespace matrix {

//
// level wise representation of an H-matrix
//
template < typename T_value >
struct level_hierarchy
{
    using  value_t = T_value;

    //
    // per level, a list of leaf blocks for each cluster is stored
    //
    // access:  [ level ][ cluster idx ] -> list of blocks
    //
    
    std::deque< std::deque< std::list< const Hpro::TMatrix< value_t > * > > >  row_hier;
    std::deque< std::deque< std::list< const Hpro::TMatrix< value_t > * > > >  col_hier;
};

template < typename value_t >
level_hierarchy< value_t >
build_level_hierarchy ( const Hpro::TMatrix< value_t > &  M )
{
    auto  hier     = level_hierarchy< value_t >();
    bool  finished = false;
    auto  current  = std::deque< std::list< const Hpro::TMatrix< value_t > * > >();
    uint  lvl      = 0;

    if ( is_blocked( M ) )
        current.push_back( { & M } );
    else
    {
        hier.row_hier.resize( 1 );
        hier.row_hier[0].push_back( { & M } );
    }// else

    while ( current.size() > 0 )
    {
        auto    next = std::deque< std::list< const Hpro::TMatrix< value_t > * > >();
        size_t  idx  = 0;

        for ( auto  mat_list : current )
        {
            //
            // <mat_list> is list of matrices for single row indexset
            //

            uint  inc = 0;
            
            for ( auto  mat : mat_list )
            {
                if ( is_blocked( mat ) )
                {
                    auto  B = cptrcast( mat, Hpro::TBlockMatrix< value_t > );

                    for ( uint  i = 0; i < B->nblock_rows(); ++i )
                    {
                        for ( uint  j = 0; j < B->nblock_cols(); ++j )
                        {
                            auto  B_ij = B->block( i, j );
                            
                            if ( ! is_null( B_ij ) )
                            {
                                if ( is_blocked( B_ij ) )
                                {
                                    if ( idx+i >= next.size() )
                                        next.resize( idx+i+1 );
                                    
                                    next[idx+i].push_back( B_ij );
                                }// if
                                else
                                {
                                    if ( lvl >= hier.row_hier.size() )
                                        hier.row_hier.resize( lvl+1 );

                                    if ( idx+i >= hier.row_hier[lvl].size() )
                                        hier.row_hier[lvl].resize( idx+i+1 );
                                        
                                    hier.row_hier[lvl][idx+i].push_back( B_ij );
                                }// else
                                
                                inc = std::max< uint >( inc, i );
                            }// if
                        }// for
                    }// for
                }// if
            }// for

            idx += inc + 1;
        }// for

        current = std::move( next );
        lvl++;
    }// while

    return hier;
}

template < typename value_t >
void
print_level_hierarchy ( const level_hierarchy< value_t > &  H )
{
    uint  lvl_idx = 0;
    
    for ( auto lvl : H.row_hier )
    {
        uint  row_idx = 0;
        
        std::cout << lvl_idx++ << std::endl;
        
        for ( auto row : lvl )
        {
            std::cout << "  " << row_idx++ << std::endl;

            std::cout << "    ";
            for ( auto mat : row )
                std::cout << mat->typestr() << " " << mat->row_is().to_string() << " × " << mat->col_is().to_string() << ", ";
            std::cout << std::endl;
        }// for
    }// for
}

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_LEVELHIERARCHY_HH

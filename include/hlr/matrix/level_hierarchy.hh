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

#include <hlr/utils/term.hh>

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
    // access:  [ level ][ cluster idx ] -> list of blocks with shared row/column cluster
    //
    
    std::deque< std::deque< std::list< const Hpro::TMatrix< value_t > * > > >  row_hier;
    std::deque< std::deque< std::list< const Hpro::TMatrix< value_t > * > > >  col_hier;

    // return number of level in hierarchy
    uint  nlevel () const { return row_hier.size(); }
};

template < typename value_t >
level_hierarchy< value_t >
build_level_hierarchy ( const Hpro::TMatrix< value_t > &  M )
{
    auto  hier        = level_hierarchy< value_t >();
    auto  row_current = std::deque< std::list< const Hpro::TMatrix< value_t > * > >();
    auto  col_current = std::deque< std::list< const Hpro::TMatrix< value_t > * > >();
    uint  lvl         = 0;

    if ( is_blocked( M ) )
    {
        row_current.push_back( { & M } );
        col_current.push_back( { & M } );
    }// if
    else
    {
        hier.row_hier.resize( 1 );
        hier.row_hier[0].push_back( { & M } );
        hier.col_hier.resize( 1 );
        hier.col_hier[0].push_back( { & M } );
    }// else

    while ( row_current.size() + col_current.size() > 0 )
    {
        auto    row_next = std::deque< std::list< const Hpro::TMatrix< value_t > * > >();
        auto    col_next = std::deque< std::list< const Hpro::TMatrix< value_t > * > >();
        size_t  row_idx  = 0;
        size_t  col_idx  = 0;

        for ( auto  mat_list : row_current )
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
                                    if ( row_idx+i >= row_next.size() )
                                        row_next.resize( row_idx+i+1 );
                                    
                                    row_next[row_idx+i].push_back( B_ij );
                                }// if
                                else
                                {
                                    if ( lvl >= hier.row_hier.size() )
                                        hier.row_hier.resize( lvl+1 );

                                    if ( row_idx+i >= hier.row_hier[lvl].size() )
                                        hier.row_hier[lvl].resize( row_idx+i+1 );
                                        
                                    hier.row_hier[lvl][row_idx+i].push_back( B_ij );
                                }// else
                                
                                inc = std::max< uint >( inc, i );
                            }// if
                        }// for
                    }// for
                }// if
            }// for

            row_idx += inc + 1;
        }// for

        row_current = std::move( row_next );

        //
        // same for columns
        //
        
        for ( auto  mat_list : col_current )
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

                    for ( uint  j = 0; j < B->nblock_cols(); ++j )
                    {
                        for ( uint  i = 0; i < B->nblock_rows(); ++i )
                        {
                            auto  B_ij = B->block( i, j );
                            
                            if ( ! is_null( B_ij ) )
                            {
                                if ( is_blocked( B_ij ) )
                                {
                                    if ( col_idx+j >= col_next.size() )
                                        col_next.resize( col_idx+j+1 );
                                    
                                    col_next[col_idx+j].push_back( B_ij );
                                }// if
                                else
                                {
                                    if ( lvl >= hier.col_hier.size() )
                                        hier.col_hier.resize( lvl+1 );

                                    if ( col_idx+j >= hier.col_hier[lvl].size() )
                                        hier.col_hier[lvl].resize( col_idx+j+1 );
                                        
                                    hier.col_hier[lvl][col_idx+j].push_back( B_ij );
                                }// else
                                
                                inc = std::max< uint >( inc, i );
                            }// if
                        }// for
                    }// for
                }// if
            }// for

            col_idx += inc + 1;
        }// for

        col_current = std::move( col_next );

        lvl++;
    }// while

    return hier;
}

template < typename value_t >
void
print ( const level_hierarchy< value_t > &  H )
{
    uint  lvl_idx = 0;
    
    for ( auto lvl : H.row_hier )
    {
        uint  row_idx = 0;
        
        std::cout << lvl_idx++ << std::endl;
        
        for ( auto row : lvl )
        {
            bool      first = true;
            indexset  rowis;
            
            std::cout << "  " << row_idx++ << std::endl;

            std::cout << "    (" << row.size() << ") ";
            for ( auto mat : row )
            {
                auto  T = mat->typestr()[0];
                
                if ( first )
                {
                    rowis = mat->row_is();
                    first = false;

                    if ( T == 'd' )
                        std::cout << rowis.to_string() << " × " << term::red() << 'D' << mat->col_is().to_string() << term::reset() << ", ";
                    else
                        std::cout << rowis.to_string() << " × " << term::green() << 'U' << mat->col_is().to_string() << term::reset() << ", ";
                }// if
                else
                {
                    HLR_ASSERT( mat->row_is() == rowis );
                    
                    if ( T == 'd' )
                        std::cout << term::red() << 'D' << mat->col_is().to_string() << term::reset() << ", ";
                    else
                        std::cout << term::green() << 'U' << mat->col_is().to_string() << term::reset() << ", ";
                }// else
            }// for
            std::cout << std::endl;
        }// for
    }// for

    lvl_idx = 0;
    
    for ( auto lvl : H.col_hier )
    {
        uint  col_idx = 0;
        
        std::cout << lvl_idx++ << std::endl;
        
        for ( auto col : lvl )
        {
            bool      first = true;
            indexset  colis;
            
            std::cout << "  " << col_idx++ << " (" << col.size() << ") " << std::endl;
            
            for ( auto mat : col )
            {
                auto  T = mat->typestr()[0];

                if ( first )
                {
                    colis = mat->col_is();
                    first = false;

                    if ( T == 'd' )
                        std::cout << colis.to_string() << " × " << term::red() << 'D' << mat->row_is().to_string() << term::reset() << ", ";
                    else
                        std::cout << colis.to_string() << " × " << term::green() << 'U' << mat->row_is().to_string() << term::reset() << ", ";
                }// if
                else
                {
                    HLR_ASSERT( mat->col_is() == colis );
                    
                    if ( T == 'd' )
                        std::cout << term::red() << 'D' << mat->row_is().to_string() << term::reset() << ", ";
                    else
                        std::cout << term::green() << 'U' << mat->row_is().to_string() << term::reset() << ", ";
                }// else
            }// for
            std::cout << std::endl;
        }// for
    }// for
}

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_LEVELHIERARCHY_HH

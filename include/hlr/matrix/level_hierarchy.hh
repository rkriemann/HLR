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

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/lrsvmatrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/uniform_lr2matrix.hh>
#include <hlr/utils/term.hh>

namespace hlr { namespace matrix {

//
// level wise representation of the leaf blocks in an H-matrix
//
template < typename T_value >
struct level_hierarchy
{
    using  value_t = T_value;

    // CRS style row-wise storage
    std::vector< std::vector< idx_t > >                             row_ptr;
    std::vector< std::vector< idx_t > >                             col_idx;
    std::vector< std::vector< const Hpro::TMatrix< value_t > * > >  row_mat;

    //
    // ctor
    //

    level_hierarchy ( const uint  nlvl )
            : row_ptr( nlvl )
            , col_idx( nlvl )
            , row_mat( nlvl )
    {}
    
    // return number of level in hierarchy
    uint  nlevel () const { return row_ptr.size(); }
};

template < typename value_t >
level_hierarchy< value_t >
build_level_hierarchy ( const Hpro::TMatrix< value_t > &  M,
                        const bool                        transposed = false )
{
    using  matrix_t = const Hpro::TMatrix< value_t >;
    
    const auto  nlvl    = Hpro::get_nlevel( M );
    auto        hier    = level_hierarchy< value_t >( nlvl );
    auto        row_ptr = std::vector< idx_t >();       // CRS data including structured matrices
    auto        col_idx = std::vector< idx_t >();
    auto        row_mat = std::vector< matrix_t * >();
    uint        lvl     = 0;
    uint        nleaves = 0;
    const auto  op_M    = ( transposed ? apply_transposed : apply_normal );

    if ( is_blocked( M ) )
    {
        row_ptr.resize( 2 );
        col_idx.resize( 1 );
        row_mat.resize( 1 );

        row_ptr[0] = 0;
        row_ptr[1] = 1;
        col_idx[0] = 0;
        row_mat[0] = & M;
    }// if
    else
    {
        //
        // single level, single block data
        //
        
        hier.row_ptr[0].resize( 2 );
        hier.row_ptr[0][0] = 0;
        hier.row_ptr[0][1] = 1;
        
        hier.col_idx[0].resize( 1 );
        hier.col_idx[0][0] = 0;
        
        hier.row_mat[0].resize( 1 );
        hier.row_mat[0][0] = & M;
        
        // hier.col_hier.resize( 1 );
        // hier.col_hier[0].push_back( { & M } );
    }// else

    while ( row_mat.size() > 0 )
    {
        //
        // store leaves and count matrices for next level
        //

        idx_t   pos   = 0;
        size_t  nnext = 0;
        uint    nrows = 0;
        
        hier.row_ptr[lvl].resize( row_ptr.size() );
        hier.col_idx[lvl].resize( nleaves );
        hier.row_mat[lvl].resize( nleaves );

        hier.row_ptr[lvl][0] = 0;
        
        for ( uint  i = 0; i < row_ptr.size()-1; ++i )
        {
            const auto  lb       = row_ptr[i];
            const auto  ub       = row_ptr[i+1];
            uint        nsubrows = 0;
            
            for ( uint  j = lb; j < ub; ++j )
            {
                auto  mat = row_mat[j];
                
                if ( ! is_blocked( mat ) )
                {
                    hier.col_idx[lvl][pos] = col_idx[j];
                    hier.row_mat[lvl][pos] = mat;
                    ++pos;
                }// if
                else
                {
                    auto  B = cptrcast( mat, Hpro::TBlockMatrix< value_t > );

                    HLR_ASSERT( B->nblock_rows() == 2 );
                    HLR_ASSERT( B->nblock_cols() == 2 );
                                                       
                    nsubrows = std::max( nsubrows, B->nblock_rows( op_M ) );
                    
                    for ( uint  ii = 0; ii < B->nblock_rows( op_M ); ++ii )
                        for ( uint  jj = 0; jj < B->nblock_cols( op_M ); ++jj )
                            if ( ! is_null( B->block( ii, jj, op_M ) ) )
                                nnext++;
                }// else
            }// for

            nrows += nsubrows;
            hier.row_ptr[lvl][i+1] = pos;
        }// for

        if ( nnext == 0 )
            break;
        
        //
        // set up data for next level
        //
        
        auto  next_row_ptr = std::vector< idx_t >();       // CRS data including structured matrices
        auto  next_col_idx = std::vector< idx_t >();
        auto  next_row_mat = std::vector< matrix_t * >();
        uint  row_idx      = 0;

        next_row_ptr.resize( nrows + 1 );
        next_col_idx.resize( nnext );
        next_row_mat.resize( nnext );

        // count entries per block row
        for ( uint  i = 0; i < row_ptr.size()-1; ++i )
        {
            const auto  lb = row_ptr[i];
            const auto  ub = row_ptr[i+1];
            uint        nsubrows = 0;

            for ( uint  j = lb; j < ub; ++j )
            {
                auto  mat = row_mat[j];
                
                if ( is_blocked( mat ) )
                {
                    auto  B = cptrcast( mat, Hpro::TBlockMatrix< value_t > );
                    
                    nsubrows = std::max( nsubrows, B->nblock_rows( op_M ) );
                    
                    for ( uint  ii = 0; ii < B->nblock_rows( op_M ); ++ii )
                        for ( uint  jj = 0; jj < B->nblock_cols( op_M ); ++jj )
                            if ( ! is_null( B->block( ii, jj, op_M ) ) )
                                next_row_ptr[ row_idx+ii ]++;
                }// else
            }// for

            row_idx += nsubrows;
        }// for

        pos = 0;

        // sum up actual row_ptr data
        for ( uint  i = 0; i < next_row_ptr.size()-1; ++i )
        {
            const auto  tmp = next_row_ptr[i];

            next_row_ptr[i] = pos;
            pos            += tmp;
        }// for
        next_row_ptr[ nrows ] = pos;

        auto  rowpos = std::vector< idx_t >( nrows );

        row_idx = 0;
        nleaves = 0;
        
        // fill column indices and matrices
        for ( uint  i = 0; i < row_ptr.size()-1; ++i )
        {
            const auto  lb = row_ptr[i];
            const auto  ub = row_ptr[i+1];
            uint        nsubrows = 0;

            for ( uint  j = lb; j < ub; ++j )
            {
                auto  mat = row_mat[j];
                
                if ( is_blocked( mat ) )
                {
                    auto        B   = cptrcast( mat, Hpro::TBlockMatrix< value_t > );
                    const auto  col = col_idx[j];

                    HLR_ASSERT(( B->nblock_rows() == 2 ) && ( B->nblock_cols() == 2 ));
                    
                    nsubrows = std::max( nsubrows, B->nblock_rows( op_M ) );

                    for ( uint  ii = 0; ii < B->nblock_rows( op_M ); ++ii )
                    {
                        for ( uint  jj = 0; jj < B->nblock_cols( op_M ); ++jj )
                        {
                            auto  B_ij = B->block( ii, jj, op_M );
                            
                            if ( ! is_null( B_ij ) )
                            {
                                const idx_t  idx = next_row_ptr[ row_idx+ii ] + rowpos[ row_idx+ii ];
                                    
                                next_col_idx[ idx ] = 2 * col + jj; // assumes 2x2 block structure !!!
                                next_row_mat[ idx ] = B_ij;
                                rowpos[ row_idx+ii ]++;

                                if ( ! is_blocked( B_ij ) )
                                    nleaves++;
                            }// if
                        }// for
                    }// for
                }// else
            }// for

            row_idx += nsubrows;
        }// for

        row_ptr = std::move( next_row_ptr );
        col_idx = std::move( next_col_idx );
        row_mat = std::move( next_row_mat );
        lvl++;
    }// while

    return hier;
}

template < typename value_t >
void
print ( const level_hierarchy< value_t > &  H,
        const bool                          transposed = false )
{
    uint        lvl_idx = 0;
    const auto  op_M    = ( transposed ? apply_transposed : apply_normal );
    
    for ( uint  lvl = 0; lvl < H.nlevel(); ++lvl )
    {
        std::cout << lvl_idx++ << std::endl;

        for ( uint  row = 0; row < H.row_ptr[lvl].size()-1; ++row )
        {
            const auto  lb = H.row_ptr[lvl][row];
            const auto  ub = H.row_ptr[lvl][row+1];

            std::cout << "  " << row << " : ";

            bool      first = true;
            indexset  rowis;

            for ( uint  j = lb; j < ub; ++j )
            {
                auto  col_idx = H.col_idx[lvl][j];
                auto  mat     = H.row_mat[lvl][j];
                auto  T       = mat->typestr()[0];
                
                if ( first )
                {
                    rowis = mat->row_is( op_M );
                    first = false;

                    if ( matrix::is_dense( mat ) )
                        std::cout << rowis.to_string() << " × " << term::red() << col_idx << " D" << mat->id() << term::reset() << ", ";
                    else if ( matrix::is_lowrank( mat ) )
                        std::cout << rowis.to_string() << " × " << term::green() << col_idx << " R" << mat->id() << term::reset() << ", ";
                    else if ( matrix::is_lowrank_sv( mat ) )
                        std::cout << rowis.to_string() << " × " << term::green() << col_idx << " R" << mat->id() << term::reset() << ", ";
                    else if ( matrix::is_uniform_lowrank( mat ) )
                        std::cout << rowis.to_string() << " × " << term::green() << col_idx << " U" << mat->id() << term::reset() << ", ";
                    else if ( matrix::is_uniform_lowrank2( mat ) )
                        std::cout << rowis.to_string() << " × " << term::green() << col_idx << " U" << mat->id() << term::reset() << ", ";
                    else
                        std::cout << rowis.to_string() << " × " << term::blue() << col_idx << " ?" << mat->id() << term::reset() << ", ";
                }// if
                else
                {
                    HLR_ASSERT( mat->row_is( op_M ) == rowis );
                    
                    if ( matrix::is_dense( mat ) )
                        std::cout << term::red() << col_idx << " D" << mat->id() << term::reset() << ", ";
                    else if ( matrix::is_lowrank( mat ) )
                        std::cout << term::green() << col_idx << " R" << mat->id() << term::reset() << ", ";
                    else if ( matrix::is_lowrank_sv( mat ) )
                        std::cout << term::green() << col_idx << " R" << mat->id() << term::reset() << ", ";
                    else if ( matrix::is_uniform_lowrank( mat ) )
                        std::cout << term::green() << col_idx << " U" << mat->id() << term::reset() << ", ";
                    else if ( matrix::is_uniform_lowrank2( mat ) )
                        std::cout << term::green() << col_idx << " U" << mat->id() << term::reset() << ", ";
                    else
                        std::cout << term::blue() << col_idx << " ?" << mat->id() << term::reset() << ", ";
                }// else
            }// for
            std::cout << std::endl;
        }// for
    }// for
}

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_LEVELHIERARCHY_HH

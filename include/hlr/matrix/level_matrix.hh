#ifndef __HLR_MATRIX_LEVELMATRIX_HH
#define __HLR_MATRIX_LEVELMATRIX_HH
//
// Project     : HLR
// Module      : level_matrix.hh
// Description : block matrix for full level of H-matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cassert>
#include <memory>
#include <vector>
#include <map>

#include <hpro/matrix/TBlockMatrix.hh>

namespace hlr { namespace matrix {

// local matrix type
DECLARE_TYPE( level_matrix );

//
// block matrix representing a single, global level
// in the H hierarchy
//
template < typename T_value >
class level_matrix : public Hpro::TBlockMatrix< T_value >
{
public:
    using  value_t      = T_value;
    using  matrix_t     = Hpro::TMatrix< value_t >;
    using  matrix_map_t = std::map< Hpro::idx_t, matrix_t * >;
    
private:
    // pointers to level matrices above and below
    std::shared_ptr< level_matrix >        _above;
    std::shared_ptr< level_matrix >        _below;

    std::map< Hpro::idx_t, matrix_map_t >  _block_rows;
    std::map< Hpro::idx_t, matrix_map_t >  _block_cols;
    
public:
    //
    // ctor
    //

    level_matrix () {}
    
    level_matrix ( const uint               nrows,
                   const uint               ncols,
                   const Hpro::TIndexSet &  rowis,
                   const Hpro::TIndexSet &  colis );

    //
    // give access to level hierarchy
    //

    level_matrix *  above () { return  _above.get(); }
    level_matrix *  below () { return  _below.get(); }

    void  set_above ( std::shared_ptr< level_matrix > &  M ) { _above = M; }
    void  set_below ( std::shared_ptr< level_matrix > &  M ) { _below = M; }

    //
    // index functions
    //

    //! return matrix at index (i,j)
    auto  block ( const uint  i,
                  const uint  j ) -> matrix_t *
    {
        return _block_rows[ i ][ j ];
    }

    //! set matrix block at block index (\a i,\a j) to matrix \a A
    void  set_block ( const uint  i,
                      const uint  j,
                      matrix_t *  A )
    {
        _block_rows[ i ][ j ] = A;
        _block_cols[ j ][ i ] = A;
    }

    //! return block-row iterator to next entry starting at (i,j)
    auto  row_iter ( const uint  i,
                     const uint  j ) -> typename matrix_map_t::iterator
    {
        auto  iter = _block_rows[ i ].begin();
        auto  end  = _block_rows[ i ].end();

        for ( ; iter != end; ++iter )
        {
            if ( (*iter).first >= j )
                return iter;
        }// for

        return end;
    }
    
    //! return block-column iterator to next entry starting at (i,j)
    auto  col_iter ( const uint  i,
                     const uint  j ) -> typename matrix_map_t::iterator
    {
        auto  iter = _block_cols[ j ].begin();
        auto  end  = _block_cols[ j ].end();

        for ( ; iter != end; ++iter )
        {
            if ( (*iter).first >= i )
                return iter;
        }// for

        return end;
    }

    //! return end of block-row i
    auto  row_end ( const uint  i ) -> typename matrix_map_t::iterator
    {
        return _block_rows[ i ].end();
    }
    
    //! return end of block-column j
    auto  col_end ( const uint  j ) -> typename matrix_map_t::iterator
    {
        return _block_cols[ j ].end();
    }
    
    // return block row/column of A
    std::pair< uint, uint >
    get_index ( const matrix_t *  A )
    {
        for ( uint  i = 0; i < this->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < this->nblock_cols(); ++j )
            {
                auto  A_ij = block( i, j );
                
                if ( A_ij == A )
                    return { i, j };
            }// for
        }// for

        return { this->nblock_rows(), this->nblock_cols() };
    }
    std::pair< uint, uint >
    get_index ( const matrix_t &  A )
    {
        return get_index( & A );
    }
    
    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( level_matrix, Hpro::TBlockMatrix< value_t > )

    //! return matrix of same class (but no content)
    virtual auto create () const -> std::unique_ptr< matrix_t >
    {
        return std::make_unique< level_matrix >();
    }

    //! return size in bytes used by this object
    virtual size_t byte_size () const
    {
        return Hpro::TBlockMatrix< value_t >::byte_size() + sizeof( _above ) + sizeof( _below );
    }
};

//
// construct set of level matrices for given H-matrix
//
template < typename value_t >
std::vector< std::shared_ptr< level_matrix< value_t > > >
construct_lvlhier ( Hpro::TMatrix< value_t > &  A );

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LEVELMATRIX_HH

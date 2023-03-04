#ifndef __HLR_MATRIX_ACCUMULATOR_BASE_HH
#define __HLR_MATRIX_ACCUMULATOR_BASE_HH
//
// Project     : HLR
// Module      : accumulator.hh
// Description : implements update accumulator for H-arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>

namespace hlr { namespace matrix {

template < typename value_t >
struct accumulator_base
{
    //
    // represents an update, i.e., matrix product
    //
    struct update
    {
        const matop_t                     op_A;
        const Hpro::TMatrix< value_t > *  A;
        const matop_t                     op_B;
        const Hpro::TMatrix< value_t > *  B;
    };
    
    // represents set of updates
    using  update_list = std::list< update >;

    // accumulated computed updates
    std::unique_ptr< Hpro::TMatrix< value_t > >   matrix;

    // accumulated pending (recursive) updates
    update_list                                   pending;

    //
    // ctors
    //

    accumulator_base ()
    {}
    
    accumulator_base ( std::unique_ptr< Hpro::TMatrix< value_t > > &&  amatrix,
                       update_list &&                                  apending )
            : matrix( std::move( amatrix ) )
            , pending( std::move( apending ) )
    {}
    
    //
    // remove update matrix
    //
    void
    clear_matrix ()
    {
        matrix.reset( nullptr );
    }

    //
    // release matrix
    //
    Hpro::TMatrix< value_t > *
    release_matrix ()
    {
        return matrix.release();
    }
};

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_ACCUMULATOR_BASE_HH

#ifndef __HLR_BEM_DENSE_HH
#define __HLR_BEM_DENSE_HH
//
// Project     : HLR
// Module      : dense.hh
// Description : dense block building
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/algebra/TLowRankApx.hh>

#include <hlr/matrix/dense_matrix.hh>

namespace hlr { namespace bem {

template < typename coeff_fn_t >
class dense_lrapx : public Hpro::TLowRankApx< typename coeff_fn_t::value_t >
{
public:
    using  value_t = typename coeff_fn_t::value_t;
    
private:
    // coefficient function
    const coeff_fn_t &  _coeff_fn;
    
public:
    // ctor
    dense_lrapx ( const coeff_fn_t &  acoeff_fn )
            : _coeff_fn( acoeff_fn )
    {}
        
    // build low rank matrix for block cluster bct with rank defined by accuracy acc
    virtual
    std::unique_ptr< Hpro::TMatrix< value_t > >
    build ( const Hpro::TBlockCluster *   bc,
            const Hpro::TTruncAcc &       acc ) const
    {
        return build( bc->is(), acc );
    }

    virtual
    std::unique_ptr< Hpro::TMatrix< value_t > >
    build ( const Hpro::TBlockIndexSet &  bis,
            const Hpro::TTruncAcc &       /* acc */ ) const
    {
        auto  D = blas::matrix< value_t >( bis.row_is().size(), bis.col_is().size() );

        _coeff_fn.eval( bis.row_is(), bis.col_is(), D.data() );

        return std::make_unique< matrix::dense_matrix< value_t > >( bis.row_is(), bis.col_is(), std::move( D ) );
    }
};


}}// namespace hlr::bem

#endif // __HLR_BEM_DENSE_HH

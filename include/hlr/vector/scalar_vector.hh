#ifndef __HLR_VECTOR_SCALAR_VECTOR_HH
#define __HLR_VECTOR_SCALAR_VECTOR_HH
//
// Project     : HLR
// Module      : scalar_vector.hh
// Description : standard scalar vector
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/vector/TScalarVector.hh>

#include <hlr/arith/blas.hh>

namespace hlr { namespace vector {

//
// import class/functions from HLIBpro but make it template
// in value type
//
template < typename T_value >
class scalar_vector : public hpro::TScalarVector
{
public:
    using  value_t = T_value;

public:
    //
    // ctors
    //
    scalar_vector ()
            : hpro::TScalarVector( hpro::value_type< value_t >::value )
    {}

    scalar_vector ( const indexset &                 ais )
            : hpro::TScalarVector( ais, hpro::value_type< value_t >::value )
    {}

    scalar_vector ( const indexset &                 ais,
                    const blas::vector< value_t > &  av )
            : hpro::TScalarVector( ais, av )
    {}

    scalar_vector ( const indexset &                 ais,
                    blas::vector< value_t > &&       av )
            : hpro::TScalarVector( ais, std::move( av ) )
    {}

    scalar_vector ( const scalar_vector &            av )
            : hpro::TScalarVector( av )
    {}

    scalar_vector ( scalar_vector &&                 av )
            : hpro::TScalarVector( std::move( av ) )
    {}

    //
    // access internal BLAS vectors
    //

    blas::vector< value_t > &        blas_vec  ()       { return hpro::blas_vec< value_t >( *this ); }
    const blas::vector< value_t > &  blas_vec  () const { return hpro::blas_vec< value_t >( *this ); }

    //
    // copy/assign methods with additional type checking
    //
    
    virtual void set_vector ( const blas::vector< real > &     vec,
                              const idx_t                      offset ) { HLR_ASSERT( ! this->is_complex() ); TScalarVector::set_vector( vec, offset ); }

    //! set internal data directly (complex valued)
    virtual void set_vector ( const blas::vector< complex > &  vec,
                              const idx_t                      offset ) { HLR_ASSERT(   this->is_complex() ); TScalarVector::set_vector( vec, offset ); }

    //! copy from vector \a v
    virtual void copy_from  ( const TScalarVector *            v )
    {
        HLR_ASSERT( ! is_null( v ) && ( this->value_type() == v->value_type() ));
        TScalarVector::copy_from( v );
    }

    //! copy to vector \a v
    virtual void copy_to ( TScalarVector * v ) const
    {
        HLR_ASSERT( ! is_null( v ) && ( this->value_type() == v->value_type() ));
        TScalarVector::copy_to( v );
    }

    using TVector::copy_to;
    using TScalarVector::copy_to;
    using TScalarVector::copy_from;

protected:

    //
    // prevent switching of value type
    //
    
    virtual void to_real    () { HLR_ASSERT( ! this->is_complex() ); }
    virtual void to_complex () { HLR_ASSERT(   this->is_complex() ); }
};

using hpro::blas_vec;
using hpro::is_scalar;

}}// namespace hlr::vector

#endif // __HLR_VECTOR_SCALAR_VECTOR_HH

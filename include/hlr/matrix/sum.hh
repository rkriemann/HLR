#ifndef __HLR_MATRIX_SUM_HH
#define __HLR_MATRIX_SUM_HH
//
// Project     : HLR
// Module      : matrix/sum
// Description : provides operator for matrix sums
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/matrix/TLinearOperator.hh>

// #include <hlr/matrix/arithmetic_support.hh>

namespace hlr { namespace matrix {

// local matrix type
DECLARE_TYPE( linop_sum );

//
// represents Σ_i α_i A_i of linear operators
//
template < typename T_value >
class linop_sum
    : public Hpro::TLinearOperator< T_value >
// , public arithmetic_support< linop_sum< T_value > >
{
public:
 
    using  value_t  = T_value;

    // some abbrv.
    using  linop_t  = Hpro::TLinearOperator< value_t >;
    using  vector_t = Hpro::TVector< value_t >;
    using  matrix_t = Hpro::TMatrix< value_t >;
   
private:

    struct summand_t {
        const linop_t *  linop;
        const value_t    scale;
        const matop_t    op;
    };
    
    // summands in matrix sum
    std::vector< summand_t >  _summands;

public:
    //
    // ctors for different numbers of summands
    //

    linop_sum ( const value_t    alpha0,
                const matop_t    op0,
                const linop_t &  A0,
                const value_t    alpha1,
                const matop_t    op1,
                const linop_t &  A1 )
            : _summands{ { &A0, alpha0, op0 },
                         { &A1, alpha1, op1 } }
    {}
    

    linop_sum ( const value_t    alpha0,
                const matop_t    op0,
                const linop_t &  A0,
                const value_t    alpha1,
                const matop_t    op1,
                const linop_t &  A1,
                const value_t    alpha2,
                const matop_t    op2,
                const linop_t &  A2 )
            : _summands{ { &A0, alpha0, op0 },
                         { &A1, alpha1, op1 },
                         { &A2, alpha2, op2 } }
    {}

    virtual ~linop_sum () {}

    //
    // linear operator properties
    //

    // return true, of operator is self adjoint
    virtual bool  is_self_adjoint () const { return false; } // TODO
    
    //
    // linear operator mapping
    //
    
    //
    // compute y = A·x with A being the sum operator
    //
    virtual void  apply       ( const vector_t *  x,
                                vector_t *        y,
                                const matop_t     op = apply_normal ) const;

    //
    // compute y = y + α·A·x with A being the sum operator
    //
    virtual void  apply_add   ( const value_t     alpha,
                                const vector_t *  x,
                                vector_t *        y,
                                const matop_t     op = apply_normal ) const;

    virtual void  apply_add   ( const value_t     alpha,
                                const matrix_t *  X,
                                matrix_t *        Y,
                                const matop_t     op = apply_normal ) const;

    // same as above but for blas types (no indexset test!)
    virtual void  apply_add   ( const value_t                    alpha,
                                const blas::vector< value_t > &  x,
                                blas::vector< value_t > &        y,
                                const matop_t                    op = apply_normal ) const;

    virtual void  apply_add   ( const value_t                    alpha,
                                const blas::matrix< value_t > &  X,
                                blas::matrix< value_t > &        Y,
                                const matop_t                    op = apply_normal ) const;

    // same as above but use given arithmetic object for computation
    template < typename arithmetic_t >
    void          apply_add   ( arithmetic_t &&                  arithmetic,
                                const value_t                    alpha,
                                const blas::vector< value_t > &  x,
                                blas::vector< value_t > &        y,
                                const matop_t                    op = apply_normal ) const;

    ///////////////////////////////////////////////////////////
    //
    // access to vector space elements
    //

    // return dimension of domain
    virtual size_t  domain_dim () const
    {
        HLR_ASSERT( _summands.size() > 0 );
        
        const auto  s = _summands.front();
    
        if ( s.op == apply_normal ) return s.linop->domain_dim();
        else                        return s.linop->range_dim();
    }
    
    // return dimension of range
    virtual size_t  range_dim  () const
    {
        HLR_ASSERT( _summands.size() > 0 );
        
        const auto  s = _summands.front();
        
        if ( s.op == apply_normal ) return s.linop->range_dim();
        else                        return s.linop->domain_dim();
    }
    
    // return vector in domain space
    virtual auto    domain_vector  () const -> std::unique_ptr< vector_t >
    {
        HLR_ASSERT( _summands.size() > 0 );
        
        const auto  s = _summands.front();
        
        if ( s.op == apply_normal ) return s.linop->domain_vector();
        else                        return s.linop->range_vector();
    }

    // return vector in range space
    virtual auto    range_vector   () const -> std::unique_ptr< vector_t >
    {
        HLR_ASSERT( _summands.size() > 0 );
        
        const auto  s = _summands.front();
        
        if ( s.op == apply_normal ) return s.linop->range_vector();
        else                        return s.linop->domain_vector();
    }

    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( linop_sum, Hpro::TLinearOperator< value_t > );
};

//
// enable arithmetic support
//
// SUPPORTS_ARITHMETIC_TEMPLATE( linop_sum )

//
// multiplication functions
//
template < typename value_t >
void
linop_sum< value_t >::apply ( const vector_t *  x,
                              vector_t *        y,
                              const matop_t     op ) const
{
    y->fill( value_t(0) );
    
    for ( auto &  s : _summands )
        s.linop->apply_add( s.scale, x, y, blas::apply_op( op, s.op ) );
}

template < typename value_t >
void
linop_sum< value_t >::apply_add   ( const value_t     alpha,
                                    const vector_t *  x,
                                    vector_t *        y,
                                    const matop_t     op ) const
{
    for ( auto &  s : _summands )
        s.linop->apply_add( s.scale, x, y, blas::apply_op( op, s.op ) );
}

template < typename value_t >
void
linop_sum< value_t >::apply_add   ( const value_t     /* alpha */,
                                    const matrix_t *  /* X */,
                                    matrix_t *        /* Y */,
                                    const matop_t     /* op */ ) const
{
    HLR_ERROR( "TODO" );
}
    
template < typename value_t >
void
linop_sum< value_t >::apply_add   ( const value_t                    alpha,
                                    const blas::vector< value_t > &  x,
                                    blas::vector< value_t > &        y,
                                    const matop_t                    op ) const
{
    for ( auto &  s : _summands )
        s.linop->apply_add( alpha*s.scale, x, y, blas::apply_op( op, s.op ) );
}

template < typename value_t >
void
linop_sum< value_t >::apply_add   ( const value_t                    alpha,
                                    const blas::matrix< value_t > &  X,
                                    blas::matrix< value_t > &        Y,
                                    const matop_t                    op ) const
{
    for ( auto &  s : _summands )
        s.linop->apply_add( alpha*s.scale, X, Y, blas::apply_op( op, s.op ) );
}

template < typename value_t >
template < typename arithmetic_t >
void
linop_sum< value_t >::apply_add   ( arithmetic_t &&                  arithmetic,
                                    const value_t                    alpha,
                                    const blas::vector< value_t > &  x,
                                    blas::vector< value_t > &        y,
                                    const matop_t                    op ) const
{
    for ( auto &  s : _summands )
        arithmetic.prod( alpha * s.scale, blas::apply_op( op, s.op ), * s.linop, x, y );
}

//
// functions to create summation objects
//
template < typename value_t >
std::unique_ptr< linop_sum< value_t > >
sum ( const Hpro::TLinearOperator< value_t > &  A0,
      const Hpro::TLinearOperator< value_t > &  A1 )
{
    return std::make_unique< linop_sum< value_t > >( value_t(1), apply_normal, A0,
                                                     value_t(1), apply_normal, A1 );
}
                 
template < typename alpha0_t,
           typename alpha1_t,
           typename value_t >
std::unique_ptr< linop_sum< value_t > >
sum ( const alpha0_t                            alpha0,
      const Hpro::TLinearOperator< value_t > &  A0,
      const alpha1_t                            alpha1,
      const Hpro::TLinearOperator< value_t > &  A1 )
{
    return std::make_unique< linop_sum< value_t > >( value_t(alpha0), apply_normal, A0,
                                                     value_t(alpha1), apply_normal, A1 );
}
                 
template < typename value_t >
std::unique_ptr< linop_sum< value_t > >
sum ( const value_t                             alpha0,
      const matop_t                             op0,
      const Hpro::TLinearOperator< value_t > &  A0,
      const value_t                             alpha1,
      const matop_t                             op1,
      const Hpro::TLinearOperator< value_t > &  A1 )
{
    return std::make_unique< linop_sum< value_t > >( alpha0, op0, A0,
                                                     alpha1, op1, A1 );
}
    

template < typename value_t >
std::unique_ptr< linop_sum< value_t > >
sum ( const Hpro::TLinearOperator< value_t > &  A0,
      const Hpro::TLinearOperator< value_t > &  A1,
      const Hpro::TLinearOperator< value_t > &  A2 )
{
    return std::make_unique< linop_sum< value_t > >( value_t(1), apply_normal, A0,
                                                     value_t(1), apply_normal, A1,
                                                     value_t(1), apply_normal, A2 );
}
    
template < typename value_t >
std::unique_ptr< linop_sum< value_t > >
sum ( const value_t                             alpha0,
      const Hpro::TLinearOperator< value_t > &  A0,
      const value_t                             alpha1,
      const Hpro::TLinearOperator< value_t > &  A1,
      const value_t                             alpha2,
      const Hpro::TLinearOperator< value_t > &  A2 )
{
    return std::make_unique< linop_sum< value_t > >( alpha0, apply_normal, A0,
                                                     alpha1, apply_normal, A1,
                                                     alpha2, apply_normal, A2 );
}
                 
template < typename value_t >
std::unique_ptr< linop_sum< value_t > >
sum ( const value_t                             alpha0,
      const matop_t                             op0,
      const Hpro::TLinearOperator< value_t > &  A0,
      const value_t                             alpha1,
      const matop_t                             op1,
      const Hpro::TLinearOperator< value_t > &  A1,
      const value_t                             alpha2,
      const matop_t                             op2,
      const Hpro::TLinearOperator< value_t > &  A2 )
{
    return std::make_unique< linop_sum< value_t > >( alpha0, op0, A0,
                                                     alpha1, op1, A1,
                                                     alpha2, op2, A2 );
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_SUM_HH

#ifndef __HLR_MATRIX_PRODUCT_HH
#define __HLR_MATRIX_PRODUCT_HH
//
// Project     : HLR
// Module      : matrix/product
// Description : provides operator for matrix products
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/matrix/TMatrixProduct.hh>

namespace hlr { namespace matrix {

#if 1

// local matrix type
DECLARE_TYPE( linop_product );

//
// represents Π_i α_i A_i of linear operators
//
template < typename T_value >
class linop_product : public Hpro::TLinearOperator< T_value >
{
public:
 
    using  value_t  = T_value;

    // some abbrv.
    using  linop_t  = Hpro::TLinearOperator< value_t >;
    using  vector_t = Hpro::TVector< value_t >;
    using  matrix_t = Hpro::TMatrix< value_t >;
   
private:

    struct factor_t {
        const linop_t *  linop;
        const value_t    scale;
        const matop_t    op;
    };
    
    // factors in matrix product
    std::vector< factor_t >  _factors;

public:
    //
    // ctors for different numbers of factors
    //

    linop_product ( const value_t    alpha0,
                    const matop_t    op0,
                    const linop_t &  A0,
                    const value_t    alpha1,
                    const matop_t    op1,
                    const linop_t &  A1 )
            : _factors{ { &A0, alpha0, op0 },
                        { &A1, alpha1, op1 } }
    {}
    

    linop_product ( const value_t    alpha0,
                    const matop_t    op0,
                    const linop_t &  A0,
                    const value_t    alpha1,
                    const matop_t    op1,
                    const linop_t &  A1,
                    const value_t    alpha2,
                    const matop_t    op2,
                    const linop_t &  A2 )
            : _factors{ { &A0, alpha0, op0 },
                        { &A1, alpha1, op1 },
                        { &A2, alpha2, op2 } }
    {}

    virtual ~linop_product () {}

    //
    // linear operator properties
    //

    // return true, of operator is self adjoint
    virtual bool  is_self_adjoint () const { return false; } // TODO
    
    //
    // linear operator mapping
    //
    
    //
    // compute y = A·x with A being the product operator
    //
    virtual void  apply       ( const vector_t *  x,
                                vector_t *        y,
                                const matop_t     op = apply_normal ) const;

    //
    // compute y = y + α·A·x with A being the product operator
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
        HLR_ASSERT( _factors.size() > 0 );
        
        const auto  s = _factors.back();
    
        if ( s.op == apply_normal ) return s.linop->domain_dim();
        else                        return s.linop->range_dim();
    }
    
    // return dimension of range
    virtual size_t  range_dim  () const
    {
        HLR_ASSERT( _factors.size() > 0 );
        
        const auto  s = _factors.front();
        
        if ( s.op == apply_normal ) return s.linop->range_dim();
        else                        return s.linop->domain_dim();
    }
    
    // return vector in domain space
    virtual auto    domain_vector  () const -> std::unique_ptr< vector_t >
    {
        HLR_ASSERT( _factors.size() > 0 );
        
        const auto  s = _factors.back();
        
        if ( s.op == apply_normal ) return s.linop->domain_vector();
        else                        return s.linop->range_vector();
    }

    // return vector in range space
    virtual auto    range_vector   () const -> std::unique_ptr< vector_t >
    {
        HLR_ASSERT( _factors.size() > 0 );
        
        const auto  s = _factors.front();
        
        if ( s.op == apply_normal ) return s.linop->range_vector();
        else                        return s.linop->domain_vector();
    }

    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( linop_product, Hpro::TLinearOperator< value_t > );
};

//
// multiplication functions
//
template < typename value_t >
void
linop_product< value_t >::apply ( const vector_t *  x,
                                  vector_t *        y,
                                  const matop_t     op ) const
{
    y->scale( value_t(0) );
        
    if ( _factors.size() == 1 )
    {
        _factors[0].linop->apply_add( _factors[0].scale, x, y, blas::apply_op( op, _factors[0].op ) );
    }// if
    else
    {
        const auto  n_fac = _factors.size();
        auto        tx    = std::unique_ptr< Hpro::TVector< value_t > >();
        auto        ty    = std::unique_ptr< Hpro::TVector< value_t > >();
        
        if ( op == apply_normal )
        {
            //
            // A₀(A₁(A₂x))
            //

            {
                auto  p = _factors[ n_fac-1 ];
                
                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = p.linop->range_vector();
                else                                              ty = p.linop->domain_vector();

                p.linop->apply_add( p.scale, x, ty.get(), blas::apply_op( op, p.op ) );
            }
            
            for ( int  i = n_fac-2; i > 0; --i )
            {
                auto  p = _factors[ i ];

                tx = std::move( ty );

                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = p.linop->range_vector();
                else                                              ty = p.linop->domain_vector();
            
                p.linop->apply_add( p.scale, tx.get(), ty.get(), blas::apply_op( op, p.op ) );
            }// for

            {
                auto  p = _factors[ 0 ];
                
                p.linop->apply_add( p.scale, ty.get(), y, blas::apply_op( op, p.op ) );
            }
        }// if
        else
        {
            //
            // (A₀A₁A₂)'x = A₂'A₁'A₀'x
            //

            {
                auto  p = _factors[ 0 ];
                
                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = p.linop->range_vector();
                else                                              ty = p.linop->domain_vector();

                p.linop->apply_add( p.scale, x, ty.get(), blas::apply_op( op, p.op ) );
            }
            
            for ( int  i = 1; i < n_fac-1; ++i )
            {
                auto  p = _factors[ i ];

                tx = std::move( ty );

                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = p.linop->range_vector();
                else                                               ty = p.linop->domain_vector();
            
                p.linop->apply_add( p.scale, tx.get(), ty.get(), blas::apply_op( op, p.op ) );
            }// for

            {
                auto  p = _factors[ n_fac-1 ];
                
                p.linop->apply_add( p.scale, ty.get(), y, blas::apply_op( op, p.op ) );
            }
        }// else
    }// else
}

template < typename value_t >
void
linop_product< value_t >::apply_add   ( const value_t     alpha,
                                        const vector_t *  x,
                                        vector_t *        y,
                                        const matop_t     op ) const
{
    if ( _factors.size() == 1 )
    {
        _factors[0].linop->apply_add( alpha * _factors[0].scale, x, y, blas::apply_op( op, _factors[0].op ) );
    }// if
    else
    {
        const auto  n_fac = _factors.size();
        auto        tx    = std::unique_ptr< Hpro::TVector< value_t > >();
        auto        ty    = std::unique_ptr< Hpro::TVector< value_t > >();

        if ( op == apply_normal )
        {
            //
            // A₀(A₁(A₂x))
            //

            {
                auto  p = _factors[ n_fac-1 ];
                
                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = p.linop->range_vector();
                else                                              ty = p.linop->domain_vector();

                p.linop->apply_add( p.scale, x, ty.get(), blas::apply_op( op, p.op ) );
            }
            
            for ( int  i = n_fac-2; i > 0; --i )
            {
                auto  p = _factors[ i ];

                tx = std::move( ty );

                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = p.linop->range_vector();
                else                                              ty = p.linop->domain_vector();
            
                p.linop->apply_add( p.scale, tx.get(), ty.get(), blas::apply_op( op, p.op ) );
            }// for

            {
                auto  p = _factors[ 0 ];
                
                p.linop->apply_add( alpha * p.scale, ty.get(), y, blas::apply_op( op, p.op ) );
            }
        }// if
        else
        {
            //
            // (A₀A₁A₂)'x = A₂'A₁'A₀'x
            //

            {
                auto  p = _factors[ 0 ];
                
                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = p.linop->range_vector();
                else                                              ty = p.linop->domain_vector();

                p.linop->apply_add( p.scale, x, ty.get(), blas::apply_op( op, p.op ) );
            }
            
            for ( int  i = 1; i < n_fac-1; ++i )
            {
                auto  p = _factors[ i ];

                tx = std::move( ty );

                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = p.linop->range_vector();
                else                                               ty = p.linop->domain_vector();
            
                p.linop->apply_add( p.scale, tx.get(), ty.get(), blas::apply_op( op, p.op ) );
            }// for

            {
                auto  p = _factors[ n_fac-1 ];
                
                p.linop->apply_add( alpha * p.scale, ty.get(), y, blas::apply_op( op, p.op ) );
            }
        }// else
    }// else
}

template < typename value_t >
void
linop_product< value_t >::apply_add   ( const value_t     /* alpha */,
                                        const matrix_t *  /* X */,
                                        matrix_t *        /* Y */,
                                        const matop_t     /* op */ ) const
{
    HLR_ERROR( "TODO" );
}
    
template < typename value_t >
void
linop_product< value_t >::apply_add   ( const value_t                    alpha,
                                        const blas::vector< value_t > &  x,
                                        blas::vector< value_t > &        y,
                                        const matop_t                    op ) const
{
    if ( _factors.size() == 1 )
    {
        _factors[0].linop->apply_add( alpha * _factors[0].scale, x, y, blas::apply_op( op, _factors[0].op ) );
    }// if
    else
    {
        const auto  n_fac = _factors.size();
        auto        tx    = blas::vector< value_t >();
        auto        ty    = blas::vector< value_t >();

        if ( op == apply_normal )
        {
            //
            // A₀(A₁(A₂x))
            //

            {
                auto  p = _factors[ n_fac-1 ];
                
                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = std::move( blas::vector< value_t >( p.linop->range_dim() ) );
                else                                              ty = std::move( blas::vector< value_t >( p.linop->domain_dim() ) );

                p.linop->apply_add( p.scale, x, ty, blas::apply_op( op, p.op ) );
            }
            
            for ( int  i = n_fac-2; i > 0; --i )
            {
                auto  p = _factors[ i ];

                tx = std::move( ty );

                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = std::move( blas::vector< value_t >( p.linop->range_dim() ) );
                else                                              ty = std::move( blas::vector< value_t >( p.linop->domain_dim() ) );
            
                p.linop->apply_add( p.scale, tx, ty, blas::apply_op( op, p.op ) );
            }// for

            {
                auto  p = _factors[ 0 ];
                
                p.linop->apply_add( alpha * p.scale, ty, y, blas::apply_op( op, p.op ) );
            }
        }// if
        else
        {
            //
            // (A₀A₁A₂)'x = A₂'A₁'A₀'x
            //

            {
                auto  p = _factors[ 0 ];
                
                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = std::move( blas::vector< value_t >( p.linop->range_dim() ) );
                else                                              ty = std::move( blas::vector< value_t >( p.linop->domain_dim() ) );

                p.linop->apply_add( p.scale, x, ty, blas::apply_op( op, p.op ) );
            }
            
            for ( int  i = 1; i < n_fac-1; ++i )
            {
                auto  p = _factors[ i ];

                tx = std::move( ty );

                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = std::move( blas::vector< value_t >( p.linop->range_dim() ) );
                else                                              ty = std::move( blas::vector< value_t >( p.linop->domain_dim() ) );
            
                p.linop->apply_add( p.scale, tx, ty, blas::apply_op( op, p.op ) );
            }// for

            {
                auto  p = _factors[ n_fac-1 ];
                
                p.linop->apply_add( alpha * p.scale, ty, y, blas::apply_op( op, p.op ) );
            }
        }// else
    }// else
}

template < typename value_t >
void
linop_product< value_t >::apply_add   ( const value_t                    alpha,
                                        const blas::matrix< value_t > &  X,
                                        blas::matrix< value_t > &        Y,
                                        const matop_t                    op ) const
{
    HLR_ERROR( "TODO" );
}

template < typename value_t >
template < typename arithmetic_t >
void
linop_product< value_t >::apply_add   ( arithmetic_t &&                  arithmetic,
                                        const value_t                    alpha,
                                        const blas::vector< value_t > &  x,
                                        blas::vector< value_t > &        y,
                                        const matop_t                    op ) const
{
    if ( _factors.size() == 1 )
    {
        _factors[0].linop->apply_add( alpha * _factors[0].scale, x, y, blas::apply_op( op, _factors[0].op ) );
    }// if
    else
    {
        const auto  n_fac = _factors.size();
        auto        tx    = blas::vector< value_t >();
        auto        ty    = blas::vector< value_t >();

        if ( op == apply_normal )
        {
            //
            // A₀(A₁(A₂x))
            //

            {
                auto  p = _factors[ n_fac-1 ];
                
                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = blas::vector< value_t >( p.linop->range_dim() );
                else                                              ty = blas::vector< value_t >( p.linop->domain_dim() );

                arithmetic.prod( p.scale, blas::apply_op( op, p.op ), * p.linop, x, ty );
            }
            
            for ( int  i = n_fac-2; i > 0; --i )
            {
                auto  p = _factors[ i ];

                tx = std::move( ty );

                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = blas::vector< value_t >( p.linop->range_dim() );
                else                                              ty = blas::vector< value_t >( p.linop->domain_dim() );
            
                arithmetic.prod( p.scale, blas::apply_op( op, p.op ), * p.linop, tx, ty );
            }// for

            {
                auto  p = _factors[ 0 ];
                
                arithmetic.prod( alpha * p.scale, blas::apply_op( op, p.op ), * p.linop, ty, y );
            }
        }// if
        else
        {
            //
            // (A₀A₁A₂)'x = A₂'A₁'A₀'x
            //

            {
                auto  p = _factors[ 0 ];
                
                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = blas::vector< value_t >( p.linop->range_dim() );
                else                                              ty = blas::vector< value_t >( p.linop->domain_dim() );

                arithmetic.prod( p.scale, blas::apply_op( op, p.op ), * p.linop, x, ty );
            }
            
            for ( int  i = 1; i < n_fac-1; ++i )
            {
                auto  p = _factors[ i ];

                tx = std::move( ty );

                if ( blas::apply_op( p.op, op ) == apply_normal ) ty = blas::vector< value_t >( p.linop->range_dim() );
                else                                              ty = blas::vector< value_t >( p.linop->domain_dim() );
            
                arithmetic.prod( p.scale, blas::apply_op( op, p.op ), * p.linop, tx, ty );
            }// for

            {
                auto  p = _factors[ n_fac-1 ];
                
                arithmetic.prod( alpha * p.scale, blas::apply_op( op, p.op ), * p.linop, ty, y );
            }
        }// else
    }// else
}

//
// functions to create product objects
//
template < typename value_t >
std::unique_ptr< linop_product< value_t > >
product ( const Hpro::TLinearOperator< value_t > &  A0,
          const Hpro::TLinearOperator< value_t > &  A1 )
{
    return std::make_unique< linop_product< value_t > >( value_t(1), apply_normal, A0,
                                                         value_t(1), apply_normal, A1 );
}
                 
template < typename alpha0_t,
           typename alpha1_t,
           typename value_t >
std::unique_ptr< linop_product< value_t > >
product ( const alpha0_t                            alpha0,
          const Hpro::TLinearOperator< value_t > &  A0,
          const alpha1_t                            alpha1,
          const Hpro::TLinearOperator< value_t > &  A1 )
{
    return std::make_unique< linop_product< value_t > >( value_t(alpha0), apply_normal, A0,
                                                         value_t(alpha1), apply_normal, A1 );
}
                 
template < typename value_t >
std::unique_ptr< linop_product< value_t > >
product ( const value_t                             alpha0,
          const matop_t                             op0,
          const Hpro::TLinearOperator< value_t > &  A0,
          const value_t                             alpha1,
          const matop_t                             op1,
          const Hpro::TLinearOperator< value_t > &  A1 )
{
    return std::make_unique< linop_product< value_t > >( alpha0, op0, A0,
                                                         alpha1, op1, A1 );
}
    

template < typename value_t >
std::unique_ptr< linop_product< value_t > >
product ( const Hpro::TLinearOperator< value_t > &  A0,
          const Hpro::TLinearOperator< value_t > &  A1,
          const Hpro::TLinearOperator< value_t > &  A2 )
{
    return std::make_unique< linop_product< value_t > >( value_t(1), apply_normal, A0,
                                                         value_t(1), apply_normal, A1,
                                                         value_t(1), apply_normal, A2 );
}
    
template < typename value_t >
std::unique_ptr< linop_product< value_t > >
product ( const value_t                             alpha0,
          const Hpro::TLinearOperator< value_t > &  A0,
          const value_t                             alpha1,
          const Hpro::TLinearOperator< value_t > &  A1,
          const value_t                             alpha2,
          const Hpro::TLinearOperator< value_t > &  A2 )
{
    return std::make_unique< linop_product< value_t > >( alpha0, apply_normal, A0,
                                                         alpha1, apply_normal, A1,
                                                         alpha2, apply_normal, A2 );
}
                 
template < typename value_t >
std::unique_ptr< linop_product< value_t > >
product ( const value_t                             alpha0,
          const matop_t                             op0,
          const Hpro::TLinearOperator< value_t > &  A0,
          const value_t                             alpha1,
          const matop_t                             op1,
          const Hpro::TLinearOperator< value_t > &  A1,
          const value_t                             alpha2,
          const matop_t                             op2,
          const Hpro::TLinearOperator< value_t > &  A2 )
{
    return std::make_unique< linop_product< value_t > >( alpha0, op0, A0,
                                                         alpha1, op1, A1,
                                                         alpha2, op2, A2 );
}

#else

//
// functions to return matrix product objects
//

template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const Hpro::TLinearOperator< value_t > *  A0,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( value_t(1), A0, is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const value_t                             alpha0,
          const Hpro::TLinearOperator< value_t > *  A0,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( alpha0, A0, is_owner );
}

template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const Hpro::TLinearOperator< value_t > *  A0,
          const Hpro::TLinearOperator< value_t > *  A1,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( value_t(1), A0, value_t(1), A1, is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const value_t                             alpha0,
          const Hpro::TLinearOperator< value_t > *  A0,
          const value_t                             alpha1,
          const Hpro::TLinearOperator< value_t > *  A1,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( alpha0, A0, alpha1, A1, is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const value_t                             alpha0,
          const matop_t                             op0,
          const Hpro::TLinearOperator< value_t > *  A0,
          const value_t                             alpha1,
          const matop_t                             op1,
          const Hpro::TLinearOperator< value_t > *  A1,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( alpha0, op0, A0, alpha1, op1, A1, is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const Hpro::TLinearOperator< value_t > *  A0,
          const Hpro::TLinearOperator< value_t > *  A1,
          const Hpro::TLinearOperator< value_t > *  A2,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( value_t(1), A0, value_t(1), A1, value_t(1), A2, is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const value_t                             alpha0,
          const Hpro::TLinearOperator< value_t > *  A0,
          const value_t                             alpha1,
          const Hpro::TLinearOperator< value_t > *  A1,
          const value_t                             alpha2,
          const Hpro::TLinearOperator< value_t > *  A2,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( alpha0, A0, alpha1, A1, alpha2, A2, is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const value_t                             alpha0,
          const matop_t                             op0,
          const Hpro::TLinearOperator< value_t > *  A0,
          const value_t                             alpha1,
          const matop_t                             op1,
          const Hpro::TLinearOperator< value_t > *  A1,
          const value_t                             alpha2,
          const matop_t                             op2,
          const Hpro::TLinearOperator< value_t > *  A2,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( alpha0, op0, A0, alpha1, op1, A1, alpha2, op2, A2, is_owner );
}


template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const Hpro::TLinearOperator< value_t > &  A0,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( value_t(1), & A0, is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const value_t                             alpha0,
          const Hpro::TLinearOperator< value_t > &  A0,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( alpha0, & A0, is_owner );
}

template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const Hpro::TLinearOperator< value_t > &  A0,
          const Hpro::TLinearOperator< value_t > &  A1,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( value_t(1), & A0,
                                                                value_t(1), & A1,
                                                                is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const value_t                             alpha0,
          const Hpro::TLinearOperator< value_t > &  A0,
          const value_t                             alpha1,
          const Hpro::TLinearOperator< value_t > &  A1,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( alpha0, & A0,
                                                                alpha1, & A1,
                                                                is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const value_t                             alpha0,
          const matop_t                             op0,
          const Hpro::TLinearOperator< value_t > &  A0,
          const value_t                             alpha1,
          const matop_t                             op1,
          const Hpro::TLinearOperator< value_t > &  A1,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( alpha0, op0, & A0,
                                                                alpha1, op1, & A1,
                                                                is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const Hpro::TLinearOperator< value_t > &  A0,
          const Hpro::TLinearOperator< value_t > &  A1,
          const Hpro::TLinearOperator< value_t > &  A2,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( value_t(1), & A0,
                                                                value_t(1), & A1,
                                                                value_t(1), & A2,
                                                                is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const value_t                             alpha0,
          const Hpro::TLinearOperator< value_t > &  A0,
          const value_t                             alpha1,
          const Hpro::TLinearOperator< value_t > &  A1,
          const value_t                             alpha2,
          const Hpro::TLinearOperator< value_t > &  A2,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( alpha0, & A0,
                                                                alpha1, & A1,
                                                                alpha2, & A2,
                                                                is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixProduct< value_t > >
product ( const value_t                             alpha0,
          const matop_t                             op0,
          const Hpro::TLinearOperator< value_t > &  A0,
          const value_t                             alpha1,
          const matop_t                             op1,
          const Hpro::TLinearOperator< value_t > &  A1,
          const value_t                             alpha2,
          const matop_t                             op2,
          const Hpro::TLinearOperator< value_t > &  A2,
          const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixProduct< value_t > >( alpha0, op0, & A0,
                                                                alpha1, op1, & A1,
                                                                alpha2, op2, & A2,
                                                                is_owner );
}

#endif

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_PRODUCT_HH

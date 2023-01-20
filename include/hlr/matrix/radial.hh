#ifndef __HLR_MATRIX_RADIAL_HH
#define __HLR_MATRIX_RADIAL_HH
//
// Project     : HLR
// Module      : covariance
// Description : various covariance coefficient functions for matrix construction
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hpro/config.h>

#if USE_GSL == 1
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#else
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/bessel.hpp>
#endif

#include <hpro/matrix/TCoeffFn.hh>
#include <hpro/base/TPoint.hh>

#include <hlr/utils/math.hh>

namespace hlr { namespace matrix {

//
// generic radial symmetric coefficient function evaluating
// given kernel k( d = |x-y| )
//
template < typename T_kernel,
           typename T_coordinate >
class radial_function : public Hpro::TCoeffFn< typename T_kernel::value_t >
{
public:
    using  kernel_t     = T_kernel;
    using  value_t      = typename kernel_t::value_t;
    using  coordinate_t = T_coordinate;
    
private:
    // arrays holding the coordinates for x and y
    const coordinate_t &  _x_vertices;
    const coordinate_t &  _y_vertices;

    // kernel function to evaluate
    const kernel_t        _kernel;
    
public:
    //!
    //! constructor
    //!
    radial_function ( const kernel_t &      kernel,
                      const coordinate_t &  x_vertices,
                      const coordinate_t &  y_vertices )
            : _x_vertices( x_vertices )
            , _y_vertices( y_vertices )
            , _kernel( kernel )
    {}

    //!
    //! coefficient evaluation
    //!
    virtual void eval  ( const std::vector< Hpro::idx_t > &  rowidxs,
                         const std::vector< Hpro::idx_t > &  colidxs,
                         value_t *                     matrix ) const
    {
        const size_t  n = rowidxs.size();
        const size_t  m = colidxs.size();

        for ( size_t  j = 0; j < m; ++j )
        {
            const auto    idx1 = colidxs[ j ];
            const auto &  y    = _y_vertices[ idx1 ];
            
            for ( size_t  i = 0; i < n; ++i )
            {
                const auto    idx0 = rowidxs[ i ];
                const auto &  x    = _x_vertices[ idx0 ];
                const auto    dist = Hpro::norm2( x - y );

                matrix[ j*n + i ] = _kernel( dist );
            }// for
        }// for
    }
    using Hpro::TCoeffFn< value_t >::eval;

    //!
    //! return format of matrix: symmetric as kernel is symmetric
    //!
    virtual Hpro::matform_t  matrix_format  () const { return Hpro::symmetric; }
};

////////////////////////////////////////////////////////////
//
// various radial symmetric kernels
//
////////////////////////////////////////////////////////////

//
//  exp(-εr)
//
template < typename T_value = double >
struct exponential_function
{
    using  value_t = T_value;

    const value_t  epsilon;

    exponential_function ( const value_t  eps )
            : epsilon( eps )
    {}

    value_t  operator () ( const value_t  r ) const
    {
        return std::exp( - epsilon * r );
    }
};

//
// Gaussian kernel
//
//  exp( - εr² )
//
template < typename T_value = double >
struct gaussian_function
{
    using  value_t = T_value;

    const value_t  epsilon;

    gaussian_function ( const value_t  eps )
            : epsilon( eps )
    {}

    value_t  operator () ( const value_t  r ) const
    {
        return std::exp( - epsilon * math::square( r ) );
    }
};

//
// multiquadric function
//  ___________
// √ 1 + (εr)²
//
template < typename T_value = double >
struct multiquadric_function
{
    using  value_t = T_value;

    value_t  epsilon;

    multiquadric_function ( const value_t  eps )
            : epsilon( eps )
    {}

    value_t  operator () ( const value_t  r ) const
    {
        return std::sqrt( value_t(1) + math::square( epsilon * r ) );
    }
};

//
// inverse multiquadric function
//      ___________
// 1 / √ 1 + (εr)²
//
template < typename T_value = double >
struct inverse_multiquadric_function
{
    using  value_t = T_value;

    value_t  epsilon;

    inverse_multiquadric_function ( const value_t  eps )
            : epsilon( eps )
    {}

    value_t  operator () ( const value_t  r ) const
    {
        return value_t(1) / std::sqrt( value_t(1) + math::square( epsilon * r ) );
    }
};

//
// thin plate spline
// 
// (εr)² log(εr)
//
template < typename T_value = double >
struct thin_plate_spline_function
{
    using  value_t = T_value;

    value_t  epsilon;

    thin_plate_spline_function ( const value_t  eps )
            : epsilon( eps )
    {}

    value_t  operator () ( const value_t  r ) const
    {
        const auto  er = epsilon * r;
        
        return math::square( er ) * math::log( er );
    }
};

//
// Rational Quadratic Function
//
//           -α
//  ⎛     r²⎞
//  ⎜1 + ───⎟
//  ⎝    αl²⎠
//
template < typename T_value = double >
struct rational_quadratic_function
{
    using  value_t = T_value;

    // α
    value_t  alpha;
    
    // 1 / (αl²)
    value_t  inv_alpha_sqlength;

    rational_quadratic_function ( const value_t  a,
                                  const value_t  l )
            : alpha( a )
            , inv_alpha_sqlength( value_t(1) / ( a * math::square(l) ) )
    {}

    value_t  operator () ( const value_t  r ) const
    {
        return std::pow( value_t(1) + inv_alpha_sqlength * math::square( r ), - alpha );
    }
};

//
// Matérn Covariance
//
//      1-ν   __   ν    __  
//     2    ⎛√2ν  ⎞   ⎛√2ν  ⎞
//  σ² ──── ⎜─── d⎟ K ⎜─── d⎟
//     Γ(ν) ⎝ l   ⎠  ν⎝ l   ⎠
//
template < typename T_value = double >
struct matern_covariance_function
{
    using  value_t = T_value;

    const value_t  sigmasq;
    const value_t  nu;
    const value_t  gamma_nu;
    const value_t  sqrnu_over_l;

    matern_covariance_function ( const value_t  asigma,
                                 const value_t  anu,
                                 const value_t  al )
            : sigmasq( math::square( asigma ) )
            , nu( anu )
            , gamma_nu( eval_gamma( asigma, anu ) )
            , sqrnu_over_l( math::sqrt( value_t(2) * anu ) / al )
    {}

    value_t  operator () ( const value_t  r ) const
    {
        if ( r <= std::numeric_limits< value_t >::epsilon() )
            return sigmasq;
                
        const auto  nu_l_r   = r * sqrnu_over_l;
        #if USE_GSL == 1
        const auto  bessel_r = gsl_sf_bessel_Knu( nu, nu_l_r );
        #else
        const auto  bessel_r = boost::math::cyl_bessel_k( nu, nu_l_r );
        #endif
        
        return gamma_nu * bessel_r * std::pow( nu_l_r, nu );
    }

    value_t  eval_gamma ( const value_t  asigma,
                          const value_t  anu ) const
    {
        #if USE_GSL == 1
        const auto  gamma_nu = gsl_sf_gamma( anu );
        #else
        const auto  gamma_nu = boost::math::tgamma( anu );
        #endif
        
        return math::sqrt( asigma ) * std::pow( value_t(2), value_t(1) - nu ) / gamma_nu;
    }
};

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_RADIAL_HH

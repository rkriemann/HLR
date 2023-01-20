//
// Project     : HLib
// File        : LogKernel.cc
// Description : functions for logarithmic kernel function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cmath>
#include <vector>

#include "hlr/apps/log_kernel.hh"

namespace hlr { namespace apps {

namespace
{

//
// coefficient function for log|x-y| in [0,1]
//
class log_coeff_func : public Hpro::TCoeffFn< double >
{
private:
    // stepwidth
    const double  _h;

public:
    // constructor
    log_coeff_func ( const double  h )
            : _h(h)
    {}

    //
    // coefficient evaluation
    //
    virtual void eval  ( const std::vector< Hpro::idx_t > &  rowidxs,
                         const std::vector< Hpro::idx_t > &  colidxs,
                         double *                            matrix ) const
    {
        const size_t  n = rowidxs.size();
        const size_t  m = colidxs.size();

        for ( size_t  j = 0; j < m; ++j )
        {
            const auto  idx1 = colidxs[ j ];
            
            for ( size_t  i = 0; i < n; ++i )
            {
                const auto  idx0 = rowidxs[ i ];
                double      value;

                if ( idx0 == idx1 ) 
                    value = -1.5*_h*_h + _h*_h*std::log(_h);
                else
                {
                    const double  dist = _h * ( std::abs( double( idx0 - idx1 ) ) - 1.0 );
                    const double  t1   = dist+1.0*_h;
                    const double  t2   = dist+2.0*_h;
            
                    value = ( - 1.5*_h*_h + 0.5*t2*t2*std::log(t2) - t1*t1*std::log(t1) );
            
                    if ( std::abs(dist) > 1e-8 )
                        value += 0.5*dist*dist*std::log(dist);
                }
        
                matrix[ j*n + i ] = double(-value);
            }// for
        }// for
    }
    using Hpro::TCoeffFn< double >::eval;

    //
    // return format of matrix, e.g. symmetric or hermitian
    //
    virtual Hpro::matform_t  matrix_format  () const { return Hpro::symmetric; }
    
};

}// namespace anonymous

//
// ctor
//
log_kernel::log_kernel ( const size_t  n )
        : _n( n )
        , _h( 1.0 / value_t(n) )
{}

//
// set up coordinates
//
std::unique_ptr< Hpro::TCoordinate >
log_kernel::coordinates () const
{
    std::vector< double * >  vertices( _n, nullptr );
    std::vector< double * >  bbmin( _n, nullptr );
    std::vector< double * >  bbmax( _n, nullptr );

    for ( size_t i = 0; i < _n; i++ )
    {
        vertices[i]    = new double;
        vertices[i][0] = _h * double(i) + ( _h / 2.0 ); // center of [i/h,(i+1)/h]

        // set bounding box (support) to [i/h,(i+1)/h]
        bbmin[i]       = new double;
        bbmin[i][0]    = _h * double(i);
        bbmax[i]       = new double;
        bbmax[i][0]    = _h * double(i+1);
    }// for

    return std::make_unique< Hpro::TCoordinate >( vertices, 1, bbmin, bbmax, Hpro::copy_coord_data );
}

//
// return coefficient function to evaluate matrix entries
//
std::unique_ptr< Hpro::TCoeffFn< log_kernel::value_t > >
log_kernel::coeff_func () const
{
    return std::make_unique< log_coeff_func >( _h );
}
    
}}// namespace hlr::apps

#ifndef __HLR_MATRIX_NESTED_CLUSTER_BASIS_HH
#define __HLR_MATRIX_NESTED_CLUSTER_BASIS_HH
//
// Project     : HLR
// Module      : matrix/nested_cluster_basis
// Description : nested cluster basis
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/config.h>

// HACK to detect presence of HLIBpro and H² support
#if defined(HPRO_USE_LIC_CHECK)
#  define HLR_HAS_H2
#endif

#include <hpro/cluster/TCluster.hh>

#if defined(HLR_HAS_H2)
#include <hpro/cluster/TClusterBasis.hh>
#endif

#include <hlr/matrix/shared_cluster_basis.hh>

namespace hlr
{ 

using indexset     = Hpro::TIndexSet;
using cluster_tree = Hpro::TCluster;
using accuracy     = Hpro::TTruncAcc;

using Hpro::idx_t;

// local matrix type
DECLARE_TYPE( nested_cluster_basis );

namespace matrix
{

#define HLR_USE_APLR_NESTED_CB  1

//
// represents cluster basis for single cluster with
// additional hierarchy
//
template < typename T_value >
class nested_cluster_basis : public compress::compressible
{
public:

    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    using  self_t  = nested_cluster_basis< value_t >;
    
private:
    // associated indexset
    const indexset                          _is;

    // cluster basis of sub clusters
    std::vector< self_t * >                 _sons;
    
    // basis
    hlr::blas::matrix< value_t >            _V;

    // transfer matrix per son
    std::vector< blas::matrix< value_t > >  _E;

    // rank (either from V or from E)
    uint                                    _rank;
    
    // mutex for synchronised changes
    std::mutex                              _mtx;

    #if HLR_HAS_COMPRESSION == 1
    #if HLR_USE_APLR_NESTED_CB == 1
    // basis and corresponding singular values
    compress::aplr::zarray                  _zV;
    blas::vector< real_t >                  _sv;
    #else
    // just basis with standard compression
    compress::zarray                        _zV;
    #endif
    // coefficients always with std. compression
    std::vector< compress::zarray >         _zE;
    #endif
    
public:
    
    // construct cluster basis corresponding to cluster <cl>
    nested_cluster_basis ( const indexset &  ais )
            : _is( ais )
            , _rank( 0 )
    {}

    // construct cluster basis corresponding to cluster <ais>
    // with leaf basis defined by <V>
    nested_cluster_basis ( const indexset &                       ais,
                           const hlr::blas::matrix< value_t > &   V )
            : _is( ais )
            , _V( V )
            , _rank( _V.ncols() )
    {}

    nested_cluster_basis ( const indexset &                       ais,
                           hlr::blas::matrix< value_t > &&        V )
            : _is( ais )
            , _V( std::move( V ) )
            , _rank( _V.ncols() )
    {}

    // construct cluster basis corresponding to cluster <ais>
    // with sons <asons> and transfer matrices <aE>
    nested_cluster_basis ( const indexset &                                ais,
                           const std::vector< self_t * > &                 asons,
                           const std::vector< blas::matrix< value_t > > &  aE )
            : _is( ais )
            , _rank( 0 )
    {
        HLR_ASSERT( aE.size() == asons.size() );
        
        _sons.resize( asons.size() );
        for ( size_t  i = 0; i < asons.size(); ++i )
            _sons[i] = asons[i];
        
        _E.resize( aE.size() );
        for ( size_t  i = 0; i < aE.size(); ++i )
            _E[i] = std::move( blas::copy( aE[i] ) );

        // ensure that all transfer matrices have same number of columns
        if ( _E.size() > 0 )
        {
            _rank = _E[0].ncols();
            
            for ( size_t  i = 1; i < _E.size(); ++i )
                HLR_ASSERT( _E[i].ncols() == _rank );
        }// if
    }

    // dtor: delete sons
    ~nested_cluster_basis ()
    {
        for ( auto  cb : _sons )
            delete cb;
    }

    //
    // access sons
    //

    // return number of sons
    uint                          nsons     () const                { return _sons.size(); }

    // access <i>'th son
    nested_cluster_basis *        son       ( const uint  i )       { return _sons[i]; }
    const nested_cluster_basis *  son       ( const uint  i ) const { return _sons[i]; }

    void                          set_son   ( const uint              i,
                                              nested_cluster_basis *  cb ) { _sons[i] = cb; }

    // set number of sons
    void                          set_nsons ( const uint  n )
    {
        _sons.resize( n );
        _E.resize( n );
    }

    //
    // access basis
    //

    // underlying indexset
    const indexset &                      is     () const { return _is; }
    
    // return rank of cluster basis
    uint                                  rank   () const { return _rank; }
    
    // return local basis
    hlr::blas::matrix< value_t >          basis  () const
    {
        HLR_ASSERT( nsons() == 0 );
        
        #if HLR_HAS_COMPRESSION == 1
        
        if ( is_compressed() )
        {
            auto  V = blas::matrix< value_t >( _is.size(), rank() );
    
            #if HLR_USE_APLR_NESTED_CB == 1
            compress::aplr::decompress_lr< value_t >( _zV, V );
            #else
            compress::decompress< value_t >( _zV, V );
            #endif
            
            return V;
        }// if

        #endif

        return _V;
    }
    
    // set local basis
    void
    set_basis  ( const blas::matrix< value_t > &  aV )
    {
        HLR_ASSERT( nsons() == 0 );
        
        if (( _V.nrows() == aV.nrows() ) && ( _V.ncols() == aV.ncols() )) blas::copy( aV, _V );
        else                                                              _V = std::move( blas::copy( aV ) );

        _rank = _V.ncols();
    }
    
    void
    set_basis  ( blas::matrix< value_t > &&  aV )
    {
        HLR_ASSERT( nsons() == 0 );
        
        if (( _V.nrows() == aV.nrows() ) && ( _V.ncols() == aV.ncols() )) blas::copy( aV, _V );
        else                                                              _V = std::move( aV );

        _rank = _V.ncols();
    }

    // set basis together with singular values
    void
    set_basis  ( const blas::matrix< value_t > &  aV,
                 const blas::vector< real_t > &   asv )
    {
        HLR_ASSERT( nsons() == 0 );
        
        if (( _V.nrows() == aV.nrows() ) && ( _V.ncols() == aV.ncols() )) blas::copy( aV, _V );
        else                                                              _V = std::move( blas::copy( aV ) );

        _rank = _V.ncols();

        #if HLR_HAS_COMPRESSION == 1 && HLR_USE_APLR_NESTED_CB == 1
        if ( _sv.length() == asv.length() ) blas::copy( asv, _sv );
        else                                _sv = std::move( blas::copy( asv ) );
        #endif
    }
    
    void
    set_basis  ( blas::matrix< value_t > &&  aV,
                 blas::vector< real_t > &&   asv )
    {
        HLR_ASSERT( nsons() == 0 );
        
        if (( _V.nrows() == aV.nrows() ) && ( _V.ncols() == aV.ncols() )) blas::copy( aV, _V );
        else                                                              _V = std::move( aV );

        _rank = _V.ncols();

        #if HLR_HAS_COMPRESSION == 1 && HLR_USE_APLR_NESTED_CB == 1
        if ( _sv.length() == asv.length() ) blas::copy( asv, _sv );
        else                                _sv = std::move( asv );
        #endif
    }

    //
    // access transfer matrices to son cluster bases
    //
    blas::matrix< value_t >  transfer_mat ( const uint  i ) const
    {
        HLR_ASSERT( nsons() > 0 );

        #if HLR_HAS_COMPRESSION == 1

        if ( is_compressed() )
        {
            auto  Ei = blas::matrix< value_t >( _E[i].nrows(), rank() );
            
            compress::decompress< value_t >( _zE[i], Ei );

            return Ei;
        }// if
        
        #endif
        
        return _E[i];
    }

    //
    // transfer vector s to basis of i'th son
    //
    blas::vector< value_t >  transfer_to_son ( const uint                       i,
                                               const blas::vector< value_t > &  s ) const
    {
        HLR_ASSERT( nsons() > 0 );
        HLR_ASSERT( i < nsons() );

        // check if there is no transfer
        if ( _E[i].nrows() == 0 )
            return blas::vector< value_t >();
        
        #if HLR_HAS_COMPRESSION == 1

        if ( is_compressed() )
        {
            auto  Ei = blas::matrix< value_t >( _E[i].nrows(), rank() );
            
            compress::decompress< value_t >( _zE[i], Ei );

            return blas::mulvec( Ei, s );
        }// if
        
        #endif
        
        return blas::mulvec( _E[i], s );
    }

    //
    // set transfer matrices to son cluster bases
    //
    void
    set_transfer ( const std::vector< blas::matrix< value_t > > &  aE )
    {
        HLR_ASSERT( nsons() > 0 );
        HLR_ASSERT( aE.size() == nsons() );

        _E.resize( aE.size() );

        for ( uint  i = 0; i < aE.size(); ++i )
            _E[i] = std::move( blas::copy( aE[i] ) );

        if ( _E.size() > 0 )
        {
            _rank = _E[0].ncols();
            
            for ( size_t  i = 1; i < _E.size(); ++i )
                HLR_ASSERT( _E[i].ncols() == _rank );
        }// if
    }
    
    void
    set_transfer ( std::vector< blas::matrix< value_t > > &&  aE )
    {
        HLR_ASSERT( nsons() > 0 );
        HLR_ASSERT( aE.size() == nsons() );

        _E = std::move( aE );

        if ( _E.size() > 0 )
        {
            _rank = _E[0].ncols();
            
            for ( size_t  i = 1; i < _E.size(); ++i )
                HLR_ASSERT( _E[i].ncols() == _rank );
        }// if
    }
    
    ///////////////////////////////////////////////////////
    //
    // basis transformation
    //

    //
    // forward transformation, e.g. compute coefficients in local basis
    // - return V^H · arg
    //
    
    hlr::blas::vector< value_t >
    transform_forward  ( const hlr::blas::vector< value_t > &  v ) const
    {
        #if HLR_COMPRESSOR == HLR_COMPRESSOR_AFLP
        if ( is_compressed() )
        {
            const auto  k = this->rank();
            
            if ( nsons() == 0 )
            {
                auto  t = blas::vector< value_t >( k );

                #if HLR_USE_APLR_NESTED_CB == 1
                compress::aflp::mulvec_lr( _is.size(), k, apply_adjoint, value_t(1), _zV, v.data(), t.data() );
                #else
                compress::aflp::mulvec( _is.size(), k, apply_adjoint, value_t(1), _zV, v.data(), t.data() );
                #endif

                return t;
            }// if
            else
            {
                //
                // compute V'·v = (∑_i V_i E_i)' v = ∑_i E_i' ( V_i' v )
                //

                blas::vector< value_t >  s( k );
        
                for ( uint  i = 0; i < nsons(); ++i )
                {
                    auto  son_i = son( i );
                    auto  v_i   = blas::vector< value_t >( v, son_i->is() - is().first() );
                    auto  s_i   = son_i->transform_forward( v_i );
                
                    compress::aflp::mulvec( _E[i].nrows(), k, apply_adjoint, value_t(1), _zE[i], s_i.data(), s.data() );
                }// for

                return std::move( s );
            }// else
        }// if
        else
        #endif
        {
            if ( nsons() == 0 )
            {
                auto  V = basis();
        
                return blas::mulvec( hlr::blas::adjoint( V ), v );
            }// if
            else
            {
                //
                // compute V'·v = (∑_i V_i E_i)' v = ∑_i E_i' ( V_i' v )
                //

                blas::vector< value_t >  s( rank() );
        
                for ( uint  i = 0; i < nsons(); ++i )
                {
                    auto  son_i = son( i );
                    auto  v_i   = blas::vector< value_t >( v, son_i->is() - is().first() );
                    auto  s_i   = son_i->transform_forward( v_i );
                
                    blas::mulvec( value_t(1), blas::adjoint( transfer_mat(i) ), s_i, value_t(1), s );
                }// for

                return std::move( s );
            }// else
        }// else
    }
    
    hlr::blas::matrix< value_t >
    transform_forward  ( const hlr::blas::matrix< value_t > &  M ) const
    {
        if ( nsons() == 0 )
        {
            auto  V = basis();
        
            return blas::prod( hlr::blas::adjoint( V ), M );
        }// if
        else
        {
            //
            // compute V'·M = (∑_i V_i E_i)' M = ∑_i E_i' ( V_i' M )
            //

            blas::matrix< value_t >  S( rank(), M.ncols() );
        
            for ( uint  i = 0; i < nsons(); ++i )
            {
                auto  son_i = son( i );
                auto  M_i   = blas::matrix< value_t >( M, son_i->is() - is().first(), blas::range::all );
                auto  S_i   = son_i->transform_forward( M_i );
                
                blas::prod( value_t(1), blas::adjoint( transfer_mat(i) ), S_i, value_t(1), S );
            }// for

            return std::move( S );
        }// else
    }
    
    //
    // backward transformation, e.g. compute vector/matrix in standard basis
    // - return V · arg
    //
    
    hlr::blas::vector< value_t >
    transform_backward  ( const hlr::blas::vector< value_t > &  s ) const
    {
        #if HLR_COMPRESSOR == HLR_COMPRESSOR_AFLP
        if ( is_compressed() )
        {
            if ( nsons() == 0 )
            {
                const auto  n = _is.size();
                auto        t = blas::vector< value_t >( n );

                #if HLR_USE_APLR_NESTED_CB == 1
                compress::aflp::mulvec_lr( n, this->rank(), apply_normal, value_t(1), _zV, s.data(), t.data() );
                #else
                compress::aflp::mulvec( n, this->rank(), apply_normal, value_t(1), _zV, s.data(), t.data() );
                #endif

                return t;
            }// if
            else
            {
                //
                // M ≔ V·s = ∑_i V_i·E_i·s = ∑_i transform_backward( son_i, E_i·s )
                //

                const auto  k = rank();
                auto        v = blas::vector< value_t >( is().size() );
            
                for ( uint  i = 0; i < nsons(); ++i )
                {
                    auto  son_i = son( i );
                    auto  s_i   = blas::vector< value_t >( _E[i].nrows() );
                    
                    compress::aflp::mulvec( _E[i].nrows(), k, apply_normal, value_t(1), _zE[i], s.data(), s_i.data() );

                    auto  t_i   = son_i->transform_backward( s_i );
                    auto  v_i   = blas::vector< value_t >( v, son_i->is() - is().first() );

                    blas::copy( t_i, v_i );
                }// for

                return std::move( v );
            }// else
        }// if
        else
        #endif
        {
            if ( nsons() == 0 )
            {
                auto  V = basis();
        
                return hlr::blas::mulvec( V, s );
            }// if
            else
            {
                //
                // M ≔ V·s = ∑_i V_i·E_i·s = ∑_i transform_backward( son_i, E_i·s )
                //

                auto  v = blas::vector< value_t >( is().size() );
            
                for ( uint  i = 0; i < nsons(); ++i )
                {
                    auto  son_i = son( i );
                    auto  s_i   = blas::mulvec( transfer_mat(i), s );
                    auto  t_i   = son_i->transform_backward( s_i );
                    auto  v_i   = blas::vector< value_t >( v, son_i->is() - is().first() );

                    blas::copy( t_i, v_i );
                }// for

                return std::move( v );
            }// else
        }// else
    }
    
    hlr::blas::matrix< value_t >
    transform_backward  ( const hlr::blas::matrix< value_t > &  S ) const
    {
        if ( nsons() == 0 )
        {
            auto  V = basis();
        
            return blas::prod( V, S );
        }// if
        else
        {
            //
            // M ≔ V·S = ∑_i V_i·E_i·S = ∑_i transform_backward( son_i, E_i·S )
            //

            blas::matrix< value_t >  M( is().size(), S.ncols() );
        
            for ( uint  i = 0; i < nsons(); ++i )
            {
                auto  son_i = son( i );
                auto  S_i   = blas::prod( transfer_mat(i), S );
                auto  M_i   = son_i->transform_backward( S_i );
                auto  M_sub = blas::matrix< value_t >( M, son_i->is() - is().first(), blas::range::all );
                
                blas::copy( M_i, M_sub );
            }// for

            return std::move( M );
        }// else
    }

    ///////////////////////////////////////////////////////
    //
    // misc.
    //

    // return copy of cluster basis
    std::unique_ptr< nested_cluster_basis >
    copy () const
    {
        auto  cb = std::make_unique< nested_cluster_basis >( _is );

        if ( nsons() == 0 )
        {
            cb->set_basis( _V );
        }// if
        else
        {
            cb->set_nsons( nsons() );
            cb->set_transfer( _E );
        }// else

        // set in case of compressed V/E        
        cb->_rank = _rank;
        
        #if HLR_HAS_COMPRESSION == 1
        if ( is_compressed() )
        {
            cb->_zV = _zV;
            cb->_zE.resize( _zE.size() );
            for ( uint  i = 0; i < _zE.size(); ++i )
                cb->_zE[i] = _zE[i];
        }// if

        #if HLR_USE_APLR_NESTED_CB == 1
        cb->_sv = std::move( blas::copy( _sv ) );
        #endif
        #endif
        
        cb->set_nsons( nsons() );

        for ( uint  i = 0; i < nsons(); ++i )
            cb->set_son( i, son(i)->copy().release() );

        return cb;
    }
    
    // return structural copy (no data) of cluster basis
    std::unique_ptr< nested_cluster_basis >
    copy_struct () const
    {
        auto  cb = std::make_unique< nested_cluster_basis >( _is );

        cb->set_nsons( nsons() );

        for ( uint  i = 0; i < nsons(); ++i )
            cb->set_son( i, son(i)->copy_struct().release() );

        return cb;
    }
    
    // return memory consumption
    size_t
    byte_size () const
    {
        size_t  n = ( sizeof(_is) + sizeof(nested_cluster_basis< value_t > *) * _sons.size() + _V.byte_size() + sizeof(_E));

        for ( uint  i = 0; i < nsons(); ++i )
            n += _E[i].byte_size();
        
        #if HLR_HAS_COMPRESSION == 1
        #if HLR_USE_APLR_NESTED_CB  == 1
        n += compress::aplr::byte_size( _zV );
        n += _sv.byte_size();
        #else
        n += hlr::compress::byte_size( _zV );
        #endif
        
        n += sizeof(_zE);
        
        for ( uint  i = 0; i < _zE.size(); ++i )
            n += hlr::compress::byte_size( _zE[i] );
        
        #endif
        
        for ( auto  son : _sons )
            n += son->byte_size();

        return  n;
    }

    // return depth of cluster basis
    size_t
    depth () const
    {
        size_t  d = 0;

        for ( auto  son : _sons )
            d = std::max( d, son->depth() );

        return d+1;
    }

    //
    // access mutex
    //

    decltype(_mtx) &  mutex () { return _mtx; }

    //
    // compression
    //

    // compress internal data based on given configuration
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const compress::zconfig_t &  zconfig );

    // compress internal data based on given accuracy
    virtual void   compress      ( const Hpro::TTruncAcc &  acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        #if HLR_HAS_COMPRESSION == 1
        return (( _zV.size() > 0 ) || ( _zE.size() > 0 ));
        #endif
        
        return false;
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        #if HLR_HAS_COMPRESSION == 1
        #if HLR_USE_APLR_NESTED_CB  == 1
        _zV = compress::aplr::zarray();
        #else
        _zV = compress::zarray();
        #endif
        _zE.resize( 0 );
        #endif
    }
};

//
// copy given cluster basis between different value types
//
template < typename value_dest_t,
           typename value_src_t = value_dest_t >
std::unique_ptr< nested_cluster_basis< value_dest_t > >
copy ( const nested_cluster_basis< value_src_t > &  src )
{
    auto  V_dest = blas::copy< value_dest_t, value_src_t >( src.basis() );
    auto  dest   = std::make_unique< nested_cluster_basis< value_dest_t > >( src.is(), std::move( V_dest ) );

    dest->set_nsons( src.nsons() );

    HLR_ERROR( "TODO" );
    
    for ( uint  i = 0; i < dest->nsons(); ++i )
    {
        if ( ! is_null( src.son( i ) ) )
            dest->set_son( i, copy< value_dest_t >( * src.son(i) ).release() );
    }// for

    return dest;
}

//
// compress internal data
//
template < typename value_t >
void
nested_cluster_basis< value_t >::compress ( const compress::zconfig_t &  zconfig )
{
    if ( is_compressed() )
        return;

    HLR_ASSERT( false );
}

template < typename value_t >
void
nested_cluster_basis< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    if ( is_compressed() )
        return;

    #if HLR_HAS_COMPRESSION == 1
        
    HLR_ASSERT( acc.rel_eps() == 0 );

    if ( nsons() == 0 )
    {
        if ( _V.nrows() * _V.ncols() == 0 )
            return;
        
        const size_t  mem_dense = sizeof(value_t) * _V.nrows() * _V.ncols();

        #if HLR_USE_APLR_NESTED_CB == 1
        {
            //
            // use adaptive precision per basis vector
            //

            HLR_ASSERT( _sv.length() == _V.ncols() );

            real_t  tol = acc.abs_eps() * _sv(0);
            auto    S   = blas::copy( _sv );

            for ( uint  l = 0; l < S.length(); ++l )
                S(l) = tol / S(l);
        
            auto  zV   = compress::aplr::compress_lr< value_t >( _V, S );
            auto  zmem = compress::aplr::compressed_size( zV );
            
            // {
            //     auto  T = blas::copy( _V );

            //     compress::aplr::decompress_lr< value_t >( zV, T );

            //     blas::add( value_t(-1), _V, T );
            //     std::cout << blas::norm_F( T ) << " / " << blas::norm_F( T ) / blas::norm_F( _V ) << std::endl;
            // }
        
            if (( zmem > 0 ) && ( zmem < mem_dense ))
            {
                _zV = std::move( zV );
                _V  = std::move( blas::matrix< value_t >( 0, _V.ncols() ) ); // remember rank
            }// if

            return;
        }// if
        #endif

        //
        // use standard compression
        //
    
        auto  zconfig = compress::get_config( acc.abs_eps() );
        auto  zV      = compress::compress< value_t >( zconfig, _V );
        auto  zmem    = compress::compressed_size( zV );

        if (( zmem > 0 ) && ( zmem < mem_dense ))
        {
            _zV = std::move( zV );
            _V  = std::move( blas::matrix< value_t >( 0, _V.ncols() ) ); // remember rank
        }// if
    }// if
    else
    {
        if ( _E.size() == 0 )
            return;

        size_t  mem_dense = 0;
        size_t  mem_compr = 0;
        auto    zconfig   = compress::get_config( acc.abs_eps() );
        auto    zE        = std::vector< compress::zarray >( _E.size() );
        
        for ( uint  i = 0; i < _E.size(); ++i )
        {
            zE[i]      = compress::compress< value_t >( zconfig, _E[i] );
            
            mem_dense += sizeof(value_t) * _E[i].nrows() * _E[i].ncols();
            mem_compr += compress::compressed_size( zE[i] );
        }// for
            
        if (( mem_compr > 0 ) && ( mem_compr < mem_dense ))
        {
            _zE = std::move( zE );

            for ( uint  i = 0; i < _E.size(); ++i )
                _E[i] = std::move( blas::matrix< value_t >( _E[i].nrows(), 0 ) ); // remember nrows
        }// if
    }// else
    
    #endif
}

//
// decompress internal data
//
template < typename value_t >
void
nested_cluster_basis< value_t >::decompress ()
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( ! is_compressed() )
        return;

    if ( nsons() == 0 )
    {
        _V = std::move( basis() );
    }// if
    else
    {
        for ( uint  i = 0; i < _E.size(); ++i )
            _E[i] = std::move( transfer_mat(i) );
    }// else
    
    remove_compressed();

    #endif
}

//
// return min/avg/max rank of given cluster basis
//
template < typename value_t >
std::tuple< uint, uint, uint >
rank_info ( const nested_cluster_basis< value_t > &  cb )
{
    auto [ min_rank, sum_rank, max_rank, nnodes ] = detail::rank_info_helper_cb( cb );

    return { min_rank, uint( double(sum_rank) / double(nnodes) ), max_rank };
}

#if defined(HLR_HAS_H2)

template < typename value_t >
std::tuple< uint, uint, uint >
rank_info ( const Hpro::TClusterBasis< value_t > &  cb )
{
    auto [ min_rank, sum_rank, max_rank, nnodes ] = detail::rank_info_helper_cb( cb );

    return { min_rank, uint( double(sum_rank) / double(nnodes) ), max_rank };
}

#endif

#undef HLR_USE_APLR_NESTED_CB

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_NESTED_CLUSTER_BASIS_HH

#ifndef __HLR_CLUSTER_WEAKADM_HH
#define __HLR_CLUSTER_WEAKADM_HH
//
// Project     : HLR
// Module      : cluster/weakadm.cc
// Description : weak admissibility for overlap check
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/cluster/TAdmCondition.hh>
#include <hpro/cluster/TGeomAdmCond.hh>
#include <hpro/cluster/TGeomCluster.hh>

#include <hlr/utils/term.hh>

namespace hlr { namespace cluster {

using strong_adm_cond = Hpro::TStdGeomAdmCond;

//
// implements weak admissibility with axis aligned
// bbox test for overlap and admissibility based
// on user defined codimension
//
class weak_adm_cond : public Hpro::TAdmCondition
{
private:
    // allowed codimension of cluster overlap
    const uint    _codim;

    // permitted overlap ratio
    const double  _overlap;
    
    // parameter for standard admissibility
    const double  _eta;
    
public:
    // ctor
    weak_adm_cond ( const uint    codim,
                    const double  overlap = 0.0,
                    const double  eta     = 2.0 )
            : TAdmCondition()
            , _codim( codim )
            , _overlap( overlap )
            , _eta( eta )
    {}

    virtual ~weak_adm_cond () {}

    //
    // return true if overlap of τ(cl) and σ(cl) has at least codimension <codim>
    //
    virtual bool is_adm ( const Hpro::TBlockCluster *  cl ) const
    {
        HLR_ASSERT( Hpro::is_geom_cluster( cl->rowcl() ) && Hpro::is_geom_cluster( cl->colcl() ) );
    
        auto  rowcl = cptrcast( cl->rowcl(), Hpro::TGeomCluster );
        auto  colcl = cptrcast( cl->colcl(), Hpro::TGeomCluster );

        // identical clusters are always inadmissibile
        if ( rowcl == colcl )
            return false;

        //
        // nested dissection case: domain-domain clusters are admissible
        //

        if ( rowcl->is_domain() && colcl->is_domain() )
            return true;

        //
        // compute codimension of overlap
        //

        const uint  dim   = rowcl->bbox().min().dim();
        uint        codim = 0;
    
        // in 1D, different clusters share at most one vertex ⇒ admissible
        if ( dim == 1 )
            return true;

        const auto  rbox = rowcl->bbox();
        const auto  cbox = colcl->bbox();
        
        HLR_ASSERT( ( rbox.max().dim() == dim ) && 
                    ( cbox.min().dim() == dim ) &&
                    ( cbox.max().dim() == dim ) );

        for ( uint  i = 0; i < dim; ++i )
        {
            const auto  rmin   = rbox.min()[i];
            const auto  rmax   = rbox.max()[i];
            
            const auto  cmin   = cbox.min()[i];
            const auto  cmax   = cbox.max()[i];
            
            const auto  rlen   = rmax - rmin;
            const auto  clen   = cmax - cmin;
            const auto  minlen = std::min( rlen, clen );
            
            if (( rmax <= cmin ) ||   // ├── τ ──┤├── σ ──┤
                ( cmax <= rmin ))     // ├── σ ──┤├── τ ──┤
            {
                codim++;
            }// if
            else if ( _overlap > 0 )
            {
                //
                // test relative overlap size
                //

                //               h
                // test ├── τ ──┼─┼── σ ──┤
                if (( rmax >= cmin ) && ( rmax <= cmax ) && (( rmax - cmin ) < _overlap ))
                    codim++;
                    
                // test ├── σ ──┼┼── τ ──┤
                else if (( cmax >= rmin ) && ( cmax <= rmax ) && (( cmax - rmin ) < _overlap ))
                    codim++;
            }// if
        }// for

        // auto  inter = Hpro::intersection( rbox, cbox );
        // auto  l0    = inter.max()[0] - inter.min()[0];
        // auto  l1    = inter.max()[1] - inter.min()[1];
        // auto  l2    = 0; // inter.max()[2] - inter.min()[2];
        // double  ol  = 1e10;

        // for ( uint  i = 0; i < 2; ++i )
        //     if ( inter.max()[i] - inter.min()[i] > 0 )
        //         ol = std::min( ol, inter.max()[i] - inter.min()[i] );
        
        // if ( codim <= 1 )
        //     std::cout << rbox.to_string() << " x " << cbox.to_string() << " : "
        //               << term::bold() << codim << ", " << ol << term::reset()
        //               << " / " << l0 << " / " << l1 << " / " << l2 << std::endl;
        // else if ( codim == 2 )
        //     std::cout << term::red() << rbox.to_string() << " x " << cbox.to_string() << " : "
        //               << term::bold() << codim << ", " << ol << term::reset()
        //               << " / " << l0 << " / " << l1 << " / " << l2 << std::endl;
        // else if ( codim == 3 )
        //     std::cout << term::green() << rbox.to_string() << " x " << cbox.to_string() << " : "
        //               << term::bold() << codim << ", " << ol << term::reset()
        //               << " / " << l0 << " / " << l1 << " / " << l2 << std::endl;
        
        if ( codim >= _codim )
            return true;
    
        //
        // test standard admissibility
        //

        return std::min( rbox.diameter(), cbox.diameter() ) <= ( _eta * rbox.distance( cbox ) );
    }
};

}}// namespace hlr::cluster

#endif // __HLR_CLUSTER_WEAKADM_HH

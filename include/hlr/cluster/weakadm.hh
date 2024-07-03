#ifndef __HLR_CLUSTER_WEAKADM_HH
#define __HLR_CLUSTER_WEAKADM_HH
//
// Project     : HLR
// Module      : cluster/weakadm.cc
// Description : weak admissibility for overlap check
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

namespace hlr { namespace cluster {

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

    // parameter for standard admissibility
    const double  _eta;
    
public:
    // ctor
    weak_adm_cond ( const uint    codim,
                    const double  eta = 1.0 )
            : TAdmCondition()
            , _codim( codim )
            , _eta( eta )
    {}

    virtual ~weak_adm_cond () {}

    //
    // return true if overlap of τ(cl) and σ(cl) has at least codimension <codim>
    //
    virtual bool is_adm ( const Hpro::TBlockCluster *  cl ) const
    {
        HLR_ASSERT( Hpro::is_geom_cluster( cl->rowcl() ) && Hpro::is_geom_cluster( cl->colcl() ) );
    
        auto  rowcl = cptrcast( c->rowcl(), Hpro::TGeomCluster );
        auto  colcl = cptrcast( c->colcl(), Hpro::TGeomCluster );

        // identical clusters are always inadmissibile
        if ( rowcl == colcl )
            return false;

        //
        // nested dissection case: domain-domain clusters are admissible
        //

        if ( rowcl->is_domain() && colcl->is_domain() )
            return true;

        //
        // compute codimension of overlap: empty intersection reduces codimension
        //

        const uint  dim   = rowcl->bbox().min().dim();
        uint        codim = dim;
    
        // in 1D, different clusters share at most one vertex => admissible
        if ( dim == 1 )
            return true;

        const auto  rbbox = rowcl->bbox();
        const auto  cbbox = colcl->bbox();
        
        HLR_ASSERT( ( rbbox.max().dim() == dim ) && 
                    ( cbbox.min().dim() == dim ) &&
                    ( cbbox.max().dim() == dim ) );

        // TODO: add "h" as overlap
        
        for ( uint  i = 0; i < dim; ++i )
        {
            if (( rbbox.max()[i] <= cbbox.min()[i] ) ||   // ├── τ ──┼── σ ──┤
                ( cbbox.max()[i] <= rbbox.min()[i] ))     // ├── σ ──┼── τ ──┤
                codim--;
        }// for

        if ( codim >= _codim )
            return true;
    
        //
        // test standard admissibility
        //

        return std::min( rowcl->diameter(), colcl->diameter() ) <= ( _eta * rowcl->distance( colcl ) );
    }
};

}}// namespace hlr::cluster

#endif // __HLR_CLUSTER_WEAKADM_HH

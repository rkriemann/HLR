//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG sequentially
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>

#include "utils/tools.hh"

#include "seq/dag.hh"

namespace HLR
{

namespace SEQ
{

//
// execute DAG <dag>
//
void
DAGExecution::run ( DAG::Graph &             dag,
                    const HLIB::TTruncAcc &  acc ) const
{
    // holds pending tasks
    std::list< DAG::Node * >  worklist;

    for ( auto  t : dag.start() )
        worklist.push_back( t );
    
    while ( ! worklist.empty() )
    {
        auto  t = behead( worklist );

        log( 4, t->to_string() );
        
        t->run( acc );

        for ( auto  succ : t->successors() )
        {
            auto  deps = succ->dec_dep_cnt();

            assert( deps >= 0 );
            
            if ( deps == 0 )
                worklist.push_front( succ );
        }// for
    }// while
}

}// namespace SEQ

}// namespace DAG

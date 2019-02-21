//
// Project     : HLib
// File        : seq_run.hh
// Description : execute DAG sequentially
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>

#include "../tools.hh"

#include "Graph.hh"

namespace DAG
{

namespace SEQ
{

//
// execute DAG <dag>
//
void
run ( Graph &                  dag,
      const HLIB::TTruncAcc &  acc )
{
    // holds pending tasks
    node_list_t  worklist;

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

//
// Project     : HLib
// File        : dag.cc
// Description : execute DAG using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>
#include <vector>
#include <unordered_map>

#include "utils/tools.hh"
#include "utils/log.hh"

#include "omp/dag.hh"

using namespace HLIB;

namespace HLR 
{

namespace DAG
{

namespace OMP
{

//
// execute DAG <dag>
//
void
run ( DAG::Graph &             dag,
      const HLIB::TTruncAcc &  acc )
{
    auto tic = Time::Wall::now();

    // keep track of dependencies for a node
    using  dep_list_t = std::list< Node * >;
    using  dep_vec_t  = std::vector< Node * >;

    std::unordered_map< Node *, dep_list_t >  dep_lists;
    std::unordered_map< Node *, dep_vec_t  >  dep_vecs;

    //
    // add node to dependency list of successors
    //
    
    for ( auto  node : dag.nodes() )
        for ( auto  succ : node->successors() )
            dep_lists[ succ ].push_back( node );

    //
    // convert lists to vectors
    //

    for ( auto  node : dag.nodes() )
    {
        dep_list_t &  dep_list = dep_lists[ node ];

        dep_vecs[ node ].reserve( dep_list.size() );

        for ( auto  dep : dep_list )
            dep_vecs[ node ].push_back( dep );
    }// for
    
    auto  toc = Time::Wall::since( tic );

    log( 2, "time for OMP DAG prepare = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
    
    //
    // loop through nodes and create OpenMP task with dependencies from dep_vecs
    //

    tic = Time::Wall::now();
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            for ( auto  node : dag.nodes() )
            {
                dep_vec_t &  deps  = dep_vecs[ node ];
                auto         cdeps = & deps[0];
                const auto   ndeps = deps.size();
                
                #pragma omp task depend( in : cdeps[0:ndeps] ) depend( out : node )
                {
                    node->run( acc );
                }// omp task
            }// for
        }// omp single
    }// omp parallel

    toc = Time::Wall::since( tic );

    log( 2, "time for OMP DAG run     = " + HLIB::to_string( "%.2fs", toc.seconds() ) );
}

}// namespace OMP

}// namespace DAG

}// namespace HLR

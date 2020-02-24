#ifndef __HLR_GASPI_HH
#define __HLR_GASPI_HH
//
// Project     : HLib
// File        : gaspi.hh
// Description : C++ GASPI/GPI wrapper
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <initializer_list>
#include <vector>

#include <GASPI.h>

#include "hlr/utils/log.hh"
#include "hlr/utils/tools.hh"
#include "hlr/utils/text.hh"

namespace hlr
{

namespace gaspi
{

//
// automatic check for each GASPI call
//
#define GASPI_CHECK_RESULT( Func, Args )                                \
    {                                                                   \
        if ( HLIB::verbose( 6 ) )                                       \
            hlr::log( 6, std::string( __ASSERT_FUNCTION ) + " : " + #Func ); \
        else if ( HLIB::verbose( 5 ) )                                  \
            hlr::log( 5, #Func );                                       \
                                                                        \
        auto  check_result = Func Args;                                 \
                                                                        \
        if ( check_result != GASPI_SUCCESS )                            \
        {                                                               \
            gaspi_string_t  err_msg;                                    \
            gaspi_print_error( check_result, & err_msg );               \
            hlr::log( 0, std::string( " in " ) + #Func + " : " + err_msg ); \
            std::exit( 1 );                                             \
        }                                                               \
    }

//
// import types
//
using  number_t          = gaspi_number_t;
using  size_t            = gaspi_size_t;
using  rank_t            = gaspi_rank_t;
using  group_t           = gaspi_group_t;
using  notification_id_t = gaspi_notification_id_t;
using  notification_t    = gaspi_notification_t;
using  segment_id_t      = gaspi_segment_id_t;

//
// general GASPI environment for initialization/finalization
//
class environment
{
public:
    environment ()
    {
        GASPI_CHECK_RESULT( gaspi_proc_init,
                            ( GASPI_BLOCK ) ); // all function calls are blocking
    }

    ~environment ()
    {
        GASPI_CHECK_RESULT( gaspi_proc_term,
                            ( GASPI_BLOCK ) );
    }
};

//
// GASPI process
//
class process
{
public:
    // return number of processes
    static rank_t
    size ()
    {
        rank_t  n = 0;

        GASPI_CHECK_RESULT( gaspi_proc_num,
                            ( & n ) );

        return n;
    }

    // return rank of calling process
    static rank_t
    rank ()
    {
        rank_t  n = 0;

        GASPI_CHECK_RESULT( gaspi_proc_rank,
                            ( & n ) );

        return n;
    }

    // return maximal number of simultaneous requests per queue
    static number_t
    nmax_requests ()
    {
        number_t  n = 0;
        
        GASPI_CHECK_RESULT( gaspi_queue_size_max, ( & n ) );

        return n;
    }

    // return number of allocated segments
    static number_t
    nalloc_segments ()
    {
        number_t  n = 0;
        
        GASPI_CHECK_RESULT( gaspi_segment_num, ( & n ) );

        return n;
    }

    // return maximal number of segments
    static number_t
    nmax_segments ()
    {
        number_t  n = 0;
        
        GASPI_CHECK_RESULT( gaspi_segment_max, ( & n ) );

        return n;
    }

    // process-wide barrier
    static void
    barrier ()
    {
        GASPI_CHECK_RESULT( gaspi_barrier,
                            ( GASPI_GROUP_ALL, GASPI_BLOCK ) );
    }
};

//
// group of processes
//
class group
{
private:
    group_t  _gaspi_group;

public:
    group ()
            : _gaspi_group( GASPI_GROUP_ALL )
    {
        GASPI_CHECK_RESULT( gaspi_group_commit, ( _gaspi_group, GASPI_BLOCK ) );
    }

    group ( const gaspi_group_t &  g )
            : _gaspi_group( g )
    {}

    group ( group &&  g )
            : _gaspi_group( g._gaspi_group )
    {
        g._gaspi_group = GASPI_GROUP_ALL;
    }

    group ( const group &  g )
            : _gaspi_group( g._gaspi_group )
    {}

    group ( std::initializer_list< rank_t >  ranks )
    {
        GASPI_CHECK_RESULT( gaspi_group_create, ( & _gaspi_group ) );

        for ( auto  rank : ranks )
            GASPI_CHECK_RESULT( gaspi_group_add, ( _gaspi_group, rank ) );
            
        GASPI_CHECK_RESULT( gaspi_group_commit, ( _gaspi_group, GASPI_BLOCK ) );
    }

    template < typename T_container >
    group ( const T_container &  ranks )
    {
        GASPI_CHECK_RESULT( gaspi_group_create, ( & _gaspi_group ) );

        for ( auto  rank : ranks )
            GASPI_CHECK_RESULT( gaspi_group_add, ( _gaspi_group, rank ) );
            
        GASPI_CHECK_RESULT( gaspi_group_commit, ( _gaspi_group, GASPI_BLOCK ) );
    }

    ~group ()
    {
        if ( _gaspi_group != GASPI_GROUP_ALL )
        {
            hlr::log( 5, "active group" );
            // GASPI_CHECK_RESULT( gaspi_group_delete, ( _gaspi_group ) );
        }// if
    }
    
    group &  operator = ( group &&  g )
    {
        _gaspi_group   = g._gaspi_group;
        g._gaspi_group = GASPI_GROUP_ALL;

        return *this;
    }

    group &  operator = ( const group &  g )
    {
        _gaspi_group = g._gaspi_group;
        
        return *this;
    }
    
    operator gaspi_group_t () const { return _gaspi_group; }

    // group-wide barrier
    void
    barrier () const
    {
        GASPI_CHECK_RESULT( gaspi_barrier, ( _gaspi_group, GASPI_BLOCK ) );
    }

    // return string representation of group (list of members)
    std::string
    to_string () const
    {
        number_t  n = 0;
        
        GASPI_CHECK_RESULT( gaspi_group_size, ( _gaspi_group, & n ) );

        std::vector< rank_t >  ranks( n );
        
        GASPI_CHECK_RESULT( gaspi_group_ranks, ( _gaspi_group, & ranks[0] ) );

        return hlr::to_string( ranks );
    }
};

//
// segment representing global memory area
//
class segment
{
private:
    segment_id_t  _id;
    size_t        _size;
    
public:
    template < typename value_t >
    segment ( const segment_id_t  sid,
              value_t *           base,
              const size_t        asize,
              group &             grp )
            : _id( sid )
            , _size( asize * sizeof(value_t) )
    {
        assert( sid < process::nmax_segments() );
        GASPI_CHECK_RESULT( gaspi_segment_use, ( _id, base, _size, gaspi_group_t( grp ), GASPI_BLOCK, 0 ) );
    }

    segment ( segment &&  seg )
            : _id(   seg._id )
            , _size( seg._size )
    {
        seg._id   = 0;
        seg._size = 0;
    }

    // segment ( const segment &  seg )
    //         : _id( seg._id )
    // {}

    ~segment ()
    {
        if ( _size > 0 )
        {
            hlr::log( 0, "segment is not free" );
            // release();
        }// if
    }

    segment &
    operator = ( segment &&  seg )
    {
        _id   = seg._id;
        _size = seg._size;

        seg._id   = 0;
        seg._size = 0;

        return *this;
    }
    
    // segment &
    // operator = ( const segment &  seg )
    // {
    //     _id = win._id;

    //     return *this;
    // }

    void release ()
    {
        assert( _size > 0 );
        
        GASPI_CHECK_RESULT( gaspi_segment_delete,
                            ( _id ) );

        _id   = 0;
        _size = 0;
    }

    segment_id_t  id   () const { return _id; }
    size_t        size () const { return _size; }
};

//
// queue
//
class queue
{
private:
    gaspi_queue_id_t  _id;

public:
    queue ()
    {
        GASPI_CHECK_RESULT( gaspi_queue_create, ( & _id, GASPI_BLOCK ) );
    }

    ~queue ()
    {
        GASPI_CHECK_RESULT( gaspi_queue_delete, ( _id ) );
    }

    // send local data to remove rank
    // - local and remove data offset are assumed to be zero
    void
    write_notify ( const segment &            src_seg,             // source segment to send
                   const rank_t               dest_rank,           // destination rank to sent to
                   const segment_id_t &       dest_seg_id,         // id of destination segment to write to
                   const notification_id_t &  dest_note_id,        // notification id to signal on remote rank
                   const number_t &           dest_note_val = 1 )  // (optional) value of notification
    {
        GASPI_CHECK_RESULT( gaspi_write_notify, ( src_seg.id(),                 
                                                  0,                 // source offset
                                                  dest_rank,
                                                  dest_seg_id,
                                                  0,                 // destination offset
                                                  src_seg.size(),    // data size
                                                  dest_note_id,       
                                                  dest_note_val,       
                                                  _id,               // queue to submit request to
                                                  GASPI_BLOCK ) );
    }
    
    // read remote data from remove rank
    // - local and remove data offset are assumed to be zero
    void
    read ( const segment &       dest_seg,           // local destination segment to write to
           const rank_t          src_rank,           // source rank to read from
           const segment_id_t &  src_seg )           // id of source segment to read from
    {
        GASPI_CHECK_RESULT( gaspi_read, ( dest_seg.id(),                 
                                          0,                 // source offset
                                          src_rank,
                                          src_seg,
                                          0,                 // destination offset
                                          dest_seg.size(),   // data size
                                          _id,               // queue to submit request to
                                          GASPI_BLOCK ) );
    }
    
    // wait for all communication request in queue to finish
    void
    wait () const
    {
        GASPI_CHECK_RESULT( gaspi_wait, ( _id, GASPI_BLOCK ) );
    }

    // return number of open communication requests in queue
    number_t
    size () const
    {
        number_t  n;
        
        GASPI_CHECK_RESULT( gaspi_queue_size, ( _id, & n ) );

        return n;
    }
};

//
// wait for notification to finish
//
notification_t
notify_wait ( const segment &            seg,       // local segment affected by communication
              const notification_id_t &  note_id )  // local notification id
{
    notification_id_t  first;
    
    GASPI_CHECK_RESULT( gaspi_notify_waitsome, ( seg.id(),      // local segment
                                                 note_id,       // local notification to wait for
                                                 1,             // wait for 1 notification
                                                 & first,       // first that have been received
                                                 GASPI_BLOCK ) );

    assert( first == note_id );
        
    notification_t  old_val;
    
    GASPI_CHECK_RESULT( gaspi_notify_reset,    ( seg.id(),       // local segment
                                                 note_id,        // local notification to reset
                                                 & old_val ) );  // old notification value

    return old_val;
}
              
              

}// namespace gaspi

}// namespace hlr

#endif // __HLR_GASPI_HH

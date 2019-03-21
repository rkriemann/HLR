#ifndef  __HLR_GASPI_HH
#define  __HLR_GASPI_HH
//
// Project     : HLib
// File        : gaspi.hh
// Description : C++ GASPI/GPI wrapper
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>

#include <GASPI.h>

#include "utils/log.hh"

namespace HLR
{

namespace GASPI
{

//
// automatic check for each GASPI call
//
#define GASPI_CHECK_RESULT( Func, Args )                                \
    {                                                                   \
        HLR::log( 5, std::string( __ASSERT_FUNCTION ) + " : " + #Func ); \
        auto _check_result = Func Args;                                 \
        assert( _check_result == GASPI_SUCCESS );                       \
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
    rank_t
    size () const
    {
        rank_t  n = 0;

        GASPI_CHECK_RESULT( gaspi_proc_num,
                            ( & n ) );

        return n;
    }

    // return rank of calling process
    rank_t
    rank () const
    {
        rank_t  n = 0;

        GASPI_CHECK_RESULT( gaspi_proc_rank,
                            ( & n ) );

        return n;
    }

    // return maximal number of simultaneous requests per queue
    number_t
    nmax_requests ()
    {
        number_t  n = 0;
        
        GASPI_CHECK_RESULT( gaspi_queue_size_max, ( & n ) );

        return n;
    }

    // return number of allocated segments
    number_t
    nalloc_segments ()
    {
        number_t  n = 0;
        
        GASPI_CHECK_RESULT( gaspi_segment_num, ( & n ) );

        return n;
    }

    // return maximal number of segments
    number_t
    nmax_segments ()
    {
        number_t  n = 0;
        
        GASPI_CHECK_RESULT( gaspi_segment_max, ( & n ) );

        return n;
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

    group ( const gaspi_group_t &  group )
            : _gaspi_group( group )
    {}

    // access MPI groupunicator
    operator gaspi_group_t () const { return _gaspi_group; }

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
        GASPI_CHECK_RESULT( gaspi_segment_use, ( _id, base, _size, gaspi_group_t( grp ), GASPI_BLOCK, 0 ) );
    }

    segment ( segment &&  seg )
            : _id(   seg._id )
            , _size( seg._size )
    {
        seg._id   = 0;
        seg._size = 0;
    }

    // segment ( const segment &  win )
    //         : _mpi_segment( win._mpi_segment )
    // {}

    ~segment ()
    {
        if ( _size > 0 )
            HLR::log( 0, "segment is not free" );
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
    // operator = ( const segment &  win )
    // {
    //     _mpi_segment = win._mpi_segment;

    //     return *this;
    // }

    void delete ()
    {
        assert( _mpi_segment != MPI_WIN_NULL );
        
        GASPI_CHECK_RESULT( gaspi_segment_delete,
                            ( & _id ) );

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
    write_notify ( const segment &            src_seg,            // source segment to send
                   const rank_t               dest_rank,          // destination rank to sent to
                   const segment &            dest_seg,           // destination segment to write to
                   const notification_id_t &  rem_note_id,        // notification id to signal on remote rank
                   const number_t &           rem_note_val = 1 )  // (optional) value of notification
    {
        notification  n;
        
        GASPI_CHECK_RESULT( gaspi_write_notify, ( src_seg.id(),                 
                                                  0,                 // source offset
                                                  dest_rank,
                                                  dest_seg.id(),
                                                  0,                 // destination offset
                                                  src_seg.size(),    // data size
                                                  rem_note_id,       
                                                  1,       
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
    void
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

    assert( frist != note_id );
        
    notification_t  old_val;
    
    GASPI_CHECK_RESULT( gaspi_notify_reset,    ( seg.id(),       // local segment
                                                 note_id,        // local notification to reset
                                                 & old_val ) );  // old notification value

    return old_val;
}
              
              

}// namespace GASPI

}// namespace HLR

#endif //  __HLR_GASPI_HH

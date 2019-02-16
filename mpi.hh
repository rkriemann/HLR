#ifndef  __MPI_HH
#define  __MPI_HH
//
// Project     : HLib
// File        : mpi.hh
// Description : C++ MPI wrapper
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <mpi.h>
#include <cassert>
#include <vector>

namespace mpi
{

#define MPI_CHECK_RESULT( MPIFunc, Args )                               \
    {                                                                   \
        int _check_result = MPIFunc Args;                               \
        assert(_check_result == MPI_SUCCESS);                           \
    }

//
// general MPI environment for initialization/finalization
//
class environment
{
public:
    environment ( int &      argc,
                  char ** &  argv )
    {
        int  provided = MPI_THREAD_SINGLE;

        MPI_CHECK_RESULT( MPI_Init_thread,
                          ( & argc, & argv, MPI_THREAD_SERIALIZED, & provided ) );

        assert(( provided != MPI_THREAD_SINGLE ) && ( provided != MPI_THREAD_FUNNELED ));
    }

    ~environment ()
    {
        MPI_CHECK_RESULT( MPI_Finalize,
                          () );
    }
};

//
// request for non-blocking communication
//
class request
{
public:
    MPI_Request  mpi_request;

public:
    request ()
            : mpi_request( MPI_REQUEST_NULL )
    {}

    request ( MPI_Request  req )
            : mpi_request( req )
    {}

    request &
    operator = ( MPI_Request  req )
    {
        mpi_request = req;
        return *this;
    }

    ~request ()
    {
        if ( mpi_request != MPI_REQUEST_NULL )
            wait();
    }
    
    // access MPI communicator
    operator MPI_Request () const { return mpi_request; }

    // wait for request to finish
    void
    wait ()
    {
        MPI_CHECK_RESULT( MPI_Wait,
                          ( & mpi_request, MPI_STATUSES_IGNORE ) );
        mpi_request = MPI_REQUEST_NULL;
    }
};

void
wait_all ( std::vector< request > &  reqs )
{
    std::vector< MPI_Request >  requests( reqs.size() );
    
    for ( int  i = 0; i < reqs.size(); ++i )
        requests[i] = reqs[i].mpi_request;

    MPI_CHECK_RESULT( MPI_Waitall,
                      ( reqs.size(), & requests[0], MPI_STATUSES_IGNORE ) );
}

//
// communicator
//
class communicator
{
private:
    std::shared_ptr< MPI_Comm >  _mpi_comm;

    struct free_comm
    {
        void
        operator () ( MPI_Comm *  comm ) const
        {
            MPI_CHECK_RESULT( MPI_Comm_free,
                              ( comm ) );
        }
    };
    
public:
    communicator ()
            : _mpi_comm( new MPI_Comm( MPI_COMM_WORLD ) )
    {}

    communicator ( const MPI_Comm &  comm )
            : _mpi_comm( new MPI_Comm( comm ), free_comm() )
    {}

    // return number of processes in communicator
    int
    size () const
    {
        int  n = 0;

        MPI_CHECK_RESULT( MPI_Comm_size,
                          ( MPI_Comm( *this ), & n ) );

        return n;
    }

    // return rank of calling process
    int
    rank () const
    {
        int  n = 0;

        MPI_CHECK_RESULT( MPI_Comm_rank,
                          ( MPI_Comm( *this ), & n ) );

        return n;
    }

    // access MPI communicator
    operator MPI_Comm () const
    {
        if ( _mpi_comm.get() != nullptr ) return *_mpi_comm;
        else                              return MPI_COMM_NULL;
    }

    // split communicator into multiple new communicators
    // all processes with the same <color> will be part of the
    // same communicator
    communicator  split ( int color ) const
    {
        MPI_Comm  new_comm;

        MPI_CHECK_RESULT( MPI_Comm_split,
                          ( MPI_Comm( *this ), color, 0, & new_comm ) );

        return communicator( new_comm );
    }

    //
    // communication functions
    //
    template < typename value_t >
    void
    broadcast ( value_t *       data,
                int             size,
                int             root )
    {
        MPI_CHECK_RESULT( MPI_Bcast,
                          ( data, size * sizeof(value_t), MPI_BYTE, root, MPI_Comm( *this ) ) );
    }

    template < typename value_t >
    void
    broadcast ( value_t &       data,
                int             root )
    {
        broadcast( & data, 1, root );
    }
    
    template < typename value_t >
    request
    ibroadcast ( value_t *  data,
                 int        n,
                 int        root )
    {
        MPI_Request  req;
    
        MPI_CHECK_RESULT( MPI_Ibcast,
                          ( data, n*sizeof(value_t), MPI_BYTE,
                            root, MPI_Comm( *this ),
                            & req ) );

        return req;
    }

    template < typename T >
    mpi::request
    ibroadcast ( T &  value,
                 int  root )
    {
        return ibroadcast< T >( & value, 1, root );
    }

};

class window
{
private:
    MPI_Win  _mpi_window;

public:
    template < typename value_t >
    window ( communicator &  comm,
             value_t *       base,
             int             size )
            : _mpi_window( MPI_WIN_NULL )
    {
        MPI_Win  win;
        
        MPI_CHECK_RESULT( MPI_Win_create,
                          ( base, size * sizeof(value_t), 1,
                            MPI_INFO_NULL,
                            MPI_Comm(comm),
                            & win ) );
        _mpi_window = win;
    }

    window ( window &&  win )
            : _mpi_window( win._mpi_window )
    {
        win._mpi_window = MPI_WIN_NULL;
    }

    window ( const window &  win )
            : _mpi_window( win._mpi_window )
    {}

    ~window ()
    {
        if ( _mpi_window != MPI_WIN_NULL )
            MPI_CHECK_RESULT( MPI_Win_free,
                              ( & _mpi_window ) );
    }

    operator MPI_Win () const { return _mpi_window; }

    window &
    operator = ( window &&  win )
    {
        _mpi_window     = win._mpi_window;
        win._mpi_window = MPI_WIN_NULL;

        return *this;
    }
    
    window &
    operator = ( const window &  win )
    {
        _mpi_window = win._mpi_window;

        return *this;
    }

    void
    fence ( int mode )
    {
        assert( _mpi_window != MPI_WIN_NULL );
        MPI_CHECK_RESULT( MPI_Win_fence,
                          ( mode, _mpi_window ) );
    }

    template < typename value_t >
    void
    get ( value_t *  dest,
          int        dest_size,
          int        from_rank,
          int        from_disp,
          int        from_size )
    {
        assert( _mpi_window != MPI_WIN_NULL );
        MPI_CHECK_RESULT( MPI_Get,
                          ( dest, sizeof(value_t) * dest_size, MPI_BYTE,
                            from_rank,
                            from_disp * sizeof(value_t), from_size * sizeof(value_t), MPI_BYTE,
                            _mpi_window ) );
    }

    template < typename value_t >
    request
    rget ( value_t *  dest,
           int        dest_size,
           int        from_rank,
           int        from_disp,
           int        from_size )
    {
        assert( _mpi_window != MPI_WIN_NULL );
        
        MPI_Request  req;
        
        MPI_CHECK_RESULT( MPI_Rget,
                          ( dest, sizeof(value_t) * dest_size, MPI_BYTE,
                            from_rank,
                            from_disp * sizeof(value_t), from_size * sizeof(value_t), MPI_BYTE,
                            _mpi_window, & req ) );

        return req;
    }
};

}// namespace mpi

#endif //  __MPI_HH

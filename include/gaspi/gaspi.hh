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
    gaspi_rank_t
    size () const
    {
        gaspi_rank_t  n = 0;

        GASPI_CHECK_RESULT( gaspi_proc_num,
                            ( & n ) );

        return n;
    }

    // return rank of calling process
    gaspi_rank_t
    rank () const
    {
        gaspi_rank_t  n = 0;

        GASPI_CHECK_RESULT( gaspi_proc_rank,
                            ( & n ) );

        return n;
    }
};
    
// //
// // group of processes
// //
// class group
// {
// private:
//     std::shared_ptr< gaspi_group_t >  _gaspi_group;

//     struct free_group
//     {
//         void
//         operator () ( gaspi_group_t *  group ) const
//         {
//             // GASPI_CHECK_RESULT( MPI_Group_free,
//             //                   ( group ) );
//         }
//     };
    
// public:
//     groupunicator ()
//             : _gaspi_group( new gaspi_group_t( GASPI_GROUP_ALL ) )
//     {}

//     groupunicator ( const gaspi_group_t &  group )
//             : _gaspi_group( new gaspi_group_t( group ), free_group() )
//     {}

//     // return number of processes in groupunicator
//     int
//     size () const
//     {
//         int  n = 0;

//         GASPI_CHECK_RESULT( gaspi_group_t_size,
//                           ( gaspi_group_t( *this ), & n ) );

//         return n;
//     }

//     // return rank of calling process
//     int
//     rank () const
//     {
//         int  n = 0;

//         GASPI_CHECK_RESULT( gaspi_group_t_rank,
//                           ( gaspi_group_t( *this ), & n ) );

//         return n;
//     }

//     // access MPI groupunicator
//     operator gaspi_group_t () const
//     {
//         if ( _gaspi_group.get() != nullptr ) return *_gaspi_group;
//         else                              return MPI_GROUP_NULL;
//     }

// };

}// namespace GASPI

}// namespace HLR

#endif //  __HLR_GASPI_HH

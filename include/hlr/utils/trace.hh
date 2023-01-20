#ifndef __HLR_UTILS_TRACE_HH
#define __HLR_UTILS_TRACE_HH
//
// Project     : HLib
// Module      : utils/trace
// Description : tracing functions based on Score-P
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#if defined(HAS_SCOREP)
#include <scorep/SCOREP_User.h>
#endif

namespace hlr { namespace trace {

#if defined(HAS_SCOREP)

inline
void
region_start ( const char *  region_name )
{
    SCOREP_USER_REGION_BY_NAME_BEGIN( region_name, SCOREP_USER_REGION_TYPE_COMMON )
}

inline
void
region_end ( const char *  region_name )
{
    SCOREP_USER_REGION_BY_NAME_END( region_name )
}

#else

inline void region_start ( const char * ) {}
inline void region_end   ( const char * ) {}

#endif

}}// namespace hlr::trace

#endif // __HLR_UTILS_TRACE_HH

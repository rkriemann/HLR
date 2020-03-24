#ifndef __HLR_UTILS_LIKWID_HH
#define __HLR_UTILS_LIKWID_HH
//
// Project     : HLR
// Module      : utils/likwid
// Description : wrapper for likwid functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#if defined(LIKWID_PERFMON)

#  include <likwid.h>

#else

#  define LIKWID_MARKER_INIT
#  define LIKWID_MARKER_CLOSE
#  define LIKWID_MARKER_START( t )
#  define LIKWID_MARKER_STOP( t )

#endif

#endif // __HLR_UTILS_LIKWID_HH

#ifndef __HLR_UTILS_ZFP_HH
#define __HLR_UTILS_ZFP_HH
//
// Project     : HLR
// Module      : utils/zfp
// Description : ZFP compression related functions and types
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#if defined(HAS_ZFP)

#include <zfpcarray2.h>

#else

// dummy type
struct zfp_config {};

// dummy functions
inline zfp_config  zfp_config_rate      ( double, bool ) { return zfp_config(); }
inline zfp_config  zfp_config_accuracy  ( double )       { return zfp_config(); }
inline zfp_config  zfp_config_precision ( uint )         { return zfp_config(); }

#endif

#endif // __HLR_UTILS_ZFP_HH

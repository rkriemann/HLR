#ifndef __HLR_COMPRESS_ZTYPES_HH
#define __HLR_COMPRESS_ZTYPES_HH
//
// Project     : HLR
// Module      : compress/ztypes
// Description : definition of supported compressor types
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cstdint>

//
// different compressor types
//
// - needs to be consistent with SConstruct file!
//

#define HLR_COMPRESSOR_NONE     0
#define HLR_COMPRESSOR_FP32     1
#define HLR_COMPRESSOR_AFL      2
#define HLR_COMPRESSOR_AFLP     3
#define HLR_COMPRESSOR_FPX      4
#define HLR_COMPRESSOR_ZFP      5
#define HLR_COMPRESSOR_SZ       6
#define HLR_COMPRESSOR_SZ3      7
#define HLR_COMPRESSOR_MGARD    8
#define HLR_COMPRESSOR_BLOSC    9
#define HLR_COMPRESSOR_POSITS   10
#define HLR_COMPRESSOR_CFLOAT   11
#define HLR_COMPRESSOR_FIXED    12

#endif // __HLR_COMPRESS_ZTYPES_HH

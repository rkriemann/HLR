#ifndef __HLR_COMPRESS_ZTYPES_HH
#define __HLR_COMPRESS_ZTYPES_HH
//
// Project     : HLR
// Module      : compress/ztypes
// Description : definition of supported compressor types
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

//
// different compressor types
//
// - needs to be consistent with SConstruct file!
//

#define HLR_COMPRESSOR_NONE     0
#define HLR_COMPRESSOR_AFL      1
#define HLR_COMPRESSOR_AFLP     2
#define HLR_COMPRESSOR_SFL      3
#define HLR_COMPRESSOR_DFL      4
#define HLR_COMPRESSOR_ZFP      5
#define HLR_COMPRESSOR_SZ       6
#define HLR_COMPRESSOR_SZ3      7
#define HLR_COMPRESSOR_MGARD    8
#define HLR_COMPRESSOR_LZ4      9
#define HLR_COMPRESSOR_ZLIB     10
#define HLR_COMPRESSOR_ZSTD     11
#define HLR_COMPRESSOR_POSITS   12
#define HLR_COMPRESSOR_FP32     13
#define HLR_COMPRESSOR_FP16     14
#define HLR_COMPRESSOR_BF16     15
#define HLR_COMPRESSOR_TF32     16
#define HLR_COMPRESSOR_BF24     17
#define HLR_COMPRESSOR_MP       18
#define HLR_COMPRESSOR_MP2      19
#define HLR_COMPRESSOR_CFLOAT   20
#define HLR_COMPRESSOR_BLOSC    21
#define HLR_COMPRESSOR_DFL2     22

#endif // __HLR_COMPRESS_ZTYPES_HH

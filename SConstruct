# -*- coding: utf-8 -*-

import os, sys
import re

from datetime import datetime

######################################################################
#
# general (default) settings
#
######################################################################

BUILD_TYPES  = [ 'debug', 'dbg', 'release', 'rel', 'release-debug', 'reldbg' ]
fullmsg      = False
buildtype    = 'debug'
warn         = False
color        = True

# cache file storing SCons settings
opts_file    = '.scons.options'

CXX          = 'g++'
CXXFLAGS     = '-std=c++20'
CPUFLAGS     = 'cpuflags'

DBGFLAGS     = '-g -frounding-math'
RELDBGFLAGS  = '-g -O2 -march=native'
RELFLAGS     = '-O3 -fomit-frame-pointer -ffast-math -funroll-loops -march=native'
OPTFLAGS     = RELFLAGS
WARNFLAGS    = '' # '-Wall'
LINKFLAGS    = '-g'
DEFINES      = 'TBB_PREVIEW_GLOBAL_CONTROL __TBB_show_deprecation_message_task_H'

# directories for the various external libraries
HPRO_DIR     = '/'
MKL_DIR      = '/'
TBB_DIR      = '/'
TASKFLOW_DIR = '/'
HPX_DIR      = '/'
GPI2_DIR     = '/'
CUDA_DIR     = '/'

eigen        = 0
EIGEN_DIR    = '/'

hdf5         = 0
hdf5_dir     = '/'

# general allocators
JEMALLOC_DIR = '/'
MIMALLOC_DIR = '/'
TCMALLOC_DIR = '/'

# tracing
likwid        = False
LIKWID_DIR    = '/'
scorep        = False
SCOREP_DIR    = '/'

# compressors
half          = False
HALF_DIR      = '/'
zfp           = False
ZFP_DIR       = '/'
sz            = False
SZ_DIR        = '/'
sz3           = False
SZ3_DIR       = '/'
mgard         = False
MGARD_DIR     = '/'
lz4           = False
LZ4_DIR       = '/'
zlib          = False
ZLIB_DIR      = '/'
zstd          = False
ZSTD_DIR      = '/'
universal     = False
UNIVERSAL_DIR = '/'
blosc         = False
BLOSC_DIR     = '/'
atc           = False
ATC_DIR       = '/'

zblas         = True

# default values for programs and frameworks
programs     = ''
frameworks   = 'seq,tbb'

# set of frameworks to use: seq, openmp, tbb, tf, hpx, mpi, gpi2 (or 'all')
FRAMEWORKS   = [ 'help',        # print help
                 'seq',
                 'omp',
                 'tbb',
                 'tf',
                 'hpx',
                 'mpi',
                 'gpi2',
                 'cuda' ]

FRAMEWORK_HELP = { 'seq'  : 'uses sequential execution',
                   'omp'  : 'uses OpenMP',
                   'tbb'  : 'uses Threading Building Blocks (see also {0}tbb_dir{1})',
                   'tf'   : 'uses C++-Taskflow              (see also {0}tf_dir{1})',
                   'hpx'  : 'uses HPX                       (see also {0}hpx_dir{1})',
                   'mpi'  : 'uses MPI; can be combined with other framework',
                   'gpi2' : 'uses GPI-2/GASPI               (see also {0}gpi2_dir{1})',
                   'cuda' : 'uses CUDA (addon framework)' }

# supported lapack libraries
LAPACKLIBS   = [ 'help',        # print help
                 'default',     # default system implementation, e.g., -llapack -lblas
                 'none',        # do not use any LAPACK library
                 'user',        # use user defined LAPACK library (see "--lapack-flags")
                 'mkl',         # use parallel Intel MKL (should be OpenMP version)
                 'mklomp',      # use OpenMP based Intel MKL
                 'mkltbb',      # use TBB based Intel MKL
                 'mklseq',      # use sequential Intel MKL
                 'mklomp64',    # use OpenMP based Intel MKL (ILP64)
                 'mkltbb64',    # use TBB based Intel MKL (ILP64)
                 'mklseq64',    # use sequential Intel MKL (ILP64)
                 'accelerate' ] # Accelerate framework on MacOS

LAPACKLIBS_HELP = { 'default'    : 'system default, e.g. {0}-llapack -lblas{1} (Linux) or {0}accelerate{1} (MacOS)',
                    'none'       : 'do not use BLAS/LAPACK',
                    'user'       : 'user defined BLAS/LAPACK (needs {0}lapackflags{1})',
                    'mkl'        : 'use MKL using 32 bit integer (default version (see also {0}mkl_dir{1})',
                    'mklomp'     : 'use MKL based on OpenMP using 32 bit integer',
                    'mkltbb'     : 'use MKL based on TBB using 32 bit integer',
                    'mklseq'     : 'use sequential MKL using 32 bit integer ({0}recommended{1})',
                    'mklomp64'   : 'use MKL based on OpenMP using 64bit integer (ILP64)',
                    'mkltbb64'   : 'use MKL based on TBB using 64bit integer (ILP64)',
                    'mklseq64'   : 'use sequential MKL  using 64bit integer (ILP64) ({0}recommended{1})',
                    'accelerate' : 'use Accelerate framework ({0}only MacOS{1})' }

# user defined linking flags for LAPACK
LAPACK_FLAGS = '-llapack -lblas'
                 
# malloc libraries (also depends on directories above)
MALLOCS      = [ 'help',        # print help
                 'default',
                 'system',
                 'jemalloc',
                 'mimalloc',
                 'tbbmalloc',
                 'tcmalloc' ]

MALLOCS_HELP = { 'default'   : 'default malloc, i.e., no overwrite',
                 'system'    : 'same as {0}default{1}',
                 'jemalloc'  : 'use jemalloc (see also {0}jemalloc_dir{1})',
                 'mimalloc'  : 'use mimalloc (see also {0}mimalloc_dir{1})',
                 'tbbmalloc' : 'use tbbmalloc',
                 'tcmalloc'  : 'use tcmalloc (see also {0}tcmalloc_dir{1})' }

# supported and active compressor
COMPRESSORS   = [ 'help',       # print help
                  'none',       # numeric ID:
                  'afl',        # 1
                  'aflp',       # 2
                  'sfl',        # 3
                  'dfl',        # 4
                  'dfl2',       # 22
                  'mp3',        # 18
                  'fp32',       # 13
                  # 'fp16',
                  # 'bf16',
                  # 'tf32',
                  # 'bf24',
                  'zfp',        # 5
                  'posits',     # 12
                  'cfloat',     # 20
                  'sz',         # 6
                  'sz3',        # 7
                  'mgard',      # 8
                  'blosc',      # 21
                  # 'lz4',
                  # 'zlib',
                  # 'zstd'
                 ]
compressor    = 'none'

COMPRESSORS_HELP = { 'none'   : 'no compression used',
                     'afl'    : 'use AFL',
                     'aflp'   : 'use AFLP',
                     'sfl'    : 'use SFL',
                     'dfl'    : 'use DFL',
                     'dfl2'   : 'use DFL2',
                     'mp3'    : 'use mixed precision storage with FP64/FP32/BF16',
                     'fp32'   : 'use FP32 for storage',
                     'zfp'    : 'use ZFP     (see also {0}zfp_dir{1})',
                     'posits' : 'use Posits  (see also {0}universal_dir{1})',
                     'cfloat' : 'use CFloats (see also {0}universal_dir{1})',
                     'sz'     : 'use SZ      (see also {0}sz_dir{1})',
                     'sz3'    : 'use SZ3     (see also {0}sz3_dir{1})',
                     'mgard'  : 'use MGARD   (see also {0}mgard_dir{1})',
                     'blosc'  : 'use Blosc   (see also {0}blosc_dir{1})',
                     # 'lz4'    : 'use LZ4     (see also {0}lz4_dir{1})',
                     # 'zlib'   : 'use Zlib    (see also {0}zlib_dir{1})',
                     # 'zstd'   : 'use Zstd    (see also {0}zstd_dir{1})',
                    }

# supported and active VALR compressor
VALR_COMPRESSORS = [ 'help',      # print help
                     'none',
                     'default',
                     'zfp',
                     'sz',
                     'sz3',
                     'mgard',
                     'afl',
                     'aflp',
                     'sfl',
                     'dfl',
                     'mp2',
                     'mp3',
                     'posits',
                     'cfloat',
                     'blosc' ]
valr = 'default'

VALR_HELP =        { 'none'    : 'no VALR compression used',
                     'default' : 'use VALR from {0}compressor{1} setting',
                     'zfp'     : 'use ZFP    (see also {0}zfp_dir{1})',
                     'posits'  : 'use Posits (see also {0}universal_dir{1})',
                     'cfloat'  : 'use CFloat (see also {0}universal_dir{1})',
                     'sz'      : 'use SZ     (see also {0}sz_dir{1})',
                     'sz3'     : 'use SZ3    (see also {0}sz3_dir{1})',
                     'mgard'   : 'use MGARD  (see also {0}mgard_dir{1})',
                     'blosc'   : 'use Blosc  (see also {0}blosc_dir{1})',
                     'afl'     : 'use AFL',
                     'aflp'    : 'use AFLP',
                     'sfl'     : 'use SFL',
                     'dfl'     : 'use DFL',
                     'mp2'     : 'use mixed precision with FP64/FP32',
                     'mp3'     : 'use mixed precision with FP64/FP32/BF16',
                    }


######################################################################
#
# helper functions
#
######################################################################

#
# return first line of output of given program
#
def readln ( prog ):
    text = ''

    try :
        file = os.popen( prog, 'r' )
        text = file.readline()
        file.close()
    except :
        pass

    return text

#
# compose actual path of program source
#
def path ( program, source ) :
    if SUBDIRS[ program ] != '' :
        return os.path.join( 'programs', SUBDIRS[ program ], source );
    else :
        return os.path.join( 'programs', source );
    
######################################################################
#
# preinitialization with known defaults
#
######################################################################

# # MKL should define MKLROOT
# if MKL_DIR == None and 'MKLROOT' in os.environ :
#     MKL_DIR = os.environ['MKLROOT']
# else :
#     MKL_DIR = '/' # to prevent error below due to invalid path

######################################################################
#
# collect all programs from "programs" sub directory
#
######################################################################

def scan_programs () :
    cc_file = re.compile( r'.*\.(cc|CC|cpp|c\+\+)\Z' )

    scanned_programs = []
    scanned_subdirs  = {}

    for root, dirs, files in os.walk( "programs", topdown = False ) :
        for filename in files :
            if cc_file.search( filename ) != None :
                # look for any framework
                for fwork in FRAMEWORKS :
                    fstr = '-' + fwork
                    pos  = filename.find( fstr )
                    if pos != -1 :
                        prog = filename[:pos]
                        if not prog in scanned_programs :
                            scanned_programs.append( prog )
                            if root == 'programs' : scanned_subdirs[prog] = ''
                            else :                  scanned_subdirs[prog] = root.replace( 'programs/', '' )

                        # print( root, filename[:pos], fwork )

    return scanned_programs, scanned_subdirs

tic = datetime.now()

PROGRAMS, SUBDIRS = scan_programs()

toc = datetime.now()

# print( "scanned programs in %.3es" % ( toc - tic ).total_seconds() )

######################################################################
#
# colorization
#
######################################################################

# default color codes
colors = { 'reset'  : '\033[0m',
           'bold'   : '\033[1m',
           'italic' : '\033[3m',
           'red'    : '\033[31m',
           'green'  : '\033[32m',
           'yellow' : '\033[33m',
           'blue'   : '\033[34m',
           'purple' : '\033[35m',
           'cyan'   : '\033[36m',
           'gray'   : '\033[37m' }

# no colors if wanted or output is not a terminal ('dumb' is for emacs)
if not color or not sys.stdout.isatty() or os.environ['TERM'] == 'dumb' :
    for key in colors.keys() :
        colors[key] = ''
else :
    # try to handle above codes on non-supported systems
    try:
        import colorama

        colorama.init()
    except :
        pass
        
######################################################################
#
# eval options
#
######################################################################

# set up command line parameters
opts = Variables( opts_file )

opts.Add( ListVariable( 'programs',      'programs to build',                 programs,   PROGRAMS   ) )
opts.Add( ListVariable( 'addprograms',   'add programs to build',             '',         PROGRAMS   ) )
opts.Add( ListVariable( 'remprograms',   'remove programs to build',          '',         PROGRAMS   ) )
opts.Add( ListVariable( 'frameworks',    'parallelization frameworks to use', frameworks, FRAMEWORKS ) )
opts.Add( ListVariable( 'addframeworks', 'add parallelization frameworks',    '',         FRAMEWORKS ) )
opts.Add( ListVariable( 'remframeworks', 'remove parallelization frameworks', '',         FRAMEWORKS ) )

opts.Add(               'cxx',       'C++ compiler to use',           CXX )
opts.Add(               'cxxflags',  'C++ compiler flags',            CXXFLAGS )
opts.Add(               'optflags',  'compiler optimization flags',   OPTFLAGS )
opts.Add(               'cpuflags',  'path to cpuflags',              CPUFLAGS )
opts.Add(               'defines',   'preprocessor defines',          DEFINES )

opts.Add( PathVariable( 'hpro_dir',      'base directory of hlibpro',     HPRO_DIR,     PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'tbb_dir',       'base directory of TBB',         TBB_DIR,      PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'tf_dir',        'base directory of C++TaskFlow', TASKFLOW_DIR, PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'hpx_dir',       'base directory of HPX',         HPX_DIR,      PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'gpi2_dir',      'base directory of GPI2',        GPI2_DIR,     PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'mkl_dir',       'base directory of MKL',         MKL_DIR,      PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'cuda_dir',      'base directory of CUDA',        CUDA_DIR,     PathVariable.PathIsDir ) )

opts.Add( PathVariable( 'jemalloc_dir',  'base directory of jemalloc',    JEMALLOC_DIR, PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'mimalloc_dir',  'base directory of mimalloc',    MIMALLOC_DIR, PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'tcmalloc_dir',  'base directory of tcmalloc',    TCMALLOC_DIR, PathVariable.PathIsDir ) )

opts.Add( EnumVariable( 'lapack',        'lapack library to use',              'default', allowed_values = LAPACKLIBS , ignorecase = 2 ) )
opts.Add(               'lapackflags',   'user defined link flags for lapack', default = LAPACK_FLAGS )
opts.Add( EnumVariable( 'malloc',        'malloc library to use',              'default', allowed_values = MALLOCS, ignorecase = 2 ) )
opts.Add( BoolVariable( 'eigen',         'use Eigen library',                  eigen ) )
opts.Add( PathVariable( 'eigen_dir',     'Eigen installation directory',       EIGEN_DIR, PathVariable.PathIsDir ) )
opts.Add( BoolVariable( 'hdf5',          'use HDF5 library',                   hdf5 ) )
opts.Add( PathVariable( 'hdf5_dir',      'HDF5 installation directory',        hdf5_dir, PathVariable.PathIsDir ) )
opts.Add( BoolVariable( 'likwid',        'use likwid library',                 likwid ) )
opts.Add( PathVariable( 'likwid_dir',    'likwid installation directory',      LIKWID_DIR, PathVariable.PathIsDir ) )
opts.Add( BoolVariable( 'scorep',        'use Score-P library',                scorep ) )
opts.Add( PathVariable( 'scorep_dir',    'Score-P installation directory',     SCOREP_DIR, PathVariable.PathIsDir ) )
opts.Add( BoolVariable( 'half',          'use half precision library',         half ) )
opts.Add( PathVariable( 'half_dir',      'half installation directory',        HALF_DIR, PathVariable.PathIsDir ) )
opts.Add( BoolVariable( 'zfp',           'use ZFP compression library',        zfp ) )
opts.Add( PathVariable( 'zfp_dir',       'ZFP installation directory',         ZFP_DIR, PathVariable.PathIsDir ) )
opts.Add( BoolVariable( 'sz',            'use SZ compression library',         sz ) )
opts.Add( PathVariable( 'sz_dir',        'SZ installation directory',          SZ_DIR, PathVariable.PathIsDir ) )
opts.Add( BoolVariable( 'sz3',           'use SZ3 compression library',        sz3 ) )
opts.Add( PathVariable( 'sz3_dir',       'SZ3 installation directory',         SZ3_DIR, PathVariable.PathIsDir ) )
opts.Add( BoolVariable( 'mgard',         'use MGARD compression library',      mgard ) )
opts.Add( PathVariable( 'mgard_dir',     'MGARD installation directory',       MGARD_DIR, PathVariable.PathIsDir ) )
# opts.Add( BoolVariable( 'lz4',           'use LZ4 compression library',        lz4 ) )
# opts.Add( PathVariable( 'lz4_dir',       'LZ4 installation directory',         LZ4_DIR, PathVariable.PathIsDir ) )
# opts.Add( BoolVariable( 'zlib',          'use ZLIB compression library',       zlib ) )
# opts.Add( PathVariable( 'zlib_dir',      'ZLIB installation directory',        ZLIB_DIR, PathVariable.PathIsDir ) )
# opts.Add( BoolVariable( 'zstd',          'use Zstd compression library',       zstd ) )
# opts.Add( PathVariable( 'zstd_dir',      'Zstd installation directory',        ZSTD_DIR, PathVariable.PathIsDir ) )
opts.Add( BoolVariable( 'universal',     'use universal number library',       universal ) )
opts.Add( PathVariable( 'universal_dir', 'universal installation directory',   UNIVERSAL_DIR, PathVariable.PathIsDir ) )
opts.Add( BoolVariable( 'blosc',         'use blosc compression library',      blosc ) )
opts.Add( PathVariable( 'blosc_dir',     'blosc installation directory',       BLOSC_DIR, PathVariable.PathIsDir ) )
opts.Add( BoolVariable( 'atc',           'use ATC compression library',        atc ) )
opts.Add( PathVariable( 'atc_dir',       'ATC installation directory',         ATC_DIR, PathVariable.PathIsDir ) )

opts.Add( EnumVariable( 'compressor',    'defined compressor',                  'none', allowed_values = COMPRESSORS,      ignorecase = 2 ) )
opts.Add( EnumVariable( 'valr',          'defined VALR compressor',             'none', allowed_values = VALR_COMPRESSORS, ignorecase = 2 ) )
opts.Add( BoolVariable( 'zblas',         'activate/deactivate compressed BLAS', zblas ) )

opts.Add( BoolVariable( 'fullmsg',   'enable full command line output',           fullmsg ) )
opts.Add( EnumVariable( 'buildtype', 'how to build the binaries (debug/release)', buildtype, allowed_values = BUILD_TYPES, ignorecase = 2 ) )
opts.Add( BoolVariable( 'warn',      'enable building with compiler warnings',    warn ) )
opts.Add( BoolVariable( 'color',     'use colored output during compilation',     color ) )

# read options from options file
opt_env = Environment( options = opts )

# apply modifiers
for opt in Split( opt_env['addprograms'] ) :
    if opt not in opt_env['programs'] :
        opt_env['programs'].append( opt )
for opt in Split( opt_env['remprograms'] ) :
    if opt in opt_env['programs'] :
        opt_env['programs'].remove( opt )
    
for opt in Split( opt_env['addframeworks'] ) :
    if opt not in opt_env['frameworks'] :
        opt_env['frameworks'].append( opt )
for opt in Split( opt_env['remframeworks'] ) :
    if opt in opt_env['frameworks'] :
        opt_env['frameworks'].remove( opt )

programs   = Split( opt_env['programs'] )
frameworks = Split( opt_env['frameworks'] )

if 'all' in programs   : programs   = PROGRAMS
if 'all' in frameworks : frameworks = FRAMEWORKS

CXX           = opt_env['cxx']
CXXFLAGS      = opt_env['cxxflags']
OPTFLAGS      = opt_env['optflags']
CPUFLAGS      = opt_env['cpuflags']
DEFINES       = opt_env['defines']

HPRO_DIR      = opt_env['hpro_dir']
TBB_DIR       = opt_env['tbb_dir']
TASKFLOW_DIR  = opt_env['tf_dir']
HPX_DIR       = opt_env['hpx_dir']
GPI2_DIR      = opt_env['gpi2_dir']

MKL_DIR       = opt_env['mkl_dir']
CUDA_DIR      = opt_env['cuda_dir']

JEMALLOC_DIR  = opt_env['jemalloc_dir']
MIMALLOC_DIR  = opt_env['mimalloc_dir']
TCMALLOC_DIR  = opt_env['tcmalloc_dir']
malloc        = opt_env['malloc']

lapack        = opt_env['lapack']
LAPACK_FLAGS  = opt_env['lapackflags']

eigen         = opt_env['eigen']
EIGEN_DIR     = opt_env['eigen_dir']
hdf5          = opt_env['hdf5']
HDF5_DIR      = opt_env['hdf5_dir']
likwid        = opt_env['likwid']
LIKWID_DIR    = opt_env['likwid_dir']
scorep        = opt_env['scorep']
SCOREP_DIR    = opt_env['scorep_dir']
half          = opt_env['half']
HALF_DIR      = opt_env['half_dir']
zfp           = opt_env['zfp']
ZFP_DIR       = opt_env['zfp_dir']
sz            = opt_env['sz']
SZ_DIR        = opt_env['sz_dir']
sz3           = opt_env['sz3']
SZ3_DIR       = opt_env['sz3_dir']
mgard         = opt_env['mgard']
MGARD_DIR     = opt_env['mgard_dir']
# lz4           = opt_env['lz4']
# LZ4_DIR       = opt_env['lz4_dir']
# zlib          = opt_env['zlib']
# ZLIB_DIR      = opt_env['zlib_dir']
# zstd          = opt_env['zstd']
# ZSTD_DIR      = opt_env['zstd_dir']
universal     = opt_env['universal']
UNIVERSAL_DIR = opt_env['universal_dir']
blosc         = opt_env['blosc']
BLOSC_DIR     = opt_env['blosc_dir']
atc           = opt_env['atc']
ATC_DIR       = opt_env['atc_dir']

compressor    = opt_env['compressor']
valr          = opt_env['valr']
zblas         = opt_env['zblas']

buildtype     = opt_env['buildtype']
fullmsg       = opt_env['fullmsg']
warn          = opt_env['warn']
color         = opt_env['color']

# remove entries to prevent saving
del opt_env['addprograms']
del opt_env['remprograms']
del opt_env['addframeworks']
del opt_env['remframeworks']

# handle 'help' requests to avoid saving
if 'help' in frameworks :
    print( "supported {0}framework{1} options: ".format( colors['bold'], colors['reset'] ) )
    for opt in FRAMEWORKS :
        if opt != 'help' :
            print( '  {0}{1:<11s}{2} : '.format( colors['bold'], opt, colors['reset'] ) + FRAMEWORK_HELP[opt].format( colors['italic'], colors['reset'] ) )
    sys.exit( 1 )
    
if lapack == 'help' :
    print( "supported {0}lapack{1} options: ".format( colors['bold'], colors['reset'] ) )
    for opt in LAPACKLIBS :
        if opt != 'help' :
            print( '  {0}{1:<11s}{2} : '.format( colors['bold'], opt, colors['reset'] ) + LAPACKLIBS_HELP[opt].format( colors['italic'], colors['reset'] ) )
    sys.exit( 1 )
    
if malloc == 'help' :
    print( 'supported {0}malloc{1} options: '.format( colors['bold'], colors['reset'] ) )
    for opt in MALLOCS :
        if opt != 'help' :
            print( '  {0}{1:<11s}{2} : '.format( colors['bold'], opt, colors['reset'] ) + MALLOCS_HELP[opt].format( colors['italic'], colors['reset'] ) )
    sys.exit( 1 )

if compressor == 'help' :
    print( 'supported {0}compressor{1} options: '.format( colors['bold'], colors['reset'] ) )
    for opt in COMPRESSORS :
        if opt != 'help' :
            print( '  {0}{1:<11s}{2} : '.format( colors['bold'], opt, colors['reset'] ) + COMPRESSORS_HELP[opt].format( colors['italic'], colors['reset'] ) )
    sys.exit( 1 )

if valr == 'help' :
    print( 'supported {0}valr{1} options: '.format( colors['bold'], colors['reset'] ) )
    for opt in VALR_COMPRESSORS :
        if opt != 'help' :
            print( '  {0}{1:<11s}{2} : '.format( colors['bold'], opt, colors['reset'] ) + VALR_HELP[opt].format( colors['italic'], colors['reset'] ) )
    sys.exit( 1 )

opts.Save( opts_file, opt_env )

######################################################################
#
# apply known defaults in case no user provided value is set
#
######################################################################

# MKL should define MKLROOT
if MKL_DIR == None or MKL_DIR == '/' :
    if 'MKLROOT' in os.environ :
        MKL_DIR = os.environ['MKLROOT']
    else :
        MKL_DIR = '/' # to prevent error below due to invalid path
    
# CUDA should define CUDA_ROOT or CUDA_HOME
if CUDA_DIR == None or CUDA_DIR == '/' :
    if 'CUDA_ROOT' in os.environ :
        CUDA_DIR = os.environ['CUDA_ROOT']
    elif 'CUDA_HOME' in os.environ :
        CUDA_DIR = os.environ['CUDA_HOME']
    else :
        CUDA_DIR = '/' # to prevent error below due to invalid path
    
######################################################################
#
# set up compilation environment
#
######################################################################

if buildtype in [ 'debug', 'dbg' ] :
    OPTFLAGS  = DBGFLAGS
    LINKFLAGS = '-g'
    DEFINES   = ''
elif buildtype in [ 'release-debug', 'reldbg' ] :
    OPTFLAGS  = RELDBGFLAGS
    LINKFLAGS = '-g'
    DEFINES   = ''
elif buildtype in [ 'release', 'rel' ] :
    OPTFLAGS  = RELFLAGS
    DEFINES   = DEFINES + ' NDEBUG'
    LINKFLAGS = ''

if warn :
    WARNFLAGS = readln( '%s --comp %s --warn' % ( CPUFLAGS, CXX ) )

# Thread Sanitizer
# CXXFLAGS  = CXXFLAGS  + ' -fsanitize=thread'
# LINKFLAGS = LINKFLAGS + ' -fsanitize=thread'

env = Environment( options    = opts, # TODO: <- check without
                   ENV        = os.environ,
                   CXX        = CXX,
                   CXXFLAGS   = Split( CXXFLAGS + ' ' + OPTFLAGS + ' ' + WARNFLAGS ),
                   LINKFLAGS  = Split( LINKFLAGS ),
                   CPPDEFINES = Split( DEFINES ) )

# include HLIBpro library
env.ParseConfig( os.path.join( HPRO_DIR, 'bin', 'hpro-config' ) + ' --cflags --lflags' )

# decative full compiler/linker output
if not fullmsg :
    env.Replace( CCCOMSTR     = ' %sCC%s     $SOURCES' % ( colors['green']  + colors['bold'], colors['reset'] )  )
    env.Replace( CXXCOMSTR    = ' %sC++%s    $SOURCES' % ( colors['green']  + colors['bold'], colors['reset'] ) )
    env.Replace( LINKCOMSTR   = ' %sLink%s   %s$TARGET%s'  % ( colors['cyan']   + colors['bold'], colors['reset'], colors['bold'], colors['reset'] ) )
    env.Replace( ARCOMSTR     = ' %sAR%s     %s$TARGET%s'  % ( colors['yellow'] + colors['bold'], colors['reset'], colors['bold'], colors['reset'] ) )
    env.Replace( RANLIBCOMSTR = ' %sIndex%s  %s$TARGET%s'  % ( colors['yellow'] + colors['bold'], colors['reset'], colors['bold'], colors['reset'] ) )

# add internal paths and libraries
env.Append(  CPPPATH = [ '#include' ] )
env.Append(  CPPPATH = [ '#programs/common' ] )
env.Prepend( LIBS    = [ 'hlr' ] )
env.Prepend( LIBPATH = [ '.' ] )

# add LAPACK library
if lapack == 'default' :
    env.Append( LIBS = [ 'lapack', 'blas' ] )
elif lapack == 'user' :
    flags = env.ParseFlags( LAPACK_FLAGS )
    env.MergeFlags( flags )
elif lapack in [ 'mkl', 'mkl64', 'mklomp', 'mklomp64' ] :
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include' ) )
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include', 'mkl' ) )
    env.Append( LIBPATH = os.path.join( MKL_DIR, 'lib', 'intel64_lin' ) ) # standard MKL
    env.Append( LIBPATH = os.path.join( MKL_DIR, 'lib', 'intel64' ) )     # oneMKL
    if lapack in [ 'mkl', 'mklomp' ] :
        env.Append( LIBS = [ 'mkl_gf_lp64' , 'mkl_gnu_thread', 'mkl_core', 'gomp' ] )
    else :
        env.Append( LIBS = [ 'mkl_gf_ilp64' , 'mkl_gnu_thread', 'mkl_core', 'gomp' ] )
elif lapack in [ 'mkltbb', 'mkltbb64' ] :
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include' ) )
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include', 'mkl' ) )
    env.Append( LIBPATH = os.path.join( MKL_DIR, 'lib', 'intel64_lin' ) ) # standard MKL
    env.Append( LIBPATH = os.path.join( MKL_DIR, 'lib', 'intel64' ) )     # oneMKL
    if lapack in [ 'mkl', 'mklomp' ] :
        env.Append( LIBS = [ 'mkl_gf_lp64' , 'mkl_tbb_thread', 'mkl_core', 'gomp' ] )
    else :
        env.Append( LIBS = [ 'mkl_gf_ilp64' , 'mkl_tbb_thread', 'mkl_core', 'gomp' ] )
elif lapack in [ 'mklseq', 'mklseq64' ] :
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include' ) )
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include', 'mkl' ) )
    env.Append( LIBPATH = os.path.join( MKL_DIR, 'lib', 'intel64_lin' ) ) # standard MKL
    env.Append( LIBPATH = os.path.join( MKL_DIR, 'lib', 'intel64' ) )     # oneMKL
    if lapack in [ 'mkl', 'mklomp' ] :
        env.Append( LIBS = [ 'mkl_gf_lp64' , 'mkl_sequential', 'mkl_core' ] )
    else :
        env.Append( LIBS = [ 'mkl_gf_ilp64' , 'mkl_sequential', 'mkl_core' ] )
elif lapack == 'accelerate' :
    env.MergeFlags( '-Wl,-framework,Accelerate' )

# include malloc library
if JEMALLOC_DIR != None and malloc == 'jemalloc' :
    env.MergeFlags( os.path.join( JEMALLOC_DIR, 'lib', 'libjemalloc.a' ) )
    env.Append( LIBS = [ 'dl', 'pthread' ] )
elif MIMALLOC_DIR != None and malloc == 'mimalloc' :
    env.MergeFlags( os.path.join( MIMALLOC_DIR, 'lib', 'libmimalloc.a' ) )
elif malloc == 'tbbmalloc' :
    env.Append( LIBPATH = os.path.join( TBB_DIR, 'lib' ) )
    env.Append( LIBS    = 'tbbmalloc' )
elif malloc == 'tcmalloc' :
    env.Append( LIBPATH = os.path.join( TCMALLOC_DIR, 'lib' ) )
    env.Append( LIBS    = 'tcmalloc' )

# include Eigen
if eigen and EIGEN_DIR != None :
    env.Append( CPPDEFINES = 'HLR_USE_EIGEN' )
    env.Append( CPPPATH    = os.path.join( EIGEN_DIR, 'include/eigen3' ) )

# include HDF5
if hdf5 and HDF5_DIR != None :
    env.Append( CPPDEFINES = 'HLR_USE_HDF5' )
    env.ParseConfig( 'PKG_CONFIG_PATH=%s pkg-config --cflags hdf5-serial' % os.path.join( HDF5_DIR, 'lib', 'pkgconfig' ) )
    env.ParseConfig( 'PKG_CONFIG_PATH=%s pkg-config --libs   hdf5-serial' % os.path.join( HDF5_DIR, 'lib', 'pkgconfig' ) )
    env.Append( LIBS = 'hdf5_cpp' )
    
# include likwid performance monitoring library
if likwid and LIKWID_DIR != None :
    env.Append( CPPDEFINES = 'HLR_USE_LIKWID' )
    env.Append( CPPPATH    = os.path.join( LIKWID_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( LIKWID_DIR, 'lib' ) )
    env.Append( LIBS       = 'likwid' )

# include Score-P tracing library
if scorep and SCOREP_DIR != None :
    env.Replace( CXX = os.path.join( SCOREP_DIR, 'bin', 'scorep' ) + ' --user --thread=pthread --mpp=none ' + CXX )
    env.Append( LIBPATH    = os.path.join( SCOREP_DIR, 'lib' ) )
    env.Append( CPPDEFINES = 'HLR_USE_SCOREP' )

# add CUDA
if 'cuda' in frameworks :
    env.Append( CPPPATH = os.path.join( CUDA_DIR, 'include' ) )
    env.Append( LIBPATH = os.path.join( CUDA_DIR, 'lib64' ) )
    env.Append( LIBS = [ 'cudart', 'cublasLt', 'cublas', 'cusolver' ] )

# support for half precision
if half :
    env.Append( CPPDEFINES = 'HLR_HAS_HALF' )
    env.Append( CPPPATH    = os.path.join( HALF_DIR, 'include' ) )

if zfp :
    env.Append( CPPDEFINES = 'HLR_HAS_ZFP' )
    env.Append( CPPPATH    = os.path.join( ZFP_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( ZFP_DIR, 'lib' ) )
    env.Append( LIBS       = [ 'zfp', 'gomp' ] ) # in case ZFP comes with OpenMP support
    
if   compressor == 'none' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=0' )
elif   compressor == 'afl' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=1' )
elif compressor == 'aflp' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=2' )
elif compressor == 'sfl' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=3' )
elif compressor == 'dfl' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=4' )
elif compressor == 'dfl2' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=22' )
elif compressor == 'mp3' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=18' )
elif compressor == 'zfp' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=5' )
    # env.Append( CPPDEFINES = 'HLR_HAS_ZFP' )
    # env.Append( CPPPATH    = os.path.join( ZFP_DIR, 'include' ) )
    # env.Append( LIBPATH    = os.path.join( ZFP_DIR, 'lib' ) )
    # env.Append( LIBS       = [ 'zfp' ] )
elif compressor == 'sz'   :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=6' )
    env.Append( CPPDEFINES = 'HLR_HAS_SZ' )
    env.Append( CPPPATH    = os.path.join( SZ_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( SZ_DIR, 'lib' ) )
    env.Append( LIBS       = [ 'SZ' ] )
elif compressor == 'sz3'  :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=7' )
    env.Append( CPPDEFINES = 'HLR_HAS_SZ3' )
    env.Append( CPPPATH    = os.path.join( SZ3_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( SZ3_DIR, 'lib' ) )
    env.Append( LIBS       = [ 'zstd' ] )
elif compressor == 'mgard' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=8' )
    env.Append( CPPDEFINES = 'HLR_HAS_MGARD' )
    env.Append( CPPPATH    = os.path.join( MGARD_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( MGARD_DIR, 'lib' ) )
    env.Append( LIBS       = [ 'mgard' ] )
# elif compressor == 'lz4' :
#     env.Append( CPPDEFINES = 'HLR_COMPRESSOR=9' )
#     env.Append( CPPDEFINES = 'HLR_HAS_LZ4' )
#     env.Append( CPPPATH    = os.path.join( LZ4_DIR, 'include' ) )
#     env.Append( LIBPATH    = os.path.join( LZ4_DIR, 'lib' ) )
#     env.Append( LIBS       = [ 'lz4' ] )
# elif compressor == 'zlib' :
#     env.Append( CPPDEFINES = 'HLR_COMPRESSOR=10' )
#     env.Append( CPPDEFINES = 'HLR_HAS_ZLIB' )
#     env.Append( CPPPATH    = os.path.join( ZLIB_DIR, 'include' ) )
#     env.Append( LIBPATH    = os.path.join( ZLIB_DIR, 'lib' ) )
#     env.Append( LIBS       = [ 'z' ] )
# elif compressor == 'zstd' :
#     env.Append( CPPDEFINES = 'HLR_COMPRESSOR=11' )
#     env.Append( CPPDEFINES = 'HLR_HAS_ZSTD' )
#     env.Append( CPPPATH    = os.path.join( ZSTD_DIR, 'include' ) )
#     env.Append( LIBPATH    = os.path.join( ZSTD_DIR, 'lib' ) )
#     env.Append( LIBS       = [ 'zstd' ] )
elif compressor == 'posits' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=12' )
    env.Append( CPPDEFINES = 'HLR_HAS_UNIVERSAL' )
    env.Append( CPPPATH    = os.path.join( UNIVERSAL_DIR, 'include' ) )
elif compressor == 'cfloat' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=20' )
    env.Append( CPPDEFINES = 'HLR_HAS_UNIVERSAL' )
    env.Append( CPPPATH    = os.path.join( UNIVERSAL_DIR, 'include' ) )
elif compressor == 'blosc' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=21' )
    env.Append( CPPDEFINES = 'HLR_HAS_BLOSC' )
    env.Append( CPPPATH    = os.path.join( BLOSC_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( BLOSC_DIR, 'lib' ) )
    env.Append( LIBS       = [ 'blosc2' ] )
elif compressor == 'fp32' :
    env.Append( CPPDEFINES = 'HLR_COMPRESSOR=13' )
# elif compressor == 'fp16' :
#     env.Append( CPPDEFINES = 'HLR_COMPRESSOR=14' )
# elif compressor == 'bf16' :
#     env.Append( CPPDEFINES = 'HLR_COMPRESSOR=15' )
# elif compressor == 'tf32' :
#     env.Append( CPPDEFINES = 'HLR_COMPRESSOR=16' )
# elif compressor == 'bf24' :
#     env.Append( CPPDEFINES = 'HLR_COMPRESSOR=17' )

if valr == 'default'  :
    if   compressor == 'none'   : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=0' )
    elif compressor == 'afl'    : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=1' )
    elif compressor == 'aflp'   : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=2' )
    elif compressor == 'sfl'    : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=3' )
    elif compressor == 'dfl'    : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=4' )
    elif compressor == 'dfl2'   : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=22' )
    elif compressor == 'mp3'    : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=18' )
    elif compressor == 'zfp'    : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=5' )
    elif compressor == 'sz'     : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=6' )
    elif compressor == 'sz3'    : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=7' )
    elif compressor == 'mgard'  : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=8' )
    elif compressor == 'posits' : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=12' )
    elif compressor == 'cfloat' : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=20' )
    elif compressor == 'blosc'  : env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=21' )
    else :
        env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=0' )
elif valr == 'none'  :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=0' )
elif valr == 'afl'  :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=1' )
elif valr == 'aflp' :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=2' )
elif valr == 'sfl'  :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=3' )
elif valr == 'dfl'  :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=4' )
elif valr == 'dfl2'  :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=22' )
elif valr == 'zfp'  :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=5' )
    env.Append( CPPDEFINES = 'HLR_HAS_ZFP' )
    env.Append( CPPPATH    = os.path.join( ZFP_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( ZFP_DIR, 'lib' ) )
    env.Append( LIBS       = [ 'zfp' ] )
elif valr == 'sz'      :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=6' )
    env.Append( CPPDEFINES = 'HLR_HAS_SZ' )
    env.Append( CPPPATH    = os.path.join( SZ_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( SZ_DIR, 'lib' ) )
    env.Append( LIBS       = [ 'SZ' ] )
elif valr == 'sz3'     :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=7' )
    env.Append( CPPDEFINES = 'HLR_HAS_SZ3' )
    env.Append( CPPPATH    = os.path.join( SZ3_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( SZ3_DIR, 'lib' ) )
    env.Append( LIBS       = [ 'zstd' ] )
elif valr == 'mgard'   :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=8' )
    env.Append( CPPDEFINES = 'HLR_HAS_MGARD' )
    env.Append( CPPPATH    = os.path.join( MGARD_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( MGARD_DIR, 'lib' ) )
    env.Append( LIBS       = [ 'mgard' ] )
elif valr == 'posits'   :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=12' )
    env.Append( CPPDEFINES = 'HLR_HAS_UNIVERSAL' )
    env.Append( CPPPATH    = os.path.join( UNIVERSAL_DIR, 'include' ) )
elif valr == 'cfloat'   :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=20' )
    env.Append( CPPDEFINES = 'HLR_HAS_UNIVERSAL' )
    env.Append( CPPPATH    = os.path.join( UNIVERSAL_DIR, 'include' ) )
elif valr == 'mp3'  :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=18' )
# elif valr == 'mp2'  :
#     env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=19' )
elif valr == 'blosc'   :
    env.Append( CPPDEFINES = 'HLR_VALR_COMPRESSOR=20' )
    env.Append( CPPDEFINES = 'HLR_HAS_BLOSC' )
    env.Append( CPPPATH    = os.path.join( BLOSC_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( BLOSC_DIR, 'lib' ) )
    env.Append( LIBS       = [ 'blosc2' ] )

if zblas :
    env.Append( CPPDEFINES = 'HLR_USE_ZBLAS=1' )
else :
    env.Append( CPPDEFINES = 'HLR_USE_ZBLAS=0' )

if atc :
    env.Append( CPPDEFINES = 'HLR_HAS_ATC' )
    env.Append( CPPPATH    = os.path.join( ATC_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( ATC_DIR, 'lib' ) )
    env.Append( LIBS       = [ 'atc', 'hptt' ] )

######################################################################
#
# target 'help'
#
######################################################################

#
# split array of strings for pretty printing in table
#
def split_str_array ( arr, n ) :
    parts = []
    line  = ''
    for i in range( len( arr ) ) :
        if i == len(arr)-1 : line = line + arr[i]
        else :               line = line + arr[i] + ', '

        if len( line ) > 40 :
            parts.append( line )
            line = ''

    if line != '' :
        parts.append( line )

    return parts

#
# helper for printing paths
#
def pathstr ( path ) :
    if path != '' : return '(' + path + ')'
    else          : return ''

#
# show output of "scons help"
#
def show_help ( target, source, env ):
    bool_str = { False : colors['bold'] + colors['red']   + '✘' + colors['reset'],
                 True  : colors['bold'] + colors['green'] + '✔'  + colors['reset'] }
    
    print() 
    print( 'Type  \'scons <option>=<value> ...\'  where <option> is one of' )
    print()
    print( '  {0}Option{1}       │ {0}Description{1}                   │ {0}Values{1}'.format( colors['bold'], colors['reset'] ) )
    print( ' ──────────────┼───────────────────────────────┼──────────' )

    parts = split_str_array( PROGRAMS, 40 )
    print( '  {0}programs{1}     │ programs to build             │'.format( colors['bold'], colors['reset'] ), parts[0] )
    for i in range( 1, len(parts) ) :
        print( '               │                               │', parts[i] )
    
    print( '  {0}frameworks{1}   │ software frameworks to use    │'.format( colors['bold'], colors['reset'] ), ', '.join( FRAMEWORKS ) )
    print( ' ──────────────┼───────────────────────────────┼──────────' )
    print( '  {0}hpro_dir{1}     │ base directory of HLIBpro     │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}tbb_dir{1}      │ base directory of TBB         │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}tf_dir{1}       │ base directory of C++TaskFlow │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}hpx_dir{1}      │ base directory of HPX         │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}gpi2_dir{1}     │ base directory of GPI2        │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}cuda_dir{1}     │ base directory of CUDA        │'.format( colors['bold'], colors['reset'] ) )
    print( ' ──────────────┼───────────────────────────────┼──────────' )
    print( '  {0}lapack{1}       │ BLAS/LAPACK library to use    │'.format( colors['bold'], colors['reset'] ), ', '.join( LAPACKLIBS ) )
    print( '  {0}lapackflags{1}  │ user provided link flags      │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}mkl_dir{1}      │ base directory of MKL         │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}hdf5{1}         │ use HDF5 library              │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}hdf5_dir{1}     │ path to HDF5 library          │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}likwid{1}       │ use LikWid library            │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}likwid_dir{1}   │ path to LikWid library        │'.format( colors['bold'], colors['reset'] ) )
    print( ' ──────────────┼───────────────────────────────┼──────────' )
    print( '  {0}malloc{1}       │ malloc library to use         │'.format( colors['bold'], colors['reset'] ), ', '.join( MALLOCS ) )
    print( '  {0}jemalloc{1}     │ use jemalloc                  │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}jemalloc_dir{1} │ path to jemalloc              │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}mimalloc{1}     │ use mimalloc                  │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}mimalloc_dir{1} │ path to mimalloc              │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}tcmalloc{1}     │ use tcmalloc                  │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}tcmalloc_dir{1} │ path to tcmalloc              │'.format( colors['bold'], colors['reset'] ) )
    print( ' ──────────────┼───────────────────────────────┼──────────' )
    print( '  {0}compressor{1}   │ compression method to use     │ {2}'.format( colors['bold'], colors['reset'], ', '.join( COMPRESSORS ) ) )
    print( '  {0}valr{1}         │ AP compression method to use  │ {2}'.format( colors['bold'], colors['reset'], ', '.join( VALR_COMPRESSORS ) ) )
    print( '  {0}zblas{1}        │ use compressed BLAS           │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( ' ──────────────┼───────────────────────────────┼──────────' )
    print( '  {0}zfp{1}          │ use ZFP library               │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}zfp_dir{1}      │ path to ZFP library           │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}sz{1}           │ use SZ library                │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}sz_dir{1}       │ path to SZ library            │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}sz3{1}          │ use SZ3 library               │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}sz3_dir{1}      │ path to SZ3 library           │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}universal{1}    │ use Universal number library  │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}universal_dir{1}│ path to Universal library     │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}mgard{1}        │ use MGARD library             │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}mgard_dir{1}    │ path to MGARD library         │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}half{1}         │ use half number library       │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}half_dir{1}     │ path to half library          │'.format( colors['bold'], colors['reset'] ) )
    # print( '  {0}lz4{1}          │ use LZ4 library               │'.format( colors['bold'], colors['reset'] ), '0/1' )
    # print( '  {0}lz4_dir{1}      │ path to LZ4 library           │'.format( colors['bold'], colors['reset'] ) )
    # print( '  {0}zlib{1}         │ use zlib library              │'.format( colors['bold'], colors['reset'] ), '0/1' )
    # print( '  {0}zlib_dir{1}     │ path to zlib library          │'.format( colors['bold'], colors['reset'] ) )
    # print( '  {0}zstd{1}         │ use Zstd library              │'.format( colors['bold'], colors['reset'] ), '0/1' )
    # print( '  {0}zstd_dir{1}     │ path to Zstd library          │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}blosc{1}        │ use Blosc2 library            │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}blosc_dir{1}    │ path to Blosc2 library        │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}atc{1}          │ use ATC library               │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}atc_dir{1}      │ path to ATC library           │'.format( colors['bold'], colors['reset'] ) )
    print( ' ──────────────┼───────────────────────────────┼──────────' )
    print( '  {0}buildtype{1}    │ how to build the binaries     │'.format( colors['bold'], colors['reset'] ), ', '.join( BUILD_TYPES ) )
    print( '  {0}warn{1}         │ enable compiler warnings      │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}fullmsg{1}      │ full command line output      │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}color{1}        │ use colored output            │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print() 
    print( 'The parameters {0}programs{1} and {0}frameworks{1} can get comma separated values:'.format( colors['bold'], colors['reset'] ) ) 
    print() 
    print( '    scons {0}programs{2}={1}dag-lu,dag-inv{2} {0}frameworks{2}={1}seq,tbb,omp{2}'.format( colors['bold'], colors['italic'], colors['reset'] ) ) 
    print() 
    print( 'For {0}malloc{1} only a single value is valid:'.format( colors['bold'], colors['reset'] ) )
    print() 
    print( '    scons {0}malloc{2}={1}jemalloc{2}'.format( colors['bold'], colors['italic'], colors['reset'] ) ) 
    print() 

help_cmd = env.Command( 'phony-target-help', None, show_help )

env.Alias( 'help', help_cmd )

######################################################################
#
# target 'options'
#
######################################################################

def show_options ( target, source, env ):
    bool_str = { False : colors['bold'] + colors['red']   + '✘' + colors['reset'],
                 True  : colors['bold'] + colors['green'] + '✔' + colors['reset'] }
    
    print() 
    print( 'Type  \'scons <option>=<value> ...\'  where <option> is one of' )
    print()
    print( '  {0}Option{1}       │ {0}Description{1}                   │ {0}Value/Directory{1}'.format( colors['bold'], colors['reset'] ) )
    print( ' ──────────────┼───────────────────────────────┼─────────────────' )
    print( '  {0}cxx{1}          │ C++ compiler                  │'.format( colors['bold'], colors['reset'] ), CXX )
    print( '  {0}cxxflags{1}     │ C++ compiler flags            │'.format( colors['bold'], colors['reset'] ), CXXFLAGS )
    print( '  {0}optflags{1}     │ compiler optimization flags   │'.format( colors['bold'], colors['reset'] ), OPTFLAGS )
    print( ' ──────────────┼───────────────────────────────┼──────────' )

    # split "programs" into smaller pieces
    parts = split_str_array( programs, 40 )
    print( '  {0}programs{1}     │ programs to build             │'.format( colors['bold'], colors['reset'] ), parts[0] )
    for i in range( 1, len(parts) ) :
        print( '               │                               │', parts[i] )
        
    print( '  {0}frameworks{1}   │ software frameworks to use    │'.format( colors['bold'], colors['reset'] ), ', '.join( frameworks ) )
    print( ' ──────────────┼───────────────────────────────┼──────────' )
    print( '  {0}hpro_dir{1}     │ base directory of HLIBpro     │'.format( colors['bold'], colors['reset'] ), HPRO_DIR )
    print( '  {0}tbb_dir{1}      │ base directory of TBB         │'.format( colors['bold'], colors['reset'] ), TBB_DIR )
    print( '  {0}tf_dir{1}       │ base directory of C++TaskFlow │'.format( colors['bold'], colors['reset'] ), TASKFLOW_DIR )
    print( '  {0}hpx_dir{1}      │ base directory of HPX         │'.format( colors['bold'], colors['reset'] ), HPX_DIR )
    print( '  {0}gpi2_dir{1}     │ base directory of GPI2        │'.format( colors['bold'], colors['reset'] ), GPI2_DIR )
    print( '  {0}cuda_dir{1}     │ base directory of CUDA        │'.format( colors['bold'], colors['reset'] ), CUDA_DIR )
    print( ' ──────────────┼───────────────────────────────┼──────────' )
    print( '  {0}lapack{1}       │ BLAS/LAPACK library to use    │'.format( colors['bold'], colors['reset'] ), lapack )
    if lapack == 'user' :
        print( '  {0}lapackflags{1}  │ user provided link flags      │ {2}'.format( colors['bold'], colors['reset'], LAPACK_FLAGS ) )
    print( '  {0}malloc{1}       │ malloc library to use         │ {2}'.format( colors['bold'], colors['reset'], malloc ),
           pathstr( JEMALLOC_DIR if malloc == 'jemalloc' else MIMALLOC_DIR if malloc == 'mimalloc' else TCMALLOC_DIR if malloc == 'tcmalloc' else '' ) )
    print( '  {0}mkl_dir{1}      │ base directory of Intel MKL   │'.format( colors['bold'], colors['reset'] ), MKL_DIR )
    print( '  {0}hdf5{1}         │ use HDF5 library              │ {2}'.format( colors['bold'], colors['reset'], bool_str[ hdf5 ] ),       pathstr( HDF5_DIR      if hdf5      else '' ) )
    print( '  {0}likwid{1}       │ use LikWid library            │ {2}'.format( colors['bold'], colors['reset'], bool_str[ likwid ] ),     pathstr( LIKWID_DIR    if likwid    else '' ) )
    print( ' ──────────────┼───────────────────────────────┼──────────' )
    print( '  {0}compressor{1}   │ compression method to use     │ {2}'.format( colors['bold'], colors['reset'], compressor ) )
    print( '  {0}valr{1}         │ AP compression method to use  │ {2}'.format( colors['bold'], colors['reset'], valr ) )
    print( '  {0}zblas{1}        │ use compressed BLAS           │ {2}'.format( colors['bold'], colors['reset'], bool_str[ zblas ] ) )
    print( ' ──────────────┼───────────────────────────────┼──────────' )
    print( '  {0}zfp{1}          │ use ZFP compression library   │ {2}'.format( colors['bold'], colors['reset'], bool_str[ zfp ] ),        pathstr( ZFP_DIR       if zfp       else '' ) )
    print( '  {0}sz{1}           │ use SZ compression library    │ {2}'.format( colors['bold'], colors['reset'], bool_str[ sz ] ),         pathstr( SZ_DIR        if sz        else '' ) )
    print( '  {0}sz3{1}          │ use SZ3 compression library   │ {2}'.format( colors['bold'], colors['reset'], bool_str[ sz3 ] ),        pathstr( SZ3_DIR       if sz3       else '' ) )
    print( '  {0}mgard{1}        │ use MGARD compression library │ {2}'.format( colors['bold'], colors['reset'], bool_str[ mgard ] ),      pathstr( MGARD_DIR     if mgard     else '' ) )
    print( '  {0}universal{1}    │ use Universal number library  │ {2}'.format( colors['bold'], colors['reset'], bool_str[ universal ] ),  pathstr( UNIVERSAL_DIR if universal else '' ) )
    print( '  {0}half{1}         │ use half number library       │ {2}'.format( colors['bold'], colors['reset'], bool_str[ half ] ),       pathstr( HALF_DIR      if half      else '' ) )
    # print( '  {0}lz4{1}          │ use LZ4 library               │ {2}'.format( colors['bold'], colors['reset'], bool_str[ lz4 ] ),        pathstr( LZ4_DIR       if lz4       else '' ) )
    # print( '  {0}zlib{1}         │ use zlib library              │ {2}'.format( colors['bold'], colors['reset'], bool_str[ zlib ] ),       pathstr( ZLIB_DIR      if zlib      else '' ) )
    # print( '  {0}zstd{1}         │ use Zstd library              │ {2}'.format( colors['bold'], colors['reset'], bool_str[ zstd ] ),       pathstr( ZSTD_DIR      if zstd      else '' ) )
    print( '  {0}blosc{1}        │ use BLOSC2 library            │ {2}'.format( colors['bold'], colors['reset'], bool_str[ blosc ] ),      pathstr( BLOSC_DIR     if blosc     else '' ) )
    print( '  {0}atc{1}          │ use ATC compression library   │ {2}'.format( colors['bold'], colors['reset'], bool_str[ atc ] ),        pathstr( ATC_DIR       if atc       else '' ) )
    print( ' ──────────────┼───────────────────────────────┼──────────' )
    print( '  {0}buildtype{1}    │ how to build the binaries     │'.format( colors['bold'], colors['reset'] ), buildtype )
    print( '  {0}warn{1}         │ enable compiler warnings      │'.format( colors['bold'], colors['reset'] ), bool_str[ warn ] )
    print( '  {0}fullmsg{1}      │ full command line output      │'.format( colors['bold'], colors['reset'] ), bool_str[ fullmsg ] )
    print( '  {0}color{1}        │ use colored output            │'.format( colors['bold'], colors['reset'] ), bool_str[ color ] )
    print() 

options_cmd = env.Command( 'phony-target-options', None, show_options )

env.Alias( 'options', options_cmd )

######################################################################
#
# HLR library and framework dependent targets
#
######################################################################

sources = [ 'src/apps/exp.cc',
            'src/apps/helmholtz.cc',
            'src/apps/laplace.cc',
            'src/apps/log_kernel.cc',
            'src/apps/radial.cc',
            'src/cluster/distr.cc',
            'src/cluster/h.cc',
            'src/cluster/hodlr.cc',
            'src/cluster/mblr.cc',
            'src/cluster/sfc.cc',
            'src/cluster/tileh.cc',
            'src/cluster/tlr.cc',
            'src/dag/graph.cc',
            'src/dag/local_graph.cc',
            'src/dag/node.cc',
            # 'src/dag/solve.cc',
            # 'src/matrix/level_matrix.cc',
            'src/matrix/print.cc',
            'src/seq/dag.cc',
            # 'src/seq/solve.cc',
            'src/utils/compare.cc',
            'src/utils/eps_printer.cc',
            'src/utils/log.cc',
            'src/utils/mach.cc',
            'src/utils/term.cc',
            'src/utils/text.cc' ]

# add when needed
if 'dag-lu' in programs :
    sources += [ 'src/dag/gauss_elim.cc',
                 'src/dag/invert.cc',
                 'src/dag/lu.cc',
                 'src/dag/lu_coarse.cc',
                 'src/dag/lu_hodlr_tiled.cc',
                 'src/dag/lu_hodlr_tiled_lazy.cc',
                 'src/dag/lu_lvl.cc',
                 'src/dag/lu_oop.cc',
                 'src/dag/lu_oop_accu.cc',
                 'src/dag/lu_oop_accu_sep.cc',
                 'src/dag/lu_oop_auto.cc',
                 'src/dag/lu_tileh.cc' ]

libhlr = env.StaticLibrary( 'hlr', sources )

Default( None )

program_list = []

#
# default sequential environment
#

if 'seq' in frameworks :
    seq = env.Clone()
        
    for program in programs :
        name   = program + '-seq'
        source = path( program, name + '.cc' )

        if os.path.exists( source ) and os.path.isfile( source ) :
            program_list.append( seq.Program( path( program, name ), [ source ] ) )

#
# OpenMP
#

if 'omp' in frameworks :
    omp = env.Clone()
    omp.Append( CXXFLAGS  = '-fopenmp' )
    omp.Append( LINKFLAGS = '-fopenmp' )

    for program in programs :
        name   = program + '-omp'
        source = path( program, name + '.cc' )

        if os.path.exists( source ) and os.path.isfile( source ) :
            program_list.append( omp.Program( path( program, name ), [ source, 'src/omp/dag.cc' ] ) )

#
# TBB
#

if 'tbb' in frameworks :
    tbb = env.Clone()
    # tbb.ParseConfig( 'PKG_CONFIG_PATH=%s pkg-config --cflags tbb' % os.path.join( TBB_DIR, 'lib', 'pkgconfig' ) )
    # tbb.ParseConfig( 'PKG_CONFIG_PATH=%s pkg-config --libs   tbb' % os.path.join( TBB_DIR, 'lib', 'pkgconfig' ) )
    tbb.Append( CPPPATH = os.path.join( TBB_DIR, 'include' ) )
    tbb.Append( LIBPATH = os.path.join( TBB_DIR, 'lib' ) )
    tbb.Append( LIBS    = [ 'tbb' ] )

    for program in programs :
        name   = program + '-tbb'
        source = path( program, name + '.cc' )

        if os.path.exists( source ) and os.path.isfile( source ) :
            program_list.append( tbb.Program( path( program, name ), [ source, 'src/tbb/dag.cc' ] ) )

#
# TaskFlow
#

if 'tf' in frameworks :
    tf = env.Clone()
    tf.MergeFlags( '-isystem ' + os.path.join( TASKFLOW_DIR, 'include' ) )
    tf.Append( LIBS = [ 'pthread' ] )
    # tf.ParseConfig( 'PKG_CONFIG_PATH=/opt/local/magma-2.5.3/lib/pkgconfig pkg-config --cflags magma' )
    # tf.ParseConfig( 'PKG_CONFIG_PATH=/opt/local/magma-2.5.3/lib/pkgconfig pkg-config --libs   magma' )
    
    for program in programs :
        name   = program + '-tf'
        source = path( program, name + '.cc' )

        # special case TF+CUDA
        if program == 'cuda' and not 'cuda' in frameworks :
            continue
        
        if os.path.exists( source ) and os.path.isfile( source ) :
            program_list.append( tf.Program( path( program, name ), [ source, 'src/tf/dag.cc' ] ) )
            
    # program_list.append( tf.Program( 'programs/magma', [ 'programs/magma.cc' ] ) )

#
# HPX
#

if 'hpx' in frameworks :
    hpx = env.Clone()
    hpx.ParseConfig( 'PKG_CONFIG_PATH=%s pkg-config --cflags hpx_application' % ( os.path.join( HPX_DIR, 'lib', 'pkgconfig' ) ) )
    hpx.ParseConfig( 'PKG_CONFIG_PATH=%s pkg-config --libs   hpx_application' % ( os.path.join( HPX_DIR, 'lib', 'pkgconfig' ) ) )
    hpx.Append( LIBS = [ 'hpx_iostreams' ] )
    
    for program in programs :
        name   = program + '-hpx'
        source = path( program, name + '.cc' )

        if os.path.exists( source ) and os.path.isfile( source ) :
            program_list.append( hpx.Program( path( program, name ), [ source, 'src/hpx/dag.cc' ] ) )

#
# MPI
#

if 'mpi' in frameworks :
    mpi = env.Clone()
    mpi.ParseConfig( 'mpic++ --showme:compile' )
    mpi.ParseConfig( 'mpic++ --showme:link' )
    
    if 'tlr'   in programs :
        program_list.append( mpi.Program( path( 'tlr', 'tlr-mpi-bcast.cc'  ) ) )
        program_list.append( mpi.Program( path( 'tlr', 'tlr-mpi-ibcast.cc' ) ) )
        program_list.append( mpi.Program( path( 'tlr', 'tlr-mpi-rdma.cc'   ) ) )
    
    if 'tileh' in programs and 'tbb' in frameworks :
        program_list.append( mpi.Program( path( 'tileh', 'tileh-mpi-bcast'  ), [ path( 'tileh', 'tileh-mpi-bcast.cc'  ), 'src/tbb/dag.o' ] ) )
        program_list.append( mpi.Program( path( 'tileh', 'tileh-mpi-ibcast' ), [ path( 'tileh', 'tileh-mpi-ibcast.cc' ), 'src/tbb/dag.o' ] ) )

#
# GASPI
#

if 'gpi2' in frameworks :
    gpi = env.Clone()
    gpi.ParseConfig( 'PKG_CONFIG_PATH=%s pkg-config --cflags GPI2' % ( os.path.join( GPI2_DIR, 'lib64', 'pkgconfig' ) ) )
    gpi.ParseConfig( 'PKG_CONFIG_PATH=%s pkg-config --libs   GPI2' % ( os.path.join( GPI2_DIR, 'lib64', 'pkgconfig' ) ) )
    gpi.Append( LIBS = [ 'pthread' ] )
    
    if 'tlr' in programs : program_list.append( gpi.Program( path( 'tlr', 'tlr-gaspi.cc' ) ) )

#
# finally set programs to be built
#

Default( program_list )

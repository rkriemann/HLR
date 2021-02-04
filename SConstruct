# -*- coding: utf-8 -*-

# to enable print() syntax with python2
from __future__ import print_function

import os, sys

######################################################################
#
# general (default) settings
#
######################################################################

fullmsg      = False
debug        = False
profile      = False
optimise     = True
warn         = False
color        = True

# cache file storing SCons settings
opts_file    = '.scons.options'

CXX          = 'g++'
CXXFLAGS     = '-std=c++17'
CPUFLAGS     = 'cpuflags'

OPTFLAGS     = '-O3 -march=native'
WARNFLAGS    = '-Wall'
LINKFLAGS    = ''
DEFINES      = 'TBB_PREVIEW_GLOBAL_CONTROL __TBB_show_deprecation_message_task_H'

# directories for the various external libraries
HPRO_DIR     = '/'
MKL_DIR      = '/'
TBB_DIR      = '/'
TASKFLOW_DIR = '/'
HPX_DIR      = '/'
GPI2_DIR     = '/'

JEMALLOC_DIR = '/'
MIMALLOC_DIR = '/'
TCMALLOC_DIR = '/'

LIKWID_DIR   = '/opt/local/likwid'
likwid       = False

# set of programs to build: dag-*, tlr, hodlr, tileh (or 'all')
PROGRAMS     = [ 'tlr',
                 'hodlr',
                 'tileh',
                 'tiled-h',
                 'tiled-hca',
                 'tiled-hodlr',
                 'dag-lu',
                 'dag-gauss',
                 'dag-inv',
                 'dag-waz',
                 'dag-hodlr',
                 'uniform',
                 'uniform-mm',
                 'uniform-lu',
                 'approx-mm',
                 'approx-lu',
                 'dpt',
                 'polykern' ]

# set of frameworks to use: seq, openmp, tbb, tf, hpx, mpi, gpi2 (or 'all')
FRAMEWORKS   = [ 'seq',
                 'omp',
                 'tbb',
                 'tf',
                 'hpx',
                 'mpi',
                 'gpi2' ]

# supported lapack libraries
LAPACKLIBS   = [ 'default',     # default system implementation, e.g., -llapack -lblas
                 'none',        # do not use any LAPACK library
                 'mkl',         # use parallel Intel MKL (should be OpenMP version)
                 'mklomp',      # use OpenMP based Intel MKL
                 'mkltbb',      # use TBB based Intel MKL
                 'mklseq',      # use sequential Intel MKL
                 'accelerate' ] # Accelerate framework on MacOS
                 
# malloc libraries (also depends on directories above)
MALLOCS      = [ 'default',
                 'system',
                 'jemalloc',
                 'mimalloc',
                 'tbbmalloc',
                 'tcmalloc' ]

# mapping of programs to subdirs
SUBDIRS      = { 'tlr'         : 'tlr',
                 'hodlr'       : 'hodlr',
                 'tileh'       : 'tileh',
                 'tiled-h'     : 'tiled',
                 'tiled-hca'   : 'tiled',
                 'tiled-hodlr' : 'tiled',
                 'dag-lu'      : 'dag',
                 'dag-gauss'   : 'dag',
                 'dag-inv'     : 'dag',
                 'dag-waz'     : 'dag',
                 'dag-hodlr'   : 'dag',
                 'uniform'     : 'uniform',
                 'uniform-mm'  : 'uniform',
                 'uniform-lu'  : 'uniform',
                 'approx-mm'   : 'approx',
                 'approx-lu'   : 'approx',
                 'dpt'         : '',
                 'polykern'    : '' }

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
# eval options
#
######################################################################

# set up command line parameters
opts = Variables( opts_file )
opts.Add( ListVariable( 'programs',      'programs to build',                 'all', PROGRAMS   ) )
opts.Add( ListVariable( 'addprograms',   'programs to build',                 '',    PROGRAMS   ) )
opts.Add( ListVariable( 'subprograms',   'programs to build',                 '',    PROGRAMS   ) )
opts.Add( ListVariable( 'frameworks',    'parallelization frameworks to use', 'all', FRAMEWORKS ) )
opts.Add( ListVariable( 'addframeworks', 'add parallelization frameworks',    '',    FRAMEWORKS ) )
opts.Add( ListVariable( 'subframeworks', 'remove parallelization frameworks', '',    FRAMEWORKS ) )

opts.Add(               'cxx',      'C++ compiler to use',           CXX )
opts.Add(               'cxxflags', 'C++ compiler flags',            CXXFLAGS )
opts.Add(               'cpuflags', 'path to cpuflags',              CPUFLAGS )
opts.Add(               'defines',  'preprocessor defines',          DEFINES )

opts.Add( PathVariable( 'hpro',     'base directory of hlibpro',     HPRO_DIR,     PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'tbb',      'base directory of TBB',         TBB_DIR,      PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'tf',       'base directory of C++TaskFlow', TASKFLOW_DIR, PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'hpx',      'base directory of HPX',         HPX_DIR,      PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'gpi2',     'base directory of GPI2',        GPI2_DIR,     PathVariable.PathIsDir ) )

opts.Add( PathVariable( 'mkl',      'base directory of MKL',         MKL_DIR,      PathVariable.PathIsDir ) )

opts.Add( PathVariable( 'jemalloc', 'base directory of jemalloc',    JEMALLOC_DIR, PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'mimalloc', 'base directory of mimalloc',    MIMALLOC_DIR, PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'tcmalloc', 'base directory of tcmalloc',    TCMALLOC_DIR, PathVariable.PathIsDir ) )

opts.Add( EnumVariable( 'lapack',   'lapack library to use',         'default', allowed_values = LAPACKLIBS, ignorecase = 2 ) )
opts.Add( EnumVariable( 'malloc',   'malloc library to use',         'default', allowed_values = MALLOCS, ignorecase = 2 ) )
opts.Add( BoolVariable( 'likwid',   'use likwid library',            likwid ) )

opts.Add( BoolVariable( 'fullmsg',  'enable full command line output',           fullmsg ) )
opts.Add( BoolVariable( 'debug',    'enable building with debug informations',   debug ) )
opts.Add( BoolVariable( 'profile',  'enable building with profile informations', profile ) )
opts.Add( BoolVariable( 'optimise', 'enable building with optimisation',         optimise ) )
opts.Add( BoolVariable( 'warn',     'enable building with compiler warnings',    warn ) )
opts.Add( BoolVariable( 'color',    'use colored output during compilation',     color ) )

# read options from options file
opt_env = Environment( options = opts )

# apply modifiers
for opt in Split( opt_env['addprograms'] ) :
    if opt not in opt_env['programs'] :
        opt_env['programs'].append( opt )
for opt in Split( opt_env['subprograms'] ) :
    if opt in opt_env['programs'] :
        opt_env['programs'].remove( opt )
    
for opt in Split( opt_env['addframeworks'] ) :
    if opt not in opt_env['frameworks'] :
        opt_env['frameworks'].append( opt )
for opt in Split( opt_env['subframeworks'] ) :
    if opt in opt_env['frameworks'] :
        opt_env['frameworks'].remove( opt )

programs   = Split( opt_env['programs'] )
frameworks = Split( opt_env['frameworks'] )

if 'all' in programs   : programs   = PROGRAMS
if 'all' in frameworks : frameworks = FRAMEWORKS

CXX          = opt_env['cxx']
CXXFLAGS     = opt_env['cxxflags']
CPUFLAGS     = opt_env['cpuflags']
DEFINES      = opt_env['defines']

HPRO_DIR     = opt_env['hpro']
TBB_DIR      = opt_env['tbb']
TASKFLOW_DIR = opt_env['tf']
HPX_DIR      = opt_env['hpx']
GPI2_DIR     = opt_env['gpi2']

MKL_DIR      = opt_env['mkl']

JEMALLOC_DIR = opt_env['jemalloc']
MIMALLOC_DIR = opt_env['mimalloc']
TCMALLOC_DIR = opt_env['tcmalloc']

lapack       = opt_env['lapack']
malloc       = opt_env['malloc']
likwid       = opt_env['likwid']

fullmsg      = opt_env['fullmsg']
debug        = opt_env['debug']
profile      = opt_env['profile']
optimise     = opt_env['optimise']
warn         = opt_env['warn']
color        = opt_env['color']

# remove entries to prevent saving
del opt_env['addprograms']
del opt_env['subprograms']
del opt_env['addframeworks']
del opt_env['subframeworks']

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
    
######################################################################
#
# colorization
#
######################################################################

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
      
######################################################################
#
# set up compilation environment
#
######################################################################

if debug :
    OPTFLAGS  = '-g -march=native'
    LINKFLAGS = '-g'
    DEFINES   = ''

if profile :
    OPTFLAGS  = '-g -pg -O3 -march=native'
    LINKFLAGS = '-g -pg'
    DEFINES   = ''

if warn :
    WARNFLAGS = readln( '%s --comp %s --warn' % ( CPUFLAGS, CXX ) )
    
env = Environment( options    = opts, # TODO: <- check without
                   ENV        = os.environ,
                   CXX        = CXX,
                   CXXFLAGS   = Split( CXXFLAGS + ' ' + OPTFLAGS + ' ' + WARNFLAGS ),
                   LINKFLAGS  = Split( LINKFLAGS ),
                   CPPDEFINES = Split( DEFINES ) )

# include HLIBpro library
env.ParseConfig( os.path.join( HPRO_DIR, 'bin', 'hlib-config' ) + ' --cflags --lflags' )

# decative full compiler/linker output
if not fullmsg :
    env.Replace( CCCOMSTR     = ' %sCC%s     $SOURCES' % ( colors['green']  + colors['bold'], colors['reset'] )  )
    env.Replace( CXXCOMSTR    = ' %sC++%s    $SOURCES' % ( colors['green']  + colors['bold'], colors['reset'] ) )
    env.Replace( LINKCOMSTR   = ' %sLink%s   %s$TARGET%s'  % ( colors['cyan']   + colors['bold'], colors['reset'], colors['bold'], colors['reset'] ) )
    env.Replace( ARCOMSTR     = ' %sAR%s     %s$TARGET%s'  % ( colors['yellow'] + colors['bold'], colors['reset'], colors['bold'], colors['reset'] ) )
    env.Replace( RANLIBCOMSTR = ' %sIndex%s  %s$TARGET%s'  % ( colors['yellow'] + colors['bold'], colors['reset'], colors['bold'], colors['reset'] ) )

# ensure NDEBUG is set in optimization mode
if not debug :
    env.Append(  CPPDEFINES = [ 'NDEBUG' ] )

# add internal paths and libraries
env.Append(  CPPPATH = [ '#include' ] )
env.Append(  CPPPATH = [ '#programs/common' ] )
env.Prepend( LIBS    = [ 'hlr' ] )
env.Prepend( LIBPATH = [ '.' ] )

# add LAPACK library
if lapack == 'default' :
    env.Append( LIBS = [ 'lapack', 'blas' ] )
elif lapack == 'mkl' or lapack == 'mklomp' :
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include' ) )
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include', 'mkl' ) )
    env.Append( LIBPATH = os.path.join( MKL_DIR, 'lib', 'intel64_lin' ) )
    env.Append( LIBS = [ 'mkl_gf_lp64' , 'mkl_gnu_thread', 'mkl_core', 'gomp' ] )
elif lapack == 'mkltbb' :
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include' ) )
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include', 'mkl' ) )
    env.Append( LIBPATH = os.path.join( MKL_DIR, 'lib', 'intel64_lin' ) )
    env.Append( LIBS = [ 'mkl_gf_lp64' , 'mkl_tbb_thread', 'mkl_core', 'gomp' ] )
elif lapack == 'mklseq' :
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include' ) )
    env.Append( CPPPATH = os.path.join( MKL_DIR, 'include', 'mkl' ) )
    env.Append( LIBPATH = os.path.join( MKL_DIR, 'lib', 'intel64_lin' ) )
    env.Append( LIBS = [ 'mkl_gf_lp64' , 'mkl_sequential', 'mkl_core' ] )
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

# include likwid performance monitoring library
if likwid and LIKWID_DIR != None :
    env.Append( CPPDEFINES = 'LIKWID_PERFMON' )
    env.Append( CPPPATH    = os.path.join( LIKWID_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( LIKWID_DIR, 'lib' ) )
    env.Append( LIBS       = 'likwid' )

######################################################################
#
# target 'help'
#
######################################################################

def show_help ( target, source, env ):
    bool_str = { False : colors['bold'] + colors['red']   + '✘' + colors['reset'],
                 True  : colors['bold'] + colors['green'] + '✔'  + colors['reset'] }
    
    print() 
    print( 'Type  \'scons <option>=<value> ...\'  where <option> is one of' )
    print()
    print( '  {0}Option{1}     │ {0}Description{1}                   │ {0}Values{1}'.format( colors['bold'], colors['reset'] ) )
    print( ' ────────────┼───────────────────────────────┼──────────' )
    print( '  {0}programs{1}   │ programs to build             │'.format( colors['bold'], colors['reset'] ), ', '.join( PROGRAMS ) )
    print( '  {0}frameworks{1} │ software frameworks to use    │'.format( colors['bold'], colors['reset'] ), ', '.join( FRAMEWORKS ) )
    print( ' ────────────┼───────────────────────────────┼──────────' )
    print( '  {0}malloc{1}     │ malloc library to use         │'.format( colors['bold'], colors['reset'] ), ', '.join( MALLOCS ) )
    print( '  {0}likwid{1}     │ use LikWid library            │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( ' ────────────┼───────────────────────────────┼──────────' )
    print( '  {0}optimise{1}   │ enable compiler optimisations │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}debug{1}      │ enable debug information      │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}profile{1}    │ enable profile information    │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}warn{1}       │ enable compiler warnings      │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}fullmsg{1}    │ full command line output      │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( '  {0}color{1}      │ use colored output            │'.format( colors['bold'], colors['reset'] ), '0/1' )
    print( ' ────────────┼───────────────────────────────┼──────────' )
    print( '  {0}hpro{1}       │ base directory of HLIBpro     │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}tbb{1}        │ base directory of TBB         │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}tf{1}         │ base directory of C++TaskFlow │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}hpx{1}        │ base directory of HPX         │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}gpi2{1}       │ base directory of GPI2        │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}jemalloc{1}   │ base directory of jemalloc    │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}mimalloc{1}   │ base directory of mimalloc    │'.format( colors['bold'], colors['reset'] ) )
    print( '  {0}tcmalloc{1}   │ base directory of tcmalloc    │'.format( colors['bold'], colors['reset'] ) )
    print() 
    print( 'The parameters {0}programs{1} and {0}frameworks{1} can get comma separated values:'.format( colors['bold'], colors['reset'] ) ) 
    print() 
    print( '    scons {0}programs{2}={1}dag-lu,dag-inv{2} {0}frameworks{2}={1}seq,tbb,omp{2}'.format( colors['bold'], colors['italic'], colors['reset'] ) ) 
    print() 
    print( 'For {0}malloc{1} only a single value is valid:'.format( colors['bold'], colors['reset'] ) )
    print() 
    print( '    scons {0}malloc{2}={1}jemalloc{2}'.format( colors['bold'], colors['italic'], colors['reset'] ) ) 
    print() 
    print( 'Paths for all software frameworks may also be adjusted in the {0}SConstruct{1} file.'.format( colors['bold'], colors['reset'] ) ) 
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
    print( '  {0}Option{1}     │ {0}Description{1}                   │ {0}Value{1}'.format( colors['bold'], colors['reset'] ) )
    print( ' ────────────┼───────────────────────────────┼──────────' )
    print( '  {0}programs{1}   │ programs to build             │'.format( colors['bold'], colors['reset'] ), ', '.join( programs ) )
    print( '  {0}frameworks{1} │ software frameworks to use    │'.format( colors['bold'], colors['reset'] ), ', '.join( frameworks ) )
    print( ' ────────────┼───────────────────────────────┼──────────' )
    print( '  {0}malloc{1}     │ malloc library to use         │'.format( colors['bold'], colors['reset'] ), malloc )
    print( '  {0}likwid{1}     │ use LikWid library            │'.format( colors['bold'], colors['reset'] ), bool_str[ likwid ] )
    print( ' ────────────┼───────────────────────────────┼──────────' )
    print( '  {0}optimise{1}   │ enable compiler optimisations │'.format( colors['bold'], colors['reset'] ), bool_str[ optimise ] )
    print( '  {0}debug{1}      │ enable debug information      │'.format( colors['bold'], colors['reset'] ), bool_str[ debug ] )
    print( '  {0}profile{1}    │ enable profile information    │'.format( colors['bold'], colors['reset'] ), bool_str[ profile ] )
    print( '  {0}warn{1}       │ enable compiler warnings      │'.format( colors['bold'], colors['reset'] ), bool_str[ warn ] )
    print( '  {0}fullmsg{1}    │ full command line output      │'.format( colors['bold'], colors['reset'] ), bool_str[ fullmsg ] )
    print( '  {0}color{1}      │ use colored output            │'.format( colors['bold'], colors['reset'] ), bool_str[ color ] )
    print( ' ────────────┼───────────────────────────────┼──────────' )
    print( '  {0}hpro{1}       │ base directory of HLIBpro     │'.format( colors['bold'], colors['reset'] ), HPRO_DIR )
    print( '  {0}tbb{1}        │ base directory of TBB         │'.format( colors['bold'], colors['reset'] ), TBB_DIR )
    print( '  {0}tf{1}         │ base directory of C++TaskFlow │'.format( colors['bold'], colors['reset'] ), TASKFLOW_DIR )
    print( '  {0}hpx{1}        │ base directory of HPX         │'.format( colors['bold'], colors['reset'] ), HPX_DIR )
    print( '  {0}gpi2{1}       │ base directory of GPI2        │'.format( colors['bold'], colors['reset'] ), GPI2_DIR )
    print( '  {0}jemalloc{1}   │ base directory of jemalloc    │'.format( colors['bold'], colors['reset'] ), JEMALLOC_DIR )
    print( '  {0}mimalloc{1}   │ base directory of mimalloc    │'.format( colors['bold'], colors['reset'] ), MIMALLOC_DIR )
    print( '  {0}tcmalloc{1}   │ base directory of tcmalloc    │'.format( colors['bold'], colors['reset'] ), TCMALLOC_DIR )
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
            'src/apps/matern_cov.cc',
            'src/cluster/distr.cc',
            'src/cluster/h.cc',
            'src/cluster/hodlr.cc',
            'src/cluster/mblr.cc',
            'src/cluster/tileh.cc',
            'src/cluster/tlr.cc',
            'src/dag/gauss_elim.cc',
            'src/dag/graph.cc',
            'src/dag/invert.cc',
            'src/dag/local_graph.cc',
            'src/dag/lu.cc',
            'src/dag/lu_coarse.cc',
            'src/dag/lu_hodlr_tiled.cc',
            'src/dag/lu_hodlr_tiled_lazy.cc',
            'src/dag/lu_lvl.cc',
            'src/dag/lu_oop.cc',
            'src/dag/lu_oop_accu.cc',
            'src/dag/lu_oop_accu_sep.cc',
            'src/dag/lu_oop_auto.cc',
            'src/dag/lu_tileh.cc',
            'src/dag/node.cc',
            'src/dag/solve.cc',
            'src/matrix/level_matrix.cc',
            'src/matrix/luinv_eval.cc',
            'src/matrix/print.cc',
            'src/seq/dag.cc',
            'src/seq/solve.cc',
            'src/utils/compare.cc',
            'src/utils/eps_printer.cc',
            'src/utils/log.cc',
            'src/utils/mach.cc',
            'src/utils/term.cc',
            'src/utils/text.cc' ]

libhlr = env.StaticLibrary( 'hlr', sources )

Default( None )

#
# default sequential environment
#

if 'seq' in frameworks :
    seq = env.Clone()
        
    for program in programs :
        name   = program + '-seq'
        source = path( program, name + '.cc' )

        if os.path.exists( source ) and os.path.isfile( source ) :
            Default( seq.Program( path( program, name ), [ source ] ) )

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
            Default( omp.Program( path( program, name ), [ source, 'src/omp/dag.cc' ] ) )

#
# TBB
#

if 'tbb' in frameworks :
    tbb = env.Clone()
    tbb.Append( CPPPATH = os.path.join( TBB_DIR, 'include' ) )
    tbb.Append( LIBPATH = os.path.join( TBB_DIR, 'lib' ) )

    for program in programs :
        name   = program + '-tbb'
        source = path( program, name + '.cc' )

        if os.path.exists( source ) and os.path.isfile( source ) :
            Default( tbb.Program( path( program, name ), [ source, 'src/tbb/dag.cc' ] ) )

#
# TaskFlow
#

if 'tf' in frameworks :
    tf = env.Clone()
    tf.MergeFlags( '-isystem ' + os.path.join( TASKFLOW_DIR, 'include' ) )
    tf.Append( LIBS = [ 'pthread' ] )
    # tf.ParseConfig( 'PKG_CONFIG_PATH=/opt/local/magma-2.5.3/lib/pkgconfig pkg-config --cflags magma' )
    # tf.ParseConfig( 'PKG_CONFIG_PATH=/opt/local/magma-2.5.3/lib/pkgconfig pkg-config --libs   magma' )
    # tf.Append( LIBS = [ 'cudart', 'cublas', 'cusolver' ] )
    
    for program in programs :
        name   = program + '-tf'
        source = path( program, name + '.cc' )

        if os.path.exists( source ) and os.path.isfile( source ) :
            Default( tf.Program( path( program, name ), [ source, 'src/tf/dag.cc' ] ) )
            
    # Default( tf.Program( 'programs/magma', [ 'programs/magma.cc' ] ) )
    # Default( tf.Program( 'programs/cuda',  [ 'programs/cuda.cc'  ] ) )

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
            Default( hpx.Program( path( program, name ), [ source, 'src/hpx/dag.cc' ] ) )

#
# MPI
#

if 'mpi' in frameworks :
    mpi = env.Clone()
    mpi.ParseConfig( 'mpic++ --showme:compile' )
    mpi.ParseConfig( 'mpic++ --showme:link' )
    
    if 'tlr'   in programs :
        Default( mpi.Program( path( 'tlr', 'tlr-mpi-bcast.cc'  ) ) )
        Default( mpi.Program( path( 'tlr', 'tlr-mpi-ibcast.cc' ) ) )
        Default( mpi.Program( path( 'tlr', 'tlr-mpi-rdma.cc'   ) ) )
    
    if 'tileh' in programs and 'tbb' in frameworks :
        Default( mpi.Program( path( 'tileh', 'tileh-mpi-bcast'  ), [ path( 'tileh', 'tileh-mpi-bcast.cc'  ), 'src/tbb/dag.o' ] ) )
        Default( mpi.Program( path( 'tileh', 'tileh-mpi-ibcast' ), [ path( 'tileh', 'tileh-mpi-ibcast.cc' ), 'src/tbb/dag.o' ] ) )

#
# GASPI
#

if 'gpi2' in frameworks :
    gpi = env.Clone()
    gpi.ParseConfig( 'PKG_CONFIG_PATH=%s pkg-config --cflags GPI2' % ( os.path.join( GPI2_DIR, 'lib64', 'pkgconfig' ) ) )
    gpi.ParseConfig( 'PKG_CONFIG_PATH=%s pkg-config --libs   GPI2' % ( os.path.join( GPI2_DIR, 'lib64', 'pkgconfig' ) ) )
    gpi.Append( LIBS = [ 'pthread' ] )
    
    if 'tlr' in programs : Default( gpi.Program( path( 'tlr', 'tlr-gaspi.cc' ) ) )

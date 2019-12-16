
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
CXXFLAGS     = '-std=c++2a'

OPTFLAGS     = '-O3 -march=native'
WARNFLAGS    = '-Wall'
LINKFLAGS    = ''
DEFINES      = 'BOOST_SYSTEM_NO_DEPRECATED'

# directories for the various external libraries
HPRO_DIR     = '/home/rok/programming/hpro/devel'
TBB_DIR      = '/usr'
TASKFLOW_DIR = '/opt/local/cpp-taskflow/2.2.0'
HPX_DIR      = '/opt/local/hpx'
GPI2_DIR     = '/opt/local/gpi2'

JEMALLOC_DIR = '/opt/local/jemalloc/5.2.1'
MIMALLOC_DIR = '/opt/local/mimalloc'
TCMALLOC_DIR = '/usr'

LIKWID_DIR   = '/opt/local/likwid'
likwid       = False

# set of programs to build: dag-*, tlr, hodlr, tileh (or "all")
PROGRAMS     = [ 'tlr', 'hodlr', 'tileh', 'dag-lu', 'dag-gauss', 'dag-inv', 'tiled-hodlr', 'dag-hodlr' ]

# set of frameworks to use: seq, openmp, tbb, tf, hpx, mpi, gpi2 (or "all")
FRAMEWORKS   = [ 'seq', 'omp', 'tbb', 'tf', 'hpx', 'mpi', 'gpi2' ]

# malloc libraries (also depends on directories above)
MALLOCS      = [ 'default', 'system', 'jemalloc', 'mimalloc', 'tbbmalloc', 'tcmalloc' ]

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

######################################################################
#
# eval options
#
######################################################################

opts = Variables( opts_file )
opts.Add( ListVariable( 'programs',   'programs to build',                 'all',     PROGRAMS   ) )
opts.Add( ListVariable( 'frameworks', 'parallelization frameworks to use', 'all',     FRAMEWORKS ) )

opts.Add( PathVariable( 'hpro',     'base directory of hlibpro',     HPRO_DIR,     PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'tbb',      'base directory of TBB',         TBB_DIR,      PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'tf',       'base directory of C++TaskFlow', TASKFLOW_DIR, PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'hpx',      'base directory of HPX',         HPX_DIR,      PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'gpi2',     'base directory of GPI2',        GPI2_DIR,     PathVariable.PathIsDir ) )

opts.Add( PathVariable( 'jemalloc', 'base directory of jemalloc',    JEMALLOC_DIR, PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'mimalloc', 'base directory of mimalloc',    MIMALLOC_DIR, PathVariable.PathIsDir ) )
opts.Add( PathVariable( 'tcmalloc', 'base directory of tcmalloc',    TCMALLOC_DIR, PathVariable.PathIsDir ) )

opts.Add( EnumVariable( 'malloc',   'malloc library to use',         'default', allowed_values = MALLOCS, ignorecase = 2 ) )
opts.Add( BoolVariable( 'likwid',   'use likwid library',            likwid ) )

opts.Add( BoolVariable( 'fullmsg',  'enable full command line output',           fullmsg ) )
opts.Add( BoolVariable( 'debug',    'enable building with debug informations',   debug ) )
opts.Add( BoolVariable( 'profile',  'enable building with profile informations', profile ) )
opts.Add( BoolVariable( 'optimise', 'enable building with optimisation',         optimise ) )
opts.Add( BoolVariable( 'warn',     'enable building with compiler warnings',    warn ) )
opts.Add( BoolVariable( 'color',    'use colored output during compilation',     color ) )

# read options from options file
opt_env    = Environment( options = opts )

programs   = Split( opt_env['programs'] )
frameworks = Split( opt_env['frameworks'] )

if 'all' in programs   : programs   = PROGRAMS
if 'all' in frameworks : frameworks = FRAMEWORKS

HPRO_DIR     = opt_env['hpro']
TBB_DIR      = opt_env['tbb']
TASKFLOW_DIR = opt_env['tf']
HPX_DIR      = opt_env['hpx']
GPI2_DIR     = opt_env['gpi2']

JEMALLOC_DIR = opt_env['jemalloc']
MIMALLOC_DIR = opt_env['mimalloc']
TCMALLOC_DIR = opt_env['tcmalloc']

malloc       = opt_env['malloc']
likwid       = opt_env['likwid']

fullmsg      = opt_env['fullmsg']
debug        = opt_env['debug']
profile      = opt_env['profile']
optimise     = opt_env['optimise']
warn         = opt_env['warn']
color        = opt_env['color']

opts.Save( opts_file, opt_env )

######################################################################
#
# colorization
#
######################################################################

colors = { 'reset'  : '\033[0m',
           'bold'   : '\033[1m',
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
    WARNFLAGS = readln( 'cpuflags --comp %s --warn' % CXX )
    
env = Environment( options    = opts,
                   ENV        = os.environ,
                   CXX        = CXX,
                   CXXFLAGS   = Split( CXXFLAGS + ' ' + OPTFLAGS + ' ' + WARNFLAGS ),
                   LINKFLAGS  = Split( LINKFLAGS ),
                   CPPDEFINES = Split( DEFINES ),
                   )

# include HLIBpro library
env.ParseConfig( os.path.join( HPRO_DIR, 'bin', 'hlib-config' ) + ' --cflags --lflags' )

# decative full compiler/linker output
if not fullmsg :
    env.Replace( CCCOMSTR     = " %sCC%s     $SOURCES" % ( colors['green']  + colors['bold'], colors['reset'] )  )
    env.Replace( CXXCOMSTR    = " %sC++%s    $SOURCES" % ( colors['green']  + colors['bold'], colors['reset'] ) )
    env.Replace( LINKCOMSTR   = " %sLink%s   %s$TARGET%s"  % ( colors['cyan']   + colors['bold'], colors['reset'], colors['bold'], colors['reset'] ) )
    env.Replace( ARCOMSTR     = " %sAR%s     %s$TARGET%s"  % ( colors['yellow'] + colors['bold'], colors['reset'], colors['bold'], colors['reset'] ) )
    env.Replace( RANLIBCOMSTR = " %sIndex%s  %s$TARGET%s"  % ( colors['yellow'] + colors['bold'], colors['reset'], colors['bold'], colors['reset'] ) )

# ensure NDEBUG is set in optimization mode
if not debug :
    env.Append(  CPPDEFINES = [ "NDEBUG" ] )

# add internal paths and libraries
env.Append(  CPPPATH = [ '#include' ] )
env.Prepend( LIBS    = [ "hlr" ] )
env.Prepend( LIBPATH = [ "." ] )

# include malloc library
if JEMALLOC_DIR != None and malloc == 'jemalloc' :
    env.MergeFlags( os.path.join( JEMALLOC_DIR, 'lib', 'libjemalloc.a' ) )
    env.Append( LIBS = [ 'dl', 'pthread' ] )
elif MIMALLOC_DIR != None and malloc == 'mimalloc' :
    env.MergeFlags( os.path.join( MIMALLOC_DIR, 'lib', 'libmimalloc.a' ) )
elif malloc == 'tbbmalloc' :
    env.Append( LIBPATH = os.path.join( TBB_DIR, "lib" ) )
    env.Append( LIBS    = 'tbbmalloc' )
elif malloc == 'tcmalloc' :
    env.Append( LIBPATH = os.path.join( TCMALLOC_DIR, "lib" ) )
    env.Append( LIBS    = 'tcmalloc' )

# include likwid performance monitoring library
if likwid and LIKWID_DIR != None :
    env.Append( CPPDEFINES = 'LIKWID_PERFMON' )
    env.Append( CPPPATH    = os.path.join( LIKWID_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( LIKWID_DIR, 'lib' ) )
    env.Append( LIBS       = 'likwid' )

######################################################################
#
# target "help"
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
    print( '  {0}programs{1}   │ programs to build             │'.format( colors['bold'], colors['reset'] ), PROGRAMS )
    print( '  {0}frameworks{1} │ software frameworks to use    │'.format( colors['bold'], colors['reset'] ), FRAMEWORKS )
    print( ' ────────────┼───────────────────────────────┼──────────' )
    print( '  {0}malloc{1}     │ malloc library to use         │'.format( colors['bold'], colors['reset'] ), MALLOCS )
    print( '  {0}likwid{1}     │ use LikWid library            │'.format( colors['bold'], colors['reset'] ), "0/1" )
    print( ' ────────────┼───────────────────────────────┼──────────' )
    print( '  {0}optimise{1}   │ enable compiler optimisations │'.format( colors['bold'], colors['reset'] ), "0/1" )
    print( '  {0}debug{1}      │ enable debug information      │'.format( colors['bold'], colors['reset'] ), "0/1" )
    print( '  {0}profile{1}    │ enable profile information    │'.format( colors['bold'], colors['reset'] ), "0/1" )
    print( '  {0}warn{1}       │ enable compiler warnings      │'.format( colors['bold'], colors['reset'] ), "0/1" )
    print( '  {0}fullmsg{1}    │ full command line output      │'.format( colors['bold'], colors['reset'] ), "0/1" )
    print( '  {0}color{1}      │ use colored output            │'.format( colors['bold'], colors['reset'] ), "0/1" )
    print() 
    print( 'The parameters {0}programs{1} and {0}frameworks{1} can get comma separated values:'.format( colors['bold'], colors['reset'] ) ) 
    print() 
    print( '    scons programs=dag-lu,dag-inv frameworks=seq,tbb,omp' ) 
    print() 
    print( 'For {0}malloc{1} only a single value is valid:'.format( colors['bold'], colors['reset'] ) )
    print() 
    print( '    scons malloc=jemalloc' ) 
    print() 
    print( 'Don\'t forget to adjust paths for all software frameworks in the file {0}SConstruct{1}.'.format( colors['bold'], colors['reset'] ) ) 
    print() 

help_cmd = env.Command( 'phony-target-help', None, show_help )

env.Alias( 'help', help_cmd )

######################################################################
#
# target "options"
#
######################################################################

def show_options ( target, source, env ):
    bool_str = { False : colors['bold'] + colors['red']   + '✘' + colors['reset'],
                 True  : colors['bold'] + colors['green'] + '✔'  + colors['reset'] }
    
    print() 
    print( 'Type  \'scons <option>=<value> ...\'  where <option> is one of' )
    print()
    print( '  {0}Option{1}     │ {0}Value{1}                   │ {0}Description{1}'.format( colors['bold'], colors['reset'] ) )
    print( ' ────────────┼─────────────────────────┼──────────────────────────' )
    print( '  {0}programs{1}   │ {2:<23} │ programs to build'.format( colors['bold'], colors['reset'], opt_env['programs'] ) )
    print( '  {0}frameworks{1} │ {2:<23} │ software frameworks to use'.format( colors['bold'], colors['reset'], opt_env['frameworks'] ) )
    print( ' ────────────┼─────────────────────────┼──────────────────────────' )
    print( '  {0}malloc{1}     │ {2:<23} │ malloc library to use'.format( colors['bold'], colors['reset'], opt_env['malloc'] ) )
    print( '  {0}likwid{1}     │ {2}                       │ use LikWid library'.format( colors['bold'], colors['reset'], bool_str[ opt_env['likwid'] ] ) )
    print( ' ────────────┼─────────────────────────┼──────────────────────────' )
    print( '  {0}optimise{1}   │ {2}                       │ enable compiler optimisations'.format( colors['bold'], colors['reset'], bool_str[ opt_env['optimise'] ] ) )
    print( '  {0}debug{1}      │ {2}                       │ enable debug information'.format( colors['bold'], colors['reset'], bool_str[ opt_env['debug'] ] ) )
    print( '  {0}profile{1}    │ {2}                       │ enable profile information'.format( colors['bold'], colors['reset'], bool_str[ opt_env['profile'] ] ) )
    print( '  {0}warn{1}       │ {2}                       │ enable compiler warnings'.format( colors['bold'], colors['reset'], bool_str[ opt_env['warn'] ] ) )
    print( '  {0}fullmsg{1}    │ {2}                       │ full command line output'.format( colors['bold'], colors['reset'], bool_str[ opt_env['fullmsg'] ] ) )
    print( '  {0}color{1}      │ {2}                       │ use colored output'.format( colors['bold'], colors['reset'], bool_str[ opt_env['color'] ] ) )
    print() 

options_cmd = env.Command( 'phony-target-options', None, show_options )

env.Alias( 'options', options_cmd )

######################################################################
#
# HLR library and framework dependent targets
#
######################################################################

libhlr = env.StaticLibrary( 'hlr', [ 'src/apps/laplace.cc',
                                     'src/apps/log_kernel.cc',
                                     'src/apps/matern_cov.cc',
                                     'src/cluster/distr.cc',
                                     'src/cluster/h.cc',
                                     'src/cluster/hodlr.cc',
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
                                     'src/dag/node.cc',
                                     'src/dag/solve.cc',
                                     'src/matrix/level_matrix.cc',
                                     'src/matrix/luinv_eval.cc',
                                     'src/seq/dag.cc',
                                     'src/seq/solve.cc',
                                     'src/utils/compare.cc',
                                     'src/utils/log.cc',
                                     'src/utils/term.cc',
                                     'src/utils/text.cc' ] )

Default( None )

#
# default sequential environment
#

if 'seq' in frameworks :
    seq = env.Clone()
        
    for program in programs :
        name   = program + '-seq'
        source = name + '.cc'

        if os.path.exists( source ) and os.path.isfile( source ) :
            Default( seq.Program( name, [ source ] ) )

#
# OpenMP
#

if 'omp' in frameworks :
    omp = env.Clone()
    omp.Append( CXXFLAGS  = "-fopenmp" )
    omp.Append( LINKFLAGS = "-fopenmp" )

    for program in programs :
        name   = program + '-omp'
        source = name + '.cc'

        if os.path.exists( source ) and os.path.isfile( source ) :
            Default( omp.Program( name, [ source, 'src/omp/dag.cc' ] ) )

#
# TBB
#

if 'tbb' in frameworks :
    tbb = env.Clone()
    tbb.Append( CPPPATH = os.path.join( TBB_DIR, "include" ) )
    tbb.Append( LIBPATH = os.path.join( TBB_DIR, "lib" ) )

    for program in programs :
        name   = program + '-tbb'
        source = name + '.cc'

        if os.path.exists( source ) and os.path.isfile( source ) :
            Default( tbb.Program( name, [ source, 'src/tbb/dag.cc' ] ) )

#
# TaskFlow
#

if 'tf' in frameworks :
    tf = env.Clone()
    tf.MergeFlags( '-isystem ' + os.path.join( TASKFLOW_DIR, "include" ) )
    tf.Append( LIBS = [ "pthread" ] )
    
    for program in programs :
        name   = program + '-tf'
        source = name + '.cc'

        if os.path.exists( source ) and os.path.isfile( source ) :
            Default( tf.Program( name, [ source, 'src/tf/dag.cc' ] ) )

#
# HPX
#

if 'hpx' in frameworks :
    hpx = env.Clone()
    hpx.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --cflags hpx_application" % ( os.path.join( HPX_DIR, 'lib', 'pkgconfig' ) ) )
    hpx.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --libs   hpx_application" % ( os.path.join( HPX_DIR, 'lib', 'pkgconfig' ) ) )
    hpx.Append( LIBS = [ "hpx_iostreams" ] )
    
    for program in programs :
        name   = program + '-hpx'
        source = name + '.cc'

        if os.path.exists( source ) and os.path.isfile( source ) :
            Default( hpx.Program( name, [ source, 'src/hpx/dag.cc' ] ) )

#
# MPI
#

if 'mpi' in frameworks :
    mpi = env.Clone()
    mpi.ParseConfig( 'mpic++ --showme:compile' )
    mpi.ParseConfig( 'mpic++ --showme:link' )
    
    if 'tlr'   in programs :
        Default( mpi.Program( 'tlr-mpi-bcast.cc' ) )
        Default( mpi.Program( 'tlr-mpi-ibcast.cc' ) )
        Default( mpi.Program( 'tlr-mpi-rdma.cc' ) )
    
    if 'tileh' in programs :
        Default( mpi.Program( 'tileh-mpi-bcast',  [ 'tileh-mpi-bcast.cc',  'src/tbb/dag.cc' ] ) )
        Default( mpi.Program( 'tileh-mpi-ibcast', [ 'tileh-mpi-ibcast.cc', 'src/tbb/dag.cc' ] ) )

#
# GASPI
#

if 'gpi2' in frameworks :
    gpi = env.Clone()
    gpi.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --cflags GPI2" % ( os.path.join( GPI2_DIR, 'lib64', 'pkgconfig' ) ) )
    gpi.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --libs   GPI2" % ( os.path.join( GPI2_DIR, 'lib64', 'pkgconfig' ) ) )
    gpi.Append( LIBS = [ "pthread" ] )
    
    if 'tlr'   in programs : Default( gpi.Program( 'tlr-gaspi.cc' ) )

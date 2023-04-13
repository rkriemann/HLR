#!/bin/bash
#
# CPU benchmarks using various HLR programs (mm, lu, etc.)
#

######################################################################
##
## common settings
##
######################################################################

# set of BLAS implementations to use
BLAS_IMPL="mkl openblas blis refblas"

# scons command for HLR
SCONS='scons -Q -D -j 8'

# directories
TOP_DIR=$( pwd )
CONTRIB_DIR=$TOP_DIR/contrib
TBB_DIR=/opt/local/tbb/2021.8

# just make sure, the directory exists
mkdir -p $CONTRIB_DIR

if [ "$?" != "0" ]; then
    echo "failed creating directory $CONTRIB_DIR"
    exit 1
fi

######################################################################
##
## determine CPU platform
##
######################################################################

ARCH=$( gcc -march=native -Q --help=target | grep march | head -n 1 | sed "s/ *-march=[ \t]*//" )

######################################################################
##
## install HPROcore alternatives
##
######################################################################

HPRO_DIR=/clusterhome/rok/programming/hpro/core

## TODO

######################################################################
##
## install BLAS alternatives
##
######################################################################

########################################
#
# Intel MKL
#
########################################

if [ "$MKLROOT" == "" ]; then
    echo "no MKLROOT is set"
    exit 1
fi

MKL_LIB_PATH="$MKLROOT/lib"
for lib in mkl_gf_lp64 mkl_sequential mkl_core; do
    if ! [ -f $MKL_LIB_PATH/lib${lib}.so ]; then
        MKL_LIB_PATH=""
        break
    fi
done

if [ "$MKL_LIB_PATH" == "" ]; then
    MKL_LIB_PATH="$MKLROOT/lib/intel64"
    for lib in mkl_gf_lp64 mkl_sequential mkl_core; do
        if ! [ -f $MKL_LIB_PATH/lib${lib}.so ]; then
            MKL_LIB_PATH=""
            break
        fi
    done
fi

if [ "$MKL_LIB_PATH" == "" ]; then
    MKL_LIB_PATH="$MKLROOT/lib/intel64_lib"
    for lib in mkl_gf_lp64 mkl_sequential mkl_core; do
        if ! [ -f $MKL_LIB_PATH/lib${lib}.so ]; then
            MKL_LIB_PATH=""
            break
        fi
    done
fi

if [ "$MKL_LIB_PATH" == "" ]; then
    echo "MKL not found in $MKLROOT"
    exit 1
else
    echo
    echo "found MKL libraries in $MKL_LIB_PATH"
    echo
fi

########################################
#
# OpenBLAS
#
########################################

OPENBLAS_PATH=$CONTRIB_DIR/OpenBLAS

if ! [ -d $OPENBLAS_PATH ] ; then
    cd $CONTRIB_DIR
    git clone https://github.com/xianyi/OpenBLAS

    if [ "$?" != "0" ]; then
        echo "failed to clone OpenBLAS"
        exit 1
    fi
fi

if ! [ -f $OPENBLAS_PATH/libopenblas.a ]; then
    cd $OPENBLAS_PATH
    git pull
    git checkout HEAD

    if [ "$?" != "0" ]; then
        echo "failed to checkout OpenBLAS"
        exit 1
    fi

    # default target which should always work
    TARGET=HASWELL

    case $ARCH in
        znver*) TARGET=ZEN ;;
        *)      TARGET=HASWELL ;;
    esac

    make TARGET=$TARGET USE_THREAD=0 USE_LOCKING=1 NO_LAPACK=1

    if [ "$?" != "0" ]; then
        echo "failed to make OpenBLAS"
        exit 1
    else
        echo
        echo "sucessfully built OpenBLAS"
        echo
    fi
else
    echo
    echo "found OpenBLAS library in $OPENBLAS_PATH"
    echo
fi

########################################
#
# BLIS
#
########################################

BLIS_PATH=$CONTRIB_DIR/blis

if ! [ -d $BLIS_PATH ] ; then
    cd $CONTRIB_DIR
    REPO="https://github.com/flame/blis"
    
    case $ARCH in
        znver*) REPO="https://github.com/amd/blis" ;;
    esac
    
    git clone $REPO blis

    if [ "$?" != "0" ]; then
        echo "failed to clone BLIS"
        exit 1
    fi
fi

# look for precompiled library
BLIS_LIB=""
if [ -f $BLIS_PATH/config.mk ]; then
    BLIS_TARGET=$( cat $BLIS_PATH/config.mk | grep -E "^CONFIG_NAME" | sed "s/.*:= //g" )
    if [ -f $BLIS_PATH/lib/$BLIS_TARGET/libblis.a ]; then
        BLIS_LIB="$BLIS_PATH/lib/$BLIS_TARGET/libblis.a"
    fi
fi

if [ "$BLIS_LIB" == "" ]; then
    cd $BLIS_PATH
    git pull
    git checkout HEAD

    if [ "$?" != "0" ]; then
        echo "failed to checkout BLIS"
        exit 1
    fi

    ./configure --disable-threading --enable-sup-handling auto
    make -j 8

    if [ "$?" != "0" ]; then
        echo "failed to make BLIS"
        exit 1
    else
        BLIS_TARGET=$( cat $BLIS_PATH/config.mk | grep -E "^CONFIG_NAME" | sed "s/.*:= //g" )
        BLIS_LIB="$BLIS_PATH/lib/$BLIS_TARGET/libblis.a"
        
        echo
        echo "sucessfully built BLIS as $BLIS_LIB"
        echo
    fi
else
    echo
    echo "found BLIS library $BLIS_LIB"
    echo
fi

########################################
#
# Reference BLAS and LAPACK
#
########################################

LAPACK_PATH=$CONTRIB_DIR/lapack

if ! [ -d $LAPACK_PATH ] ; then
    cd $CONTRIB_DIR
    git clone https://github.com/Reference-LAPACK/lapack

    if [ "$?" != "0" ]; then
        echo "failed to clone LAPACK"
        exit 1
    fi
fi

if ! [ -f $LAPACK_PATH/build/lib/liblapack.a ]; then
    cd $LAPACK_PATH
    git pull
    git checkout HEAD

    if [ "$?" != "0" ]; then
        echo "failed to checkout lapack"
        exit 1
    fi

    if ! [ -d $LAPACK_PATH/build ]; then
        mkdir -p $LAPACK_PATH/build
    fi

    cd $LAPACK_PATH/build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_FLAGS_RELEASE="-O3 -march=native -funroll-loops" ..

    if [ "$?" != "0" ]; then
        echo "failed to configure LAPACK"
        exit 1
    fi

    make -j 8

    if [ "$?" != "0" ]; then
        echo "failed to make LAPACK"
        exit 1
    else
        echo
        echo "sucessfully built LAPACK"
        echo
    fi
else
    echo
    echo "found LAPACK in $LAPACK_PATH"
    echo
fi

########################################
#
# link flags for libraries
#
########################################

MKL="-L$MKL_LIB_PATH -lmkl_gf_lp64 -lmkl_sequential -lmkl_core"
OPENBLAS="$LAPACK_PATH/build/lib/liblapack.a $OPENBLAS_PATH/libopenblas.a -lgfortran"
BLIS="$LAPACK_PATH/build/lib/liblapack.a $BLIS_LIB -lgfortran -lpthread"
REFBLAS="$LAPACK_PATH/build/lib/liblapack.a  $LAPACK_PATH/build/lib/libblas.a -lgfortran"

# make sure MKL is found at runtime
export LD_LIBRARY_PATH=$MKL_LIB_PATH:$LD_LIBRARY_PATH

######################################################################
##
## generate configurations
##
######################################################################

cat > laplace.conf <<EOF
[app]
appl = laplaceslp
adm = std
cluster = h

[arith]
ntile = 64
eps = 1e-6
EOF

cat > materncov.conf <<EOF
[app]
appl = materncov
adm = weak
cluster = h

[arith]
ntile = 64
eps = 1e-6
EOF

######################################################################
##
## build programs
##
######################################################################

#
# update HLR
#

git pull

if [ "$?" != "0" ]; then
    echo "error while updating HLR"
    exit 1
fi

#
# define common settings without building anything
#
SCONS="$SCONS programs=approx-mm,approx-lu,uniform-mm,uniform-lu frameworks=seq,tbb buildtype=release hpro=$HPRO_DIR tbb=$TBB_DIR"

build () {
    NAME=$1
    LFLAGS=$2

    $SCONS lapack=user lapackflags="$LFLAGS"

    if [ "$?" != "0" ]; then
        echo "error while building HLR with $LFLAGS"
        exit 1
    fi

    for fwork in seq tbb; do
        for prog in approx/approx-mm-${fwork} approx/approx-lu-${fwork} uniform/uniform-mm-${fwork} uniform/uniform-lu-${fwork} ; do

            # wait till it becomes available (distributed FS issues)
            while ! [ -x programs/$prog ]; do sleep 1 ; done
            
            bprog=`basename programs/$prog`
            mv programs/$prog ${bprog}-${NAME}
        done
    done
}

echo
echo "building MKL programs"
echo

build "mkl" "$MKL"

echo
echo "building OpenBLAS programs"
echo

build "openblas" "$OPENBLAS"

echo
echo "building BLIS programs"
echo

build "blis" "$BLIS"

echo
echo "building Reference BLAS programs"
echo

build "refblas" "$REFBLAS"

######################################################################
##
## benchmarks
##
######################################################################

# ARITH=std
# APPROX=svd

# optional numactl call
NUMACTL=""

run_benchmark () {
    prog=$1
    app=$2
    args="$3"

    logbase=$app--$( basename $prog )

    # for blas in $BLAS_IMPL ; do
    #     for arith in $ARITH ; do
    #         for apx in $APPROX ; do
    #             echo "    $prog-$blas-$arith-$apx"
    #             $prog-$blas $args --arith $arith --approx $apx > ${logbase}--${blas}--${arith}--${apx}.log
    #         done
    #     done
    # done

    for blas in $BLAS_IMPL ; do
        echo "    $prog-$blas-$arith-$apx"
        $NUMACTL $prog-$blas $args > ${logbase}--${blas}.log
    done
}

########################################
#
# sequential
#
########################################

echo
echo "running sequential benchmarks"
echo

COMMON_ARGS="-e 1e-4 --nbench 10 --tbench 1e10 --arith std --approx svd"

echo "  Laplace"
ARGS="--config laplace.conf --grid sphere-6 $COMMON_ARGS"
run_benchmark ./approx-mm-seq laplace "$ARGS"
run_benchmark ./approx-lu-seq laplace "$ARGS"
run_benchmark ./uniform-mm-seq laplace "$ARGS"
run_benchmark ./uniform-lu-seq laplace "$ARGS"

echo "  MaternCov"
ARGS="--config materncov.conf --grid randcube-32768 $COMMON_ARGS"
run_benchmark ./approx-mm-seq materncov "$ARGS"
run_benchmark ./approx-lu-seq materncov "$ARGS"
run_benchmark ./uniform-mm-seq materncov "$ARGS"
run_benchmark ./uniform-lu-seq materncov "$ARGS"

########################################
#
# parallel
#
########################################

NUMACTL="numactl -i all"

echo
echo "running parallel benchmarks"
echo

COMMON_ARGS="-e 1e-6 --nbench 10 --tbench 1e10 --approx svd"

echo "  Laplace"
ARGS="--config laplace.conf --grid sphere-8 $COMMON_ARGS"
run_benchmark ./approx-mm-tbb laplace "$ARGS --arith std"
run_benchmark ./approx-lu-tbb laplace "$ARGS --arith dagstd"
run_benchmark ./uniform-mm-tbb laplace "$ARGS --arith std"

echo "  MaternCov"
ARGS="--config materncov.conf --grid randcube-131072 $COMMON_ARGS"
run_benchmark ./approx-mm-tbb materncov "$ARGS --arith std"
run_benchmark ./approx-lu-tbb materncov "$ARGS --arith dagstd"
run_benchmark ./uniform-mm-tbb materncov "$ARGS --arith std"


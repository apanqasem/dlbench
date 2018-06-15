#!/bin/bash

if [ $# -eq 1 ] && [ "$1" = "--help" ]; then 
	echo "usage: ./build.sh -l <layout> -t <agent> -c -v -m <mode>"
	exit 0
fi

CC=/usr/bin/gcc
CXX=g++
SNK=snack.sh
SNKHSAIL=/opt/amd/cloc/bin/snackhsail.sh

while [ $# -gt 0 ]; do
  key="$1"
  case $key in
    -l|--layout)
      layout="$2"
      shift # option has parameter
			;;
    --pattern)
      pattern="$2"
      shift 
			if [ ${pattern} = "C2GI" ]; then
					kiters="$2"
					shift
			fi
			;;
		-m|--mode)
			mode="$2"
			shift
			;;
		-s|--size)
			size="$2"
			shift
			;;
		--pixels)
			pixels="$2"
			shift
			;;
		--threads)
			threads="$2"
			shift
			;;
    --placement)
      placement="$2"
			shift
			;;
		-r|--regs)
			regs="Y"
			;;
		-p|--sparsity)
			sparsity="$2"
			shift
			;;
		-x|--tile)
			tile="$2"
			shift
			;;
		-d|--coarsen)
			cfactor="$2"
			shift
			;;
		--mem)
			mem="$2"
			shift
			;;
		-i|--intensity)
			intensity="$2"
			shift
			;;
    -c|--copy)
      copy="COPY"
			;;
		-t|--agent)
			agent="$2"
			shift
			;;
    -b|--brig)
      module_type="BRIG"
      ;;
    -a|--alloc)
      alloc="$2"
			shift
			;;
    -v|--verbose)
      verbose="VERBOSE"
			;;
    *)
			echo "Unknown option:" $key
			exit 0
			;;
  esac
  shift
done

[ "$layout" ] || { layout="AOS"; }
[ "$pattern" ] || { pattern="C2G"; }
[ "$kiters" ] || { kiters="1"; }

[ "$mem" ] || { mem="1"; }
[ "$cfactor" ] || { cfactor="1"; }
[ "$size" ] || { size="1"; }
[ "$pixels" ] || { pixels="4096"; }
[ "$threads" ] || { threads=${pixels}; }
[ "$sparsity" ] || { sparsity="1"; }
[ "$tile" ] || { tile="64"; }

[ "$intensity" ] || { intensity="1"; }  
[ "$mode" ] || { mode="build";} 
[ "$agent" ] || { agent="DEVICE";}
[ "$placement" ] || { placement="DEV";}
[ "$copy" ] || { copy="NOCOPY";}
[ "$verbose" ] || { verbose="CURT";}

LAYOUTS="AOS DA SOA CA"

host=`hostname`
case $host in 
	xn0|xn1|xn2|xn3|xn4|xn5|xn6|xn7|xn8|xn9)
		node="kaveri"
		;;
	c0|c1|c2|c3)
		node="carrizo"
		;;
	t1|ROCNREDLINE)
		node="fiji"
		;;
	knuth|ada.cs.txstate.edu|capi.cs.txstate.edu|t0|t2|minksy)
		node="cuda"
		;;
	paripp2)
		node="paripp2"
		;;
	*)
		echo "unknown host node" $host
		exit 0
esac

HCC=hcc
HCC_CONFIG=`hcc-config --cxxflags`
CXXFLAGS="${CXXFLAGS} ${HCC_CONFIG} -std=c++11 -O3"
HCC_CONFIG=`hcc-config --ldflags`
LDFLAGS="${LDFLAGS} ${HCC_CONFIG}"


CXX=g++
INCPATH=-I.

DFLAGS=-DDEBUG

if [ $((${pixels} % ${threads})) -ne 0 ]; then
		echo "Build failed!" 
		echo "Number of threads must divide number of pixels: pixels = ${pixels}, threads = ${threads}" 
		exit 0
fi


if [ $mode = "clean" ]; then
	rm -rf *.o  *~ 
	for l in ${LAYOUTS}; do
		rm -rf dlbench_${l}
	done
	exit 0
fi	

# build from C source 
if [ $mode = "build" ]; then
		${HCC} -c -w -g ${CXXFLAGS} ${INCPATH} ${FLAGS} -DHC -DMEM=${mem} -DCOARSENFACTOR=${cfactor} -DTILESIZE=${tile} -DINTENSITY=${intensity} -DSPARSITY_VAL=${sparsity} -DPIXELS=${pixels} -D__THREADS=${threads}  -DIMGS=${size} -D${layout} -D${pattern} -DKITERS=${kiters} -D${agent} -D${placement} dlbench.cpp
    ${HCC} -o dlbench_${layout} dlbench.o -lpthread	${LDFLAGS}	

fi


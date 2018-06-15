#!/bin/bash

if [ $# -eq 1 ] && [ "$1" = "--help" ]; then 
	echo "usage: ./build.sh -l <layout> -a <allocation> -t <agent> -c -v -m <mode>"
	exit 0
fi

[ -z ${HSA_RUNTIME_PATH} ] && HSA_RUNTIME_PATH=/opt/rocm/hsa

MAPIPATH=/home/apan/git/mapi

CC=/usr/bin/gcc
CC=/opt/rocm/hcc/bin/clang
CXX=g++
CLOC_PATH=/opt/rocm/hcc2/bin
CLOC_PATH=/opt/rocm/hcc2_0.4-0/bin

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
		-p|--sparsity)
			sparsity="$2"
			shift
			;;
		--mem)
			mem="$2"
			shift
			;;
		-d|--coarsen)
			coarsen="$2"
			shift
			;;
		--sweeps)
			sweeps="$2"
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
		-x|--tile)
			tile="$2"
			shift
			;;
		-i|--intensity)
			intensity="$2"
			shift
			;;
    -c|--copy)
      copy="COPY"
			;;
    -v|--verbose)
      verbose="VERBOSE"
			;;
    -g|--debug)
      debug="true"
			;;
    -b|--brig)
      module_type="BRIG"
            ;;
    -a|--alloc)
      alloc="$2"
			shift
			;;
		-t|--agent)
			agent="$2"
			shift
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
[ "$mem" ] || { mem="2"; }
[ "$coarsen" ] || { coarsen="1"; }
[ "$size" ] || { size="1000"; }
[ "$pixels" ] || { pixels="1024"; }
[ "$threads" ] || { threads=${pixels}; }
[ "$sweeps" ] || { sweeps="1"; }
[ "$sparsity" ] || { sparsity="1"; }
[ "$intensity" ] || { intensity="1"; }
[ "$tile" ] || { tile="64"; }
[ "$agent" ] || { agent="DEVICE";}
[ "$copy" ] || { copy="NOCOPY";}
[ "$alloc" ] || { alloc="FINE";}
[ "$verbose" ] || { verbose="CURT";}
[ "$module_type" ] || { module_type="AMDGCN";}

[ "$mode" ] || { mode="build";} 

LAYOUTS="AOS DA SOA CA"

if [ $mode = "clean" ]; then
	rm -rf *.o  *~ dlbench.hsaco dlbench.brig
	for l in ${LAYOUTS}; do
		rm -rf dlbench_${l}
	done
	exit 0
fi	


host=`hostname`
case $host in 
	xn0|xn1|xn2|xn3|xn4|xn5|xn6|xn7|xn8|xn9)
		node="kaveri"
		;;
	c0|c1|c2|c3)
		node="carrizo"
		;;
	paryzen1|t1|ROCNREDLINE)
		node="fiji"
		gpu="gfx803"
		opt_level=2
		;;
  paripp2)
    node="vega"
		gpu="gfx900"
		opt_level=1  # bug in cloc.sh; doesn't build at opt_level 2
    ;;
	*)
		echo "unknown host node" $host
		exit 0
esac

OBJS="dlbench.o util.o host_kernels.o memorg.o" 

if [ $((${pixels} % ${threads})) -ne 0 ]; then
		echo "Build failed!" 
		echo "Number of threads must divide number of pixels: pixels = ${pixels}, threads = ${threads}" 
		exit 0
fi
C_PARAM_DEFS="-D${agent} -D${alloc} -D${copy} -D${layout} -D${verbose} -DMEM=${mem} -DIMGS=${size}\
              -DSPARSITY_VAL=${sparsity} -DPIXELS=${pixels} -D__THREADS=${threads} -DSWEEP_VAL=${sweeps} -DTILESIZE=${tile}\
              -DCOARSENFACTOR=${coarsen} -DINTENSITY=${intensity} -D${pattern} -DKITERS=${kiters} -D${module_type}"

CL_PARAM_DEFS="-DTILESIZE=${tile} -DCOARSENFACTOR=${coarsen} -DMEM=${mem}\
               -DSPARSITY_VAL=${sparsity} -DPIXELS=${pixels} -D__THREADS=${threads} -DSWEEP_VAL=${sweeps} -DINTENSITY=${intensity}"

# build from C source 
if [ $mode = "build" ]; then
		${CC} -O3 -I. -I${MAPIPATH}/include -I${HSA_RUNTIME_PATH}/include/hsa ${C_PARAM_DEFS} -c util.c #-std=c99
		${CC} -g -O3 -I. -I${MAPIPATH}/include -I${HSA_RUNTIME_PATH}/include/hsa ${C_PARAM_DEFS} -c memorg.c #-std=c99
		${CC} -O3 -I. -I${MAPIPATH}/include -I${HSA_RUNTIME_PATH}/include/hsa ${C_PARAM_DEFS} -c host_kernels.c  #-std=c99

	if [ ${module_type} = "BRIG" ]; then 
		# tools not yet installed on ROCNREDLINE. so just copy pre-generated files 
		if [ $host = "ROCNREDLINE" ]; then 
					echo "cp $node.dlbench.brig dlbench.brig"
					cp $node.dlbench.brig dlbench.brig
		else
			${CLOC_PATH}/cloc.sh -brig -libgcn /opt/rocm/libamdgcn -clopts "-I. ${CL_PARAM_DEFS}" -opt 2 dlbench.cl
		fi
	else

		if [ $host = "ROCNREDLINE" ]; then 
			echo "cp $node.dlbench.hsaco dlbench.hsaco"
			cp $node.dlbench.hsaco dlbench.hsaco
		else
			if [ $debug ]; then 
				echo "${CLOC_PATH}/cloc.sh -mcpu ${gpu}  -clopts "-I. ${CL_PARAM_DEFS}" -opt ${opt_level} dlbench.cl"
			fi
			${CLOC_PATH}/cloc.sh -mcpu ${gpu}  -clopts "-I. ${CL_PARAM_DEFS}" -opt ${opt_level}  dlbench.cl
		fi
	fi

	${CC} -O3 -w -g -I${MAPIPATH}/include -I${HSA_RUNTIME_PATH}/include/hsa  -I. ${C_PARAM_DEFS} -c dlbench.c #-std=c99
	${CC} -o dlbench_${layout}  ${OBJS} ${LFLAGS} -L${MAPIPATH}/lib -L/opt/rocm/lib -lhsa-runtime64 -lmapi -lpthread
fi


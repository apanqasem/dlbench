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
    -b|--blksize)
      blksize=$2
			shift
      ;;
    -a|--alloc)
      alloc="$2"
			shift
			;;
    -v|--verbose)
      verbose=true
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
[ "$blksize" ] || { blksize="256"; }

[ "$intensity" ] || { intensity="1"; }  
[ "$mode" ] || { mode="build";} 
[ "$agent" ] || { agent="DEVICE";}
[ "$placement" ] || { placement="DM";}
[ "$copy" ] || { copy="NOCOPY";}

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
	*)
		echo "unknown host node" $host
		exit 0
esac


CC=nvcc
CXX=g++
CXXFLAGS=-O3
INCPATH=-I.

if [ $host = "t0" ] || [ $host = "knuth" ]; then 
	ARCH="-arch sm_30"
fi
if [ $host = "t2" ]; then 
	ARCH="-arch sm_52"
fi
if [ $host = "capi.cs.txstate.edu" ]; then 
	ARCH="-arch sm_35"
fi
if [ $host = "ada.cs.txstate.edu" ]; then 
	ARCH="-arch sm_61"
fi
if [ $host = "minksy" ]; then 
	ARCH="-arch sm_50"
fi

FLAGS="-g ${ARCH} -O3 -ccbin g++" # --maxrregcount=16"  #
LFLAGS=-lcudadevrt 
DFLAGS=-DDEBUG

if [ $mode = "clean" ]; then
	rm -rf *.o  *~ 
	for l in ${LAYOUTS}; do
		rm -rf dlbench_${l}
	done
	exit 0
fi	

if [ $((${pixels} % ${threads})) -ne 0 ]; then
		echo "Build failed!" 
		echo "Number of threads must divide number of pixels: pixels = ${pixels}, threads = ${threads}" 
		exit 0
fi

DEFS="-DMEM=${mem} -DCOARSENFACTOR=${cfactor} -DTILESIZE=${tile} -DINTENSITY=${intensity} -DSPARSITY_VAL=${sparsity} -DPIXELS=${pixels} -DIMGS=${size} -D${layout} -D${agent} -D${placement} -DBLKS=${blksize}"

# build from C source 
if [ $mode = "build" ]; then
#		if [ $regs ]; then 
				${CC} -o dlbench_${layout} ${INCPATH} ${FLAGS} -w  ${DEFS} dlbench.cu -lpthread # 2> tmp
				if [ "${verbose}" ]; then 
						echo "${CC} -o dlbench_${layout} ${INCPATH} ${FLAGS} --keep --ptxas-options -v ${DEFS} dlbench.cu -lpthread"
				fi
				#				res=`cat tmp | grep registers | awk '{print $5}'`
#				echo $res
#				rm tmp
#		fi
else
	${CC} -o dlbench_${layout} -w -g ${INCPATH} ${FLAGS} -DMEM=${mem} -DCOARSENFACTOR=${cfactor} -DTILESIZE=${tile} -DINTENSITY=${intensity} -DSPARSITY_VAL=${sparsity} -DPIXELS=${pixels} -D__THREADS=${threads}  -DIMGS=${size} -D${layout} -D${pattern} -DKITERS=${kiters} -D${agent} -D${placement} dlbasic.cu -lpthread 
	if [ "${verbose}" ]; then 
			echo 	${CC} -o dlbench_${layout} -w -g ${INCPATH} ${FLAGS} -DMEM=${mem} -DCOARSENFACTOR=${cfactor} -DTILESIZE=${tile} -DINTENSITY=${intensity} -DSPARSITY_VAL=${sparsity} -DPIXELS=${pixels} -D__THREADS=${threads}  -DIMGS=${size} -D${layout} -D${pattern} -DKITERS=${kiters} -D${agent} -D${placement} dlbasic.cu -lpthread 
	fi
fi


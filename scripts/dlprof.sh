#!/bin/bash

function usage() {
/bin/cat 2>&1 <<"EOF"
     
    Explore DLBENCH parameter space, one dimension at a time. Print summary 
    performance statistics. Output should typically be redirected to file 

    Expects to find auxiuliary scripts dlbench_driver.sh, perf.sh and perf_cuda.sh in PATH. 

    For more details, see paper.  

    Usage: dlprof.sh [ options ] 
    
    Options:
       --help             print this help message
       -c,--counters      collect performance counter data 
       -h,--host          include experiments on host

       -p,--sparsity      sparse data access pattern 
       -b,--bw            bandwidth pressure
       -i,--intensity     arithmetic intensity 
       -s,--size          task granularity
       -a,--alloc         memory pool allocation (i.e., fine, coarse, device)
       -d,--default       single timing experiment with default config
       --sweeps           sweeps over dataset

    Options with values:

    Examples:
       dlprof.sh --sparsity -h -c    // sparsity experiments, include host, include perf counters 


EOF
	exit 1
}


if [ "$1" = "--help" ]; then
	usage
  exit 0
fi

while [ $# -gt 0 ]; do
  key="$1"
  case $key in
    -p|--sparsity)
      exper=sparsity
			;;
		-d|--default)
			exper=default
			;;
		-b|--bw)
			exper=bw
			;;
		-i|--intensity)   
			exper=intensity            
			;;
		-a|--alloc)   
			exper=alloc            
			;;
		-s|--size)   
			exper=size            
			;;
		--sweeps)   
			exper=sweeps            
			;;
		-c|--counters)   
			perf_counters="Y"            
			;;
		-h|--host)   
			host="Y"            
			;;
    *)
			echo "Unknown option:" $key
			exit 0
			;;
  esac
  shift
done

[ "${exper}" ] || { exper="intensity"; }



# determine node architecture (HSA cluster)
hostname=`hostname`
case $hostname in 
	xn0|xn1|xn2|xn3|xn4|xn5|xn6|xn7|xn8|xn9)
		node="kaveri"
		;;
	c0|c1|c2|c3)
		node="carrizo"
		;;
	t1|ROCNREDLINE)
		node="fiji"
		;;
	capi.cs.txstate.edu|t0|t2)
		node="cuda"
		;;
	*)
		echo "unknown host node" $hostname
		exit 0
esac

# check if auxiliary scripts are available 

driver=dlbench_driver.sh 
perf=perf.sh
perf_cuda=perf_cuda.sh

[ `which $driver` ] || { echo "could not find $driver" ; exit 0; }
#[ `which $perf` ] || { echo "could not find $perf" ; exit 0; }


# if [ $node = "cuda" ]; then 
# 	[ `which ${perf_cuda}` ] || { echo "could not find ${perf_cuda}" ; exit 0; }
# fi

function print_header() {
	param=$1
	layouts="AOS DA CA"
	if [ $host ]; then
		echo -n ${param},AOS_HOST_TIME,AOS_CP_TIME,
	else
		echo -n ${param},
	fi
	for layout in ${layouts}; do
		echo -n ${layout}_TIME,${layout}_CP_TIME
		if [ ${layout} !=  "CA" ]; then  # CA is last 
			echo -n ","
		fi
		if [ ${perf_counters} ]; then 
			if [ ${layout} ==  "CA" ]; then  # counter values continue after kernel times
				echo -n ","
			fi
			echo -n ${layout}_VGPR,${layout}_SGPR,${layout}_CacheHit,${layout}_MemUnitStalled,${layout}_MemUnitBusy,
      echo -n ${layout}_VALUInsts,${layout}_VFetchInsts,${layout}_VWriteInsts,${layout}_VALUBusy,
      echo -n ${layout}_SFetchInsts,${layout}_SALUBusy
			if [ ${layout} !=  "CA" ]; then  # CA is last 
				echo -n ","
			fi
		fi
	done
	echo "" # EOL
}

function print_results() {
	param=$1
	layout=$2
	i=$3
	j=$4

	if [ $layout = "AOS_HOST" ]; then 
		echo -n ${param},${host_time[$j]},
	else
		if [ $layout = "AOS" ] && [ ! "$host" ]; then  # AOS is first 
			echo -n ${param},
		fi
#		echo -n ${device_time[$i,$j]}
		if [ ${perf_counters} ]; then 
#			echo -n ${device_time[$i,$j]},${device_perf_mem[$i,$j]},${device_perf_alu[$i,$j]}
			echo -n ${device_time[$i,$j]},${device_perf[$i,$j]}
		fi
		if [ $layout !=  "CA" ]; then  # CA is last 
			echo -n ","
		fi
	fi
 
}

function alloc() {

	print_header alloc

	if [ $node = "fiji" ]; then 
		allocs="FINE COARSE DEVMEM"
	fi
  if [ $node = "kaveri" ]; then 
		allocs="FINE DEVMEM"
	fi
  if [ $node = "carrizo" ]; then 
		allocs="FINE COARSE"
	fi 
  if [ $node = "cuda" ]; then 
		allocs="FINE"
	fi 
	
	j=0
	for alloc in ${allocs}; do
		i=0
		for layout in ${layouts}; do 
			params="-l $layout -- -l $layout -a ${alloc}"
			res=`dlbench_driver.sh ${params}`
			device_time[$i,$j]=$res			

			if [ ${perf_counters} ]; then 
				if [ $node = "cuda" ]; then 
	 				device_perf[$i,$j]=`perf_cuda.sh ./dlbench_${layout}` 
				else 
					device_perf_mem[$i,$j]=`perf.sh -i mem.txt ./dlbench_${layout}` 
					device_perf_alu[$i,$j]=`perf.sh -i alu.txt ./dlbench_${layout}` 
				fi
			fi
			print_results ${alloc} ${layout} $i $j 
			i=$(($i + 1))
		done
		echo ""   # EOL after one set of sparsity values
		j=$(($j + 1))
	done
}

function sweeps() {
	print_header sweeps

	if [ $node = "fiji" ]; then 
		allocs="FINE COARSE DEVMEM"
	fi
  if [ $node = "kaveri" ]; then 
		allocs="FINE DEVMEM"
	fi
  if [ $node = "carrizo" ]; then 
		allocs="FINE"
	fi 
  if [ $node = "cuda" ]; then 
		allocs="FINE"
	fi 
	
	sweeps="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
#         "1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 150 200 250 300 350 400 450 500"
#        "1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100"


	size=100

	for alloc in ${allocs}; do
		j=0
		for sweep in ${sweeps}; do 
			i=0
			for layout in ${layouts}; do 
				params="-l $layout -- -l $layout --sweeps ${sweep} -a ${alloc} -s ${size}"
				res=`dlbench_driver.sh ${params}`
				device_time[$i,$j]=$res			
				
				if [ ${perf_counters} ]; then 
					if [ $node = "cuda" ]; then 
	 					device_perf[$i,$j]=`perf_cuda.sh ./dlbench_${layout}` 
					else 
						device_perf_mem[$i,$j]=`perf.sh -i mem.txt ./dlbench_${layout}` 
						device_perf_alu[$i,$j]=`perf.sh -i alu.txt ./dlbench_${layout}` 
					fi
				fi
				print_results ${sweep} ${layout} $i $j 
				i=$(($i + 1))
			done
			echo ""   # EOL after one set of sparsity values
			j=$(($j + 1))
		done
	done
}

function arithmetic_intensity() {

# intensities="1 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 1000 1500 2000 4000"
#	intensities="1 500 1000 2000 4000 6000 8000 10000" # 12000 14000 16000" # 18000 20000"
#  intensities="1 200 400 600 800 1000 2000 4000"
	intensities="1 20 40 60 80 100 120 140 160 180 200 220 240"
	layouts="AOS DA CA"

	mem=2  # set to low memory divergence 

#	print_header intensity

	j=0
	for intensity in ${intensities}; do 
		i=0
		if [ ${host} ]; then 
			params="-l AOS -- -l AOS -t HOST -i ${intensity} --mem $mem"
			res=`dlbench_driver.sh $params`
			host_time[$j]=$res			
			print_results ${intensity} AOS_HOST $i $j
		fi

		for layout in ${layouts}; do 
			fiji_params="-l $layout -- -l $layout -a DEVMEM -i ${intensity} --mem $mem"
			non_fiji_params="-l $layout -- -l $layout -i ${intensity} --mem $mem"

			if [ $node = "fiji" ]; then 
				res=`dlbench_driver.sh ${fiji_params}`
			else 
				res=`dlbench_driver.sh ${non_fiji_params}`
			fi
			device_time[$i,$j]=$res			
			if [ ${perf_counters} ]; then 
				if [ $node = "cuda" ]; then 
#	 				device_perf[$i,$j]=`perf_cuda.sh ./dlbench_${layout}` 
	 				device_perf[$i,$j]=`get_primary_gpu.sh --metric pwr -- ./dlbench_${layout}` 
				else 
					device_perf_mem[$i,$j]=`perf.sh -i mem.txt ./dlbench_${layout}` 
					device_perf_alu[$i,$j]=`perf.sh -i alu.txt ./dlbench_${layout}` 
				fi
			fi
			print_results ${intensity} ${layout} $i $j 
			i=$(($i + 1))
		done
		echo ""   # EOL after one set of intensity values
		j=$(($j + 1))
	done
}

function bandwidth_pressure() {

	memrefs="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18"
	layouts="AOS DA CA"

#	print_header memref

	j=0
	for memref in ${memrefs}; do 
		i=0
		if [ ${host} ]; then 
			params="-l AOS -- -l AOS -t HOST --mem $memref"
			res=`dlbench_driver.sh $params`
			host_time[$j]=$res			
			print_results ${memref} AOS_HOST $i $j
		fi

		for layout in ${layouts}; do 
			fiji_params="-l $layout -- -l $layout --mem $memref"
			non_fiji_params="-l $layout -- -l $layout -i 200 --mem $memref"

			if [ $node = "fiji" ]; then 
				res=`dlbench_driver.sh ${fiji_params}`
			else 
				res=`dlbench_driver.sh ${non_fiji_params}`
			fi
			device_time[$i,$j]=$res			
			if [ ${perf_counters} ]; then 
				if [ $node = "cuda" ]; then 
#	 				device_perf[$i,$j]=`perf_cuda.sh ./dlbench_${layout}` 
	 				device_perf[$i,$j]=`get_primary_gpu.sh --metric pwr -- ./dlbench_${layout}` 
				else 
					device_perf_mem[$i,$j]=`perf.sh -i mem.txt ./dlbench_${layout}` 
					device_perf_alu[$i,$j]=`perf.sh -i alu.txt ./dlbench_${layout}` 
				fi
			fi
			print_results ${memref} ${layout} $i $j 
			i=$(($i + 1))
		done
		echo ""   # EOL after one set of memref values
		j=$(($j + 1))
	done
}

function granularity() {

#	sizes="1 20 40 60 80 100 120 140 160 180 200 250 300 400 500 750 1000 1500 2000 4000 5000 6000 7000 8000"
	sizes="1 20 40 60 80 100 120 140 160 180 200 250 300 400"
	#500 750 1000 1500 2000 4000 5000 6000 7000 8000"
#	print_header granularity
	layouts="AOS DA CA"

	j=0
	for size in ${sizes}; do 
		i=0
		if [ ${host} ]; then 
			params="-l AOS -- -l AOS -t HOST -s ${size}"
			res=`dlbench_driver.sh $params`
			host_time[$j]=$res			
			print_results ${intensity} AOS_HOST $i $j
		fi

		for layout in ${layouts}; do 
			fiji_params="-l $layout -- -l $layout -s $size"
			non_fiji_params="-l $layout -- -l $layout -s $size"

			if [ $node = "fiji" ]; then 
				res=`dlbench_driver.sh ${fiji_params}`
			else 
				res=`dlbench_driver.sh ${non_fiji_params}`
			fi
			device_time[$i,$j]=$res			
			if [ ${perf_counters} ]; then 
				if [ $node = "cuda" ]; then 
#	 				device_perf[$i,$j]=`perf_cuda.sh ./dlbench_${layout}` 
	 				device_perf[$i,$j]=`get_primary_gpu.sh --metric pwr -- ./dlbench_${layout}` 
				else 
					device_perf_mem[$i,$j]=`perf.sh -i mem.txt ./dlbench_${layout}` 
					device_perf_alu[$i,$j]=`perf.sh -i alu.txt ./dlbench_${layout}` 
				fi
			fi
			print_results ${size} ${layout} $i $j 
			i=$(($i + 1))
		done
		echo ""   # EOL after one set of size values
		j=$(($j + 1))
	done

}

function sparsity() {

	sparsities="1 2 4 8 16 32 64"
	layouts="AOS DA CA"

	#print_header sparsity

	j=0
	for sparsity in ${sparsities}; do 
		i=0
		if [ ${host} ]; then 
			# sparsity not supported on HOST
			echo " " > /dev/null
		fi

		for layout in ${layouts}; do 
			fiji_params="-l $layout -- -l $layout -a DEVMEM --sparsity $sparsity"
			non_fiji_params="-l $layout -- -l $layout --sparsity $sparsity"

			if [ $node = "fiji" ]; then 
				res=`dlbench_driver.sh ${fiji_params}`
			else 
				res=`dlbench_driver.sh ${non_fiji_params}`
			fi
			device_time[$i,$j]=$res			
			if [ ${perf_counters} ]; then 
				if [ $node = "cuda" ]; then 
#	 				device_perf[$i,$j]=`perf_cuda.sh ./dlbench_${layout}` 
	 				device_perf[$i,$j]=`get_primary_gpu.sh --metric pwr -- ./dlbench_${layout}` 
				else 
					device_perf_mem[$i,$j]=`perf.sh -i mem.txt ./dlbench_${layout}` 
					device_perf_alu[$i,$j]=`perf.sh -i alu.txt ./dlbench_${layout}` 
				fi
			fi
			print_results ${sparsity} ${layout} $i $j 
			i=$(($i + 1))
		done
		echo ""   # EOL after one set of sparsity values
		j=$(($j + 1))
	done

}


function reg_pressure() {

	memrefs="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18"
	layouts="AOS DA CA"

	print_header regs

	j=0
	for memref in ${memrefs}; do 
		i=0
		if [ ${host} ]; then 
			params="-l AOS -- -l AOS -t HOST --mem $memref"
			res=`dlbench_driver.sh $params`
			host_time[$j]=$res			
			print_results ${memref} AOS_HOST $i $j
		fi

		for layout in ${layouts}; do 
			amd_params="-s -l $layout -- -i 300 -l $layout --mem $memref"
			cuda_params="-r -l $layout --mem $memref"

			if [ $node = "cuda" ]; then 
				res=`./build.sh ${cuda_params}`
			else
				res=`dlbench_driver.sh ${amd_params}`
			fi
			device_time[$i,$j]=$res			
			if [ ${perf_counters} ]; then 
				if [ $node = "cuda" ]; then 
	 				device_perf[$i,$j]=`perf_cuda.sh ./dlbench_${layout}` 
				else 
					device_perf_mem[$i,$j]=`perf.sh -i mem.txt ./dlbench_${layout}` 
					device_perf_alu[$i,$j]=`perf.sh -i alu.txt ./dlbench_${layout}` 
				fi
			fi
			print_results ${memref} ${layout} $i $j 
			i=$(($i + 1))
		done
		echo ""   # EOL after one set of memref values
		j=$(($j + 1))
	done
}

# reg_pressure 
# exit 0

if [ $exper = "default" ]; then 
	>&2 echo "Default timing"
	default_timing
fi
if [ $exper = "alloc" ]; then 
	>&2 echo "Allocation"
	alloc
fi
if [ $exper = "sweeps" ]; then 
	>&2 echo "Sweeps and Allocation"
	sweeps
fi
if [ ${exper} = "bw" ]; then 
	>&2 echo "Bandwidth pressure"
	bandwidth_pressure 
fi
if [ $exper = "intensity" ]; then 
	>&2 echo "Arithmetic intensity"
	arithmetic_intensity
fi
if [ $exper = "size" ]; then 
	>&2 echo "Task granularity"
	granularity
fi
if [ $exper = "sparsity" ]; then 
	>&2 echo "Sparsity"
	sparsity
fi





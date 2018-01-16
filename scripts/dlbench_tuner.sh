#!/bin/bash

function usage() {
/bin/cat 2>&1 <<"EOF"
     
    Execute DLBENCH kernel k times and report average times. Build parameters
    are passed to build script as is. Expects to find build.sh in current directory

    Usage: dlbench_driver.sh [ options ] -- [ build parameters ]
    
    Options:
       --help           print this help message
       -v, --verbose    verbose mode  

    Optionss with values:
       -r, --rpts <n>            number of repeated executions     
       -l, --layout <layout>     AOS, DA, CA or SOA. must match build parameter 

    Examples:
       dlbench_driver.sh -l DA -s 1     // build DA kernel with problem size 1

EOF
	exit 1
}

function usage() {
    echo "usage: dl_bench_tuner.sh [OPTIONS]"
    echo "tune parameters of dl_bench"
    echo ""
		echo "Options:"
		echo -e "   --help\t\t  print this help message"
}

intensity="1 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500"
layouts="AOS DA SOA CA"
agents="HOST DEVICE"

agent=DEVICE

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
	t0|t2)
		node="cuda"
		;;
	*)
		echo "unknown host node" $host
		exit 0
esac

function mem_ai() {

#	mem="1 2 4 8 16 32 64"
	mem="256 512 768 1024 2048 4096 6144 8192 10240 12888" # "16384 32768"
	intensity=1
	i=0
	layout=AOS
	for m in ${mem}; do
		res=`dlbench_driver.sh -l $layout -s 100 -r ${m} -i ${intensity}`
	  
		agent_device_aos[$i]=$res			
		aos_perf[$i]=`perf.sh -i mem_ai.txt ./dlbench_${layout}` 
#		aos_perf_alu[$i]=`perf.sh -i alu.txt ./dlbench_${layout}` 
#		aos_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
		i=$(($i+1))
	done

	i=0
	layout=DA
	for m in ${mem}; do
		res=`dlbench_driver.sh -l $layout -s 100 -r ${m} -i ${intensity}`
		agent_device_da[$i]=$res			
		da_perf[$i]=`perf.sh -i mem_ai.txt ./dlbench_${layout}` 
#		da_perf_alu[$i]=`perf.sh -i alu.txt ./dlbench_${layout}` 
#		da_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
		i=$(($i+1))
	done

	i=0
	for m in ${mem}; do 
		echo ${m},${agent_device_aos[$i]},${aos_perf[$i]},${agent_device_da[$i]},${da_perf[$i]}
		i=$(($i+1))
	done
}


function mem_access() {

	mem="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18"

	i=0
	layout=AOS
	for m in ${mem}; do
		if [ $node = "fiji" ]; then 
 			res=`dlbench_driver.sh -l $layout --mem $m`
		else
			res=`dlbench_driver.sh -l $layout --mem $m`
		fi
		agent_device_aos[$i]=$res			
		if [ $node = "cuda" ]; then 
			aos_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
		else 
			aos_perf[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
		fi
		i=$(($i+1))
	done

	i=0
	layout=DA
	for m in ${mem}; do
		if [ $node = "fiji" ]; then 
			res=`dlbench_driver.sh -l $layout --mem $m`
		else
			res=`dlbench_driver.sh -l $layout --mem $m`
		fi
		agent_device_da[$i]=$res			
		if [ $node = "cuda" ]; then 
			da_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
		else 
			da_perf[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
		fi
		i=$(($i+1))
	done

	i=0
	for m in ${mem}; do
		echo ${m},${agent_device_aos[$i]},${aos_perf[$i]},${agent_device_da[$i]},${da_perf[$i]}
		i=$(($i+1))
	done
}

function sparse() {

	sparse="1 2 4 8 16 32 64"

 	i=0
	layout=AOS
	for m in ${sparse}; do
		if [ $node = "fiji" ]; then 
 			res=`dlbench_driver.sh -l $layout -a DEVMEM --sparsity $m`
		else
			res=`dlbench_driver.sh -l $layout --sparsity $m`
		fi
		agent_device_aos[$i]=$res			
		if [ $node = "cuda" ]; then 
			aos_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
		else 
			aos_perf[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
		fi
		i=$(($i+1))
	done

	i=0
	layout=DA
	for m in ${sparse}; do
		if [ $node = "fiji" ]; then 
			res=`dlbench_driver.sh -l $layout -a DEVMEM --sparsity $m`
		else
			res=`dlbench_driver.sh -l $layout --sparsity $m`
		fi
		agent_device_da[$i]=$res			
		if [ $node = "cuda" ]; then 
			da_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
		else 
			da_perf[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
		fi
		i=$(($i+1))
	done

	i=0
	layout=CA
	for m in ${sparse}; do
		if [ $node = "fiji" ]; then 
			res=`dlbench_driver.sh -l $layout -a DEVMEM --sparsity $m`
		else
			res=`dlbench_driver.sh -l $layout --sparsity $m`
		fi
		agent_device_ca[$i]=$res			
		if [ $node = "cuda" ]; then 
			ca_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
		else 
			ca_perf[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
		fi
		i=$(($i+1))
	done


	i=0
	for m in ${sparse}; do
		echo ${m},${agent_device_aos[$i]},${aos_perf[$i]},${agent_device_da[$i]},${da_perf[$i]},${agent_device_ca[$i]},${ca_perf[$i]}
		i=$(($i+1))
	done
}

# layout single experiment. default values for size and intensity 
function layout() {
	i=0
	layout=AOS
	res=`dlbench_driver.sh -l $layout`
	agent_device_aos[$i]=$res			
	aos_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
	layout=DA
	res=`dlbench_driver.sh -l $layout`
	agent_device_da[$i]=$res			
	da_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
	layout=SOA
	res=`dlbench_driver.sh -l $layout`
	agent_device_soa[$i]=$res			
	soa_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
	layout=CA
	tile=1024
	res=`dlbench_driver.sh -x ${tile} -l $layout`
	agent_device_ca[$i]=$res			
	ca_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 

	echo ${layout},${agent_device_aos[$i]},${aos_perf[$i]}
	echo ${layout} ${agent_device_da[$i]},${da_perf[$i]}
	echo ${layout} ${agent_device_soa[$i]},${soa_perf[$i]}
	echo ${layout} ${agent_device_ca[$i]},${ca_perf[$i]}
}

# single layout with DEVMEM and size 500 
function layout_devmem() {
	i=0
	layout=AOS
	res=`dlbench_driver.sh -a DEVMEM -s 500 -l $layout`
	agent_device_aos[$i]=$res			
	aos_perf_mem[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
	aos_perf_inst[$i]=`perf.sh -i alu.txt ./dlbench_${layout}` 
	echo ${layout},${agent_device_aos[$i]},${aos_perf_mem[$i]},${aos_perf_inst[$i]}

	layout=DA
	res=`dlbench_driver.sh -a DEVMEM -s 500 -l $layout`
	agent_device_da[$i]=$res			
	da_perf_mem[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
	da_perf_inst[$i]=`perf.sh -i alu.txt ./dlbench_${layout}` 
	echo ${layout},${agent_device_da[$i]},${da_perf_mem[$i]},${da_perf_inst[$i]}

	layout=SOA
	res=`dlbench_driver.sh -l $layout`
	agent_device_soa[$i]=$res			

	# perf counters not working for dynamic SOA 
#	soa_perf_mem[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
#	soa_perf_inst[$i]=`perf.sh -i alu.txt ./dlbench_${layout}` 
	echo ${layout},${agent_device_soa[$i]} # ,${soa_perf_mem[$i]},${soa_perf_inst[$i]}

	layout=CA
	# tile=64  # Fiji
	# tile=128 # Carrizo
	tile=64  # Kaveri
	res=`dlbench_driver.sh -x ${tile} -a DEVMEM -s 500 -l $layout`
	agent_device_ca[$i]=$res			
	ca_perf_mem[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
	ca_perf_inst[$i]=`perf.sh -i alu.txt ./dlbench_${layout}` 

}
	
# intensity experiment
function intensity() {

	intensity="1 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 1000 1500 2000 4000"

	i=0
	for intense in ${intensity}; do 
		layout=AOS
		if [ $node = "fiji" ]; then 
			res=`dlbench_driver.sh -l $layout -i ${intense} --mem 14`
		else 
			res=`dlbench_driver.sh -l $layout -i ${intense} --mem 2`
		fi
		agent_device_aos[$i]=$res			
		if [ $node = "cuda" ]; then 
	 		aos_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
		else 
			aos_perf[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
			aos_perf_alu[$i]=`perf.sh -i alu.txt ./dlbench_${layout}` 
		fi


 		layout=DA
		if [ $node = "fiji" ]; then 
			res=`dlbench_driver.sh -l $layout -i ${intense} --mem 14`
		else 
			res=`dlbench_driver.sh -l $layout -i ${intense} --mem 2`
		fi
		agent_device_da[$i]=$res			
		if [ $node = "cuda" ]; then 
	 		da_perf[$i]=`perf_cuda.sh ./dlbench_${layout}` 
		else 
			da_perf[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
			da_perf_alu[$i]=`perf.sh -i alu.txt ./dlbench_${layout}` 
		fi
	
		echo ${intense},${agent_device_aos[$i]},${aos_perf[$i]},${agent_device_da[$i]},${da_perf[$i]}
		i=$(($i+1))
	done
}

function task_granularity() {
	sizes="1 20 40 60 80 100 120 140 160 180 200 250 300 400 500 750 1000 1500 2000 4000"
	sizes="5000 6000 7000 8000"
#	sizes="10 50 100 150 200 250 300 350 400 500 1000  2000" #  4000 8000 16000 24000" #  32000 40000 48000" # 56000 64000"

	agent=DEVICE
	intensities="300 400"
	mem=2
	for intense in ${intensities}; do
		layout=AOS
		i=0
		for size in ${sizes}; do 
			if [ $node = "fiji" ]; then 
				res=`dlbench_driver.sh -i ${intense} -a DEVMEM -l $layout -t $agent -s $size --mem ${mem}`
			else 
				res=`dlbench_driver.sh -i ${intense} -l $layout -t $agent -s $size --mem ${mem}`
			fi
			agent_device_aos[$i]=$res
			aos_perf[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
			i=$(($i+1))
		done
		# layout=AOS
		# agent=DEVICE
		# i=0
		# for size in ${sizes}; do 
		# 	res=`dlbench_driver.sh -i ${intensity} -l $layout -t $agent -s $size`
		# 	agent_device_aos[$i]=$res			
		# 	i=$(($i+1))
		# done
		layout=DA
		agent=DEVICE
		i=0
		for size in ${sizes}; do 
			if [ $node = "fiji" ]; then 
				res=`dlbench_driver.sh -i ${intense} -a DEVMEM -l $layout -t $agent -s $size --mem ${mem}`
			else 
				res=`dlbench_driver.sh -i ${intense} -l $layout -t $agent -s $size --mem ${mem}`
			fi
			agent_device_da[$i]=$res			
			da_perf[$i]=`perf.sh -i mem.txt ./dlbench_${layout}` 
			i=$(($i+1))
		done
		# layout=CA
		# agent=DEVICE
		# i=0
		# for size in ${sizes}; do 
		# 	res=`dlbench_driver.sh -i ${intense} -l $layout -t $agent -s $size`
		# 	agent_device_ca[$i]=$res			
		# 	i=$(($i+1))
		# done


#	done
		
#	echo "SIZE,AOS_HOST_TIME,AOS_HOST_CP_TIME,AOS_DEV_TIME,AOS_DEV_CP_TIME,DA_DEV_TIME,DA_DEV_CP_TIME,CA_DEV_TIME,CA_DEV_CP_TIME" 
	
		i=0
		for size in ${sizes}; do 
			echo ${size},${agent_device_aos[$i]},${aos_perf[$i]},${agent_device_da[$i]},${da_perf[$i]}
			i=$(($i+1))
		done	
	done
}
	


#task_granularity 

# mem_ai

# mem_access

#intensity

sparse

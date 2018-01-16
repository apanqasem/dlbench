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
       dlbench_driver.sh -l DA -- -l DA -s 1     // build DA kernel with problem size 1

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
    -r|--rpts)
      rpts="$2"
      shift 
			;;
		-w|--showregs)
			regs="Y"
			;;
		-v|--verbose)
			verbose="y"
			;;
		-l|--layout)   
			layout="$2"            
    	shift
			;;
		--)                      # begin build params 
			shift          
			args=$@
			build_args=1
			;;
    *)
			if [ ! "${build_args}" ]; then 
				echo "Unknown option:" $key
				exit 0
			fi
			;;
  esac
  shift
done


[ "$rpts" ] || { rpts=3;}
[ "$layout" ] || { layout="AOS";}

if [ $DEBUG ]; then 
	echo -e "verbose:\t\t" $verbose
	echo -e "rpts:\t\t" $rpts 
	echo -e "layout:\t\t" $layout
	echo -e "build params:\t" $args 
fi

SCRIPT="dbench_driver.sh"
[ -x build.sh ] || { echo "${SCRIPT}: build.sh not found"; exit 0; }

./build.sh -m clean 
./build.sh ${args}   &> /dev/null

if [ $regs ]; then 
	res=`perf.sh -i mem.txt ./dlbench_${layout}` 
	res=`echo $res | awk -F "," '{print $1","$2}'`
	echo $res
else 
	#	 throw out value for first execution (always anomolous)
	./dlbench_${layout}  &> /dev/null
	
	if [ ${verbose} ]; then 
		./dlbench_${layout} 
	else
		time=0
		cp_time=0
		flop=0
		# repeat execution rpts times 
		for ((i=0; i < rpts; i++)); do
			result=`./dlbench_${layout} 2> /dev/null`
			this_time=`echo $result | awk -F "," '{print $1}'`
			this_cp_time=`echo $result | awk -F "," '{print $2}'`
			time=`echo ${this_time} $time | awk '{print $1 + $2}'`
			cp_time=`echo ${this_cp_time} ${cp_time} | awk '{print $1 + $2}'`
		done
		
		# get average 
		time=`echo $time ${rpts} | awk '{printf "%3.2f", $1/$2}'`
		cp_time=`echo ${cp_time} ${rpts} | awk '{printf "%3.2f", $1/$2}'`
		
		echo ${time}","${cp_time}
	fi
fi
	
	

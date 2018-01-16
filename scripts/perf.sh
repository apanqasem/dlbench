#!/bin/bash

function usage() {
    echo "usage: perf.sh -e [event_name] [OPTIONS] [executable]"
    echo "Wrapper for sprofiler"
    echo ""
    echo "Options:"
    echo -e "   --help\t\t  print this help message"
}

if [ "$1" = "--help" ]; then
  usage
  exit 0
fi

while [ $# -gt 0 ]; do
  key="$1"
  case $key in
    -e|--event)
      event="$2"
      shift # option has parameter
      ;;
		--pwr)
			pwr=true
			;;
    -o|--outfile)
      outfile="$2"
      shift # option has parameter
      ;;
    -i|--infile)
      infile="$2"
      shift # option has parameter
      ;;
     *)
		 if [ ! $prog ]; then 
			 prog="$1"
		 else
			 echo "Unknown option:" $key
       exit 0
		 fi
     ;;
  esac
  shift
done

[ "$prog" ] || { echo "no executable specified. Exiting..."; exit 0 ; }
[ "$event" ] || { event="Wavefront" ;}
[ "$outfile" ] || { outfile="$prog.csv" ;}

PROFILER=/opt/rocm/profiler/bin/CodeXLGpuProfiler

if [ "$infile" ]; then 
	rm -rf $outfile
	${PROFILER} --hsapmc -c $infile -o $outfile ./$prog  &> /dev/null
	if ! [ -r $outfile ]; then 
		echo "Profiling failed. Exiting"
		exit 0
	fi
	val=`tail -1 $outfile`
	events=`wc -l $infile | awk '{print $1}'`
	i=1
	j=0
	while read event; do
		event_val=`echo $val | awk -v awk_events=$events -v awk_i=$i '{j = (NF - awk_events) + awk_i; print $j}'`
		event_val=`echo ${event_val} | sed 's/,//g'`
		perf_events[$j]=${event_val}
		i=$((i+1))
		j=$((j+1))
	done < $infile	
else 
	echo $event > event.txt
	rm -rf $outfile
	${PROFILER} --hsapmc -c event.txt -o $outfile ./$prog  &> /dev/null
	if ! [ -r $outfile ]; then 
		echo "Profiling failed. Exiting"
		exit 0
	fi
	val=`tail -1 $outfile  | awk '{print $NF}'`
	rm -rf event.txt 
	echo $event $val
fi

k=0;
while [ $k -lt $(($j-1)) ]; do
	echo -n ${perf_events[$k]},
	k=$((k+1))
done 
echo ${perf_events[$k]}

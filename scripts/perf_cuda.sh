#!/bin/bash

get_primary_gpu.sh -m pwr -- ./$1
exit 0
prog=$1
#events=`nvprof --metrics gld_transactions,gst_transactions  ./${prog} 2>&1 | tail -2 | awk  '{print $NF}'`
#events=`echo $events | sed 's/ /,/'`

#events1=`nvprof --metrics gld_efficiency,gst_efficiency  ./${prog} 2>&1 | tail -2 | awk  '{print $NF}'`
#events1=`echo $events1 | sed 's/ /,/' | sed 's/\%//g'`

events2=`nvprof --metrics gld_throughput,gst_throughput  ./${prog} 2>&1 | tail -2 | awk  '{print $NF}'`
events2=`echo $events2 | sed 's/ /,/' | sed 's/GB\/s//g'`

events3=`nvprof --metrics dram_read_throughput,dram_write_throughput ./${prog} 2>&1 | tail -2 | awk  '{print $NF}'`
events3=`echo $events3 | sed 's/ /,/' | sed 's/GB\/s//g'`

events1=`nvprof --metrics gld_transactions_per_request,gst_transactions_per_request  ./${prog} 2>&1 | tail -2 | awk  '{print $NF}'`
events1=`echo $events1 | sed 's/ /,/'`

#events2=`nvprof --metrics local_memory_overhead,local_replay_overhead  ./${prog} 2>&1 | tail -2 | awk  '{print $NF}'`
#events2=`echo $events2 | sed 's/ /,/'`

#events2=`nvprof --metrics local_memory_overhead ./${prog} 2>&1 | tail -2 | awk  '{print $NF}'`
#events2=`echo $events2 | sed 's/ /,/'`


#events3=`nvprof --metrics ldst_executed,ldst_issued  ./${prog} 2>&1 | tail -2 | awk  '{print $NF}'`
#events3=`echo $events3 | sed 's/ /,/'`

#events4=`nvprof --metrics local_load_transactions,local_store_transactions  ./${prog} 2>&1 | tail -2 | awk  '{print $NF}'`
#events4=`echo $events4 | sed 's/ /,/'`

#events5=`nvprof --metrics l2_l1_read_hit_rate,l2_read_transactions  ./${prog} 2>&1 | tail -2 | awk  '{print $NF}'`
#events5=`echo $events5 | sed 's/ /,/'`

#events5=`nvprof --metrics l2_read_transactions  ./${prog} 2>&1 | tail -2 | awk  '{print $NF}'`
#events5=`echo $events5 | sed 's/ /,/'`

#echo $events,$events1,$events2,$events3,$events4,$events5
echo $events1,$events2,$events3 # ,$events4,$events5

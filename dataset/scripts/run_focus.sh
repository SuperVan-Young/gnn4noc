#!/bin/bash
cnt=0

FOCUS=/home/xuechenhao/focus_scheduler/focus.py  # change to your root
BENCHMARK=/home/xuechenhao/gnn4noc/dataset/benchmark

for task in $(ls ${BENCHMARK})
do
    # at most 8 tasks run parallelly
    if [ $cnt -eq 8 ] ; then
        wait
        cnt=0
    fi

    TASK_ROOT=/home/xuechenhao/gnn4noc/dataset/sim_result/"${task%.yaml}"_b1w1024_8x8
    num=`ls ${TASK_ROOT} -l | grep "^-" | wc -l`

    if [ $num -lt 3 ] ; then

        echo ${task}
        # to run timeloop, use mode "ted"
        # to run simulation, use mode "sd"    
        python ${FOCUS} -bm ${BENCHMARK}/${task} -d 8 -b 1 -fr 1024-1024-1024 tesd &

        cnt=$(($cnt+1))

    elif [ $num -le 4 ] ; then

        echo ${task}
        # to run timeloop, use mode "ted"
        # to run simulation, use mode "sd"    
        python ${FOCUS} -bm ${BENCHMARK}/${task} -d 8 -b 1 -fr 1024-1024-1024 sd &

        cnt=$(($cnt+1))

    fi
done
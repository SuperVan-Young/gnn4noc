#!/bin/bash
cnt=0

FOCUS=/home/xuechenhao/focus_scheduler/focus.py
BENCHMARK=/home/xuechenhao/gnn4noc/dataset/benchmark
tasks=$(ls ../benchmark)

cd /home/xuechenhao/focus_scheduler
pwd

python ${FOCUS} -bm ${BENCHMARK}/alexnet_2.yaml -d 4 -b 1 -fr 1024-1024-1024 ds &
wait

for task in tasks
do
    if [ $cnt -eq 8 ] ; then
        wait
        cnt=0
        echo "************************"
    fi

    # put your command here

    cnt=$(($cnt+1))
done
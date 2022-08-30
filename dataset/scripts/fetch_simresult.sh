#!/bin/bash

SIM_RESULT_ROOT=../sim_result

FOCUS_ROOT=/home/xuechenhao/focus_scheduler  # change to your focus
OP_GRAPH_ROOT=${FOCUS_ROOT}/buffer/op_graph
SIMULATOR_ROOT=${FOCUS_ROOT}/simulator/tasks

NOC_SUFFIX=b1w1024_8x8

for task in $(ls ../benchmark)
do
    task=${task%.yaml}_${NOC_SUFFIX}
    TASK_ROOT=${SIM_RESULT_ROOT}/${task}
    SIMULATOR_TASK_ROOT=${SIMULATOR_ROOT}/${task}

    if ! [ -e $TASK_ROOT ]
    then
        echo $task
        mkdir $TASK_ROOT
    fi

    op_graph=op_graph_${task}.gpickle
    cp ${OP_GRAPH_ROOT}/${op_graph} ${TASK_ROOT}/op_graph.gpickle

    cp ${SIMULATOR_TASK_ROOT}/out.log ${TASK_ROOT}/out.log
    cp ${SIMULATOR_TASK_ROOT}/routing_board ${TASK_ROOT}/routing_board
    cp ${SIMULATOR_TASK_ROOT}/spatial_spec ${TASK_ROOT}/spatial_spec

done
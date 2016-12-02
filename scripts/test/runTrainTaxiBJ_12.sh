#!/bin/sh
export THEANO_FLAGS="base_compiledir=${HOME}/tmp/gpu1,device=gpu1"
#export DATAPATH=$HOME/data/traffic_flow
export DATAPATH=$HOME/dev/DeepST/data
#export PYTHONPATH=$HOME/workspace/DeepST:$PYTHONPATH

mkdir LOG
lrs="0.0002"
ts="1"
ps="1"
cs="3"
i=0
Layers="12"
GPUS=("0")
for l in $Layers ; do
for lr in $lrs ; do
    for c in $cs ; do
    for p in $ps ; do
    for t in $ts ; do
    j=${GPUS[$i]}
    echo "GPU ${j}"
    THEANO_FLAGS="base_compiledir=${HOME}/tmp/gpu${j},device=gpu${j}" python trainTaxiBJ.py $lr $c $p $t $l #1>ret$lr-c$c-p$p-t$t-resnet$l.txt 2>LOG/ret$lr-c$c-p$p-t$t-resnet$l.log &
    (( i+= 1))
done
done
done
done
    if [ $i -eq 4 ] ; then
        echo $i
        i=0
        echo $i
        wait
    fi
done

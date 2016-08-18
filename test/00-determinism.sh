#!/bin/bash

for i in `seq 4`; do
  THEANO_FLAGS="device=gpu0" nmt-train -c confs/debug.conf -T attnorm -o adam -l 0.002 -g 0 -w 0.00003 &> /dev/null &
  THEANO_FLAGS="device=gpu1" nmt-train -c confs/debug.conf -T attnorm -o adam -l 0.002 -g 5 -w 0.00003 &> /dev/null &
  THEANO_FLAGS="device=gpu2" nmt-train -c confs/debug.conf -T attnorm -o adadelta -l 1 -g 0 -w 0.00003 &> /dev/null &
  THEANO_FLAGS="device=gpu3" nmt-train -c confs/debug.conf -T attnorm -o adadelta -l 1 -g 5 -w 0.00003 &> /dev/null &
  wait
done

#for sys in $(ls --color=none *run1*log); do k=${i/run1.log/run}; python ~/git/nmtpy/test/is-same.py "${k}"*log; done

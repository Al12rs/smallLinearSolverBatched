#!/bin/bash
module load cuda

nvprof --export-profile timeline.prof -f --profile-from-start off --cpu-profiling on  ./Release/linearSolverBatchedLinux
#--export-profile timeline.prof -f
#srun --nodes=1 --ntasks-per-node=1 --mem=6G --time=3 --cpus-per-task=1 --partition=gll_usr_gpuprod --gres=gpu:kepler:1 ./moduleLoad.sh
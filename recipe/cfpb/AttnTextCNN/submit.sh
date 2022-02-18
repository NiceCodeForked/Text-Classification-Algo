#!/bin/bash

parentdir='/share/mini1/res/t/asr/studio/read-us/timit/demixing/nlp-algo'
recipedir='recipe/cfpb/AttnTextCNN'
submitjob='/share/mini1/sw/mini/jet/latest/tools/submitjob'

number_of_gpus=1
number_of_threads=$(($number_of_gpus * 4))

# gpu='GeForceRTX3090'
gpu='GeForceGTXTITANX'
# gpu='GeForceGTX1080Ti'
# gpu='GeForceRTX3060'

S=$parentdir/$recipedir/run.sh
L=$parentdir/$recipedir/run_log/$(date +"%Y_%m_%d_%I_%M_%p")-logging.log

$submitjob -g$number_of_gpus -M$number_of_threads -m 15000 -o -l gputype=$gpu -eo $L $S
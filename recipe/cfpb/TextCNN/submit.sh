#!/bin/bash

parentdir='/share/mini1/res/t/asr/studio/read-us/timit/demixing/Text-Classification-Algo'
recipedir='recipe/cfpb/TextCNN'
submitjob='/share/mini1/sw/mini/jet/latest/tools/submitjob'

number_of_gpus=1
number_of_threads=$(($number_of_gpus * 4))

# gpu='GeForceRTX3090'
gpu='GeForceGTXTITANX'
# gpu='GeForceGTX1080Ti'
# gpu='GeForceRTX3060'

S=$parentdir/$recipedir/run.sh
L=$parentdir/$recipedir/logging.log

$submitjob -g$number_of_gpus -M$number_of_threads -m 15000 -o -l gputype=$gpu -eo $L $S
#!/bin/bash

parentdir='/share/mini1/res/t/asr/studio/read-us/timit/demixing/nlp-algo'
logdir='recipe/cfpb/AttnTextCNN/run_log'

# Print out latest log inside of run_log folder
tail -f "$(ls $parentdir/$logdir/*.log -Art | tail -n 1)"
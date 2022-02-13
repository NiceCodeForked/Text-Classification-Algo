#!/bin/bash

pythonpath="/share/mini1/sw/std/python/anaconda3-2020.11/2020.11/envs/demixing2/bin/python"
parentdir='/share/mini1/res/t/asr/studio/read-us/timit/demixing/Text-Classification-Algo'
recipedir='recipe/cfpb/TextCNN'

execfile=$parentdir/$recipedir/train.py
$pythonpath $execfile
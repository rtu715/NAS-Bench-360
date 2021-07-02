#!/bin/bash


TASK=$1
TYPE=$2  #default/tuning/retrain

if [ $TYPE = 'default' ]
then
	det experiment create scripts/$TASK.yaml .
else
	det experiment create scripts/$TASK'_'$TYPE.yaml .
fi

#!/bin/bash
TASK=$1
TYPE=$2  #default/tuning/retrain
TEST=$3  #local/aws
if [ $TYPE = 'default' ]; then
	if [ $TEST = 'aws' ]; then
		det experiment create scripts/$TASK.yaml .
	else
		det experiemnt create scripts/$TASK.yaml . --local --test
	fi
else
	if [ $TEST = 'aws' ];then
		det experiment create scripts/$TASK'_'$TYPE.yaml . 
	else
		det experiment create scripts/$TASK'_'$TYPE.yaml . --local --test
	fi
fi

#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]
then
    tensorboard --logdir /home/tat/mounts/deepmachine/MIRAUAR/results/logs/
elif [[ "$OSTYPE" == "darwin"* ]]
then
    # Mac OSX
    tensorboard --logdir /Users/tat-macbook/mounts/deepmachine/MIRAUAR/results/logs/
fi

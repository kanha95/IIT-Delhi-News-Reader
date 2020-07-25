#!/bin/bash

if [ $1 -eq 1 ]
then
    python3 runNBSVM.py
elif [ $1 -eq 2 ]
then
    python3 runDistilBert.py
else
    python3 runMLSTM.py
fi

#!/bin/bash

source env/bin/activate
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm

if [ $1 -eq 1 ]
then
    python3 app.py 1
elif [ $1 -eq 2 ]
then
    python3 app.py 2
else
    python3 app.py 3
fi

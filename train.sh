#!/usr/bin/env bash

CONFIG=$1

python run.py -p train -c $CONFIG -b 24 -gpu '0, 1, 2, 3, 4, 5, 6, 7'
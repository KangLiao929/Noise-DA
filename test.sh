#!/usr/bin/env bash

CONFIG=$1

python run.py -p test -c $CONFIG -b 1 -gpu '0'
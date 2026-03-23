#!/bin/bash

source $(dirname "$0")/.venv/bin/activate
python3 $(dirname "$0")/do.py $1 $2 $3 $4 $5


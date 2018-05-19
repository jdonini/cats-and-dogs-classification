#!/bin/bash

progress(){
  echo -n "$0: Please wait ."
  while true
  do
    sleep 5
    echo -n "."
  done
}

dobackup(){
    # put backup commands here
    conda env create -f environment-gpu.yml
}

# Start it in the background
progress &

# You need to use the PID to kill the function
MYSELF=$!

# Transfer control to dobackup()
dobackup

# Kill progress
kill $MYSELF >/dev/null 2>&1

echo -n "... done."
echo
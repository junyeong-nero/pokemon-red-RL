#!/bin/bash

while true; do
	# latest file with .zip extension
	latest_file=$(ls -lt runs/*.zip | awk 'NR==1 {print $9}' | sed 's/\.zip$//')
	echo "starting with checkpoint: $latest_file"
	echo "$latest_file" | python train_agent.py

done

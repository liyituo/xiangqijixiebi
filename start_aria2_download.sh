#!/bin/bash
source /etc/network_turbo
/root/miniconda3/bin/python /root/generate_aria2.py
echo "Starting aria2c high-speed multi-threaded download..."
aria2c -c -j 3 -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=5 -i /root/aria2_urls.txt > /root/aria2_download.log 2>&1 &
echo "Aria2 background task started, PID: $!"

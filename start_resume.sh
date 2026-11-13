#!/bin/bash
source /etc/network_turbo
pkill -f aria2c 2>/dev/null
sleep 1
aria2c -c -j 2 -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=5 -i /root/aria2_resume.txt > /root/aria2_resume.log 2>&1 &
echo "Aria2 resume started, PID: $!"

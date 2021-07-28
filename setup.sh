#!/bin/sh
add-apt-repository ppa:jonathonf/ffmpeg-4
apt-get update
apt-get install ffmpeg
apt-get install python python-pip build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev -y
#!/bin/bash
g++ -std=c++11 -g0 -O3 stereo_calib.cpp -o stereo_calibration `pkg-config --cflags --libs opencv`
g++ -std=c++11 -g0 -O3 imagelist_creator.cpp -o image_list_creator `pkg-config --cflags --libs opencv`
cp stereo_calibration image_list_creator ../

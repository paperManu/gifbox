#!/bin/bash
rm grabs/*
./gifomaton
cd grabs
../image_list_creator images.xml camera_*
../stereo_calibration images.xml -w 9 -h 6
cp extrinsics.yml intrinsics.yml ..
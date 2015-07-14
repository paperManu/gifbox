#!/bin/bash
rm -f /tmp/camera_*.jpg
gifengine
pushd ./
cd /tmp
image_list_creator images.xml camera_*
stereo_calibration images.xml -w 9 -h 6
popd
cp /tmp/extrinsics.yml /tmp/intrinsics.yml ./

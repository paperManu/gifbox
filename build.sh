#!/bin/bash
g++ -std=c++11 -g0 -O3 -o gifomaton main.cpp stereoCamera.cpp `pkg-config --cflags --libs opencv`

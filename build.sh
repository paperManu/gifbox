#!/bin/bash
g++ -Wall -std=c++11 -o gifomaton gifbox.cpp stereoCamera.cpp v4l2output.cpp `pkg-config --cflags --libs opencv`

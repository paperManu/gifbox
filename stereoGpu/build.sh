#!/bin/bash
g++ -v -Wall -std=c++11 main.cpp -o stereoGpu `pkg-config --cflags --libs opencv`

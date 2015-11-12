/*
 * Copyright (C) 2015 Emmanuel Durand
 *
 * This file is part of GifBox.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GifBox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GifBox.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef RGBDCAMERA_H
#define RGBDCAMERA_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

/*************/
class K2Camera
{
    public:
        K2Camera();
        ~K2Camera();

        explicit operator bool() const
        {
            return _ready;
        }

        bool grab();
        bool isReady() const;
        cv::Mat retrieveRGB();
        cv::Mat retrieveDisparity();
        cv::Mat retrieveDepthMask();
        void saveToDisk();

        void activateCalibration() {_activateCalibration = !_activateCalibration;}
        void showCalibrationLines() {_showCalibrationLines = !_showCalibrationLines;}
        void setWhiteBalance(float r, float g, float b) {_balanceRed = r; _balanceGreen = g; _balanceBlue = b;}

    private:
        bool _ready {false};

        libfreenect2::Freenect2 _freenect2;
        std::unique_ptr<libfreenect2::Freenect2Device> _device {nullptr};
        std::unique_ptr<libfreenect2::PacketPipeline> _pipeline {nullptr};

        std::thread _grabThread;
        std::mutex _grabMutex;
        bool _continueGrab {false};

        std::chrono::system_clock::time_point _startTime;

        bool _showCalibrationLines {false};
        bool _activateCalibration {true};
        float _bgLearningTime {0.001};

        // White balance parameters
        float _balanceRed {1.f};
        float _balanceGreen {1.f};
        float _balanceBlue {1.f};

        // Frames, on host and client
        cv::Mat _rgbMap;
        cv::Mat _depthMap;
        cv::Mat _depthMask;

        // Filtering stuff
        cv::Mat _closeElement;
        cv::Mat _erodeElement;
        cv::Mat _dilateElement;
        cv::Ptr<cv::BackgroundSubtractorMOG2> _bgSubtractor;

        unsigned int _captureIndex {0};
};

#endif

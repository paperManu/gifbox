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
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "./libfreenect.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

/*************/
class FreenectCamera : public Freenect::FreenectDevice
{
    public:
        FreenectCamera(freenect_context* ctx, int index)
            : Freenect::FreenectDevice(ctx, index)
        {
            _width = 640;
            _height = 480;
            _frameSize = _width * _height;

            _bufferVideo.resize(_frameSize * 3);
            _bufferDepth.resize(_frameSize);
        }

        /*****/
        void VideoCallback(void* rgb, uint32_t timestamp)
        {
            std::unique_lock<std::mutex> lock(_rgbMutex);
            uint8_t* rgbPtr = static_cast<uint8_t*>(rgb);
            std::copy(rgbPtr, rgbPtr + _frameSize * 3, _bufferVideo.begin());
            _newRgbFrame = true;
        }

        /*****/
        void DepthCallback(void* depth, uint32_t timestamp)
        {
            std::unique_lock<std::mutex> lock(_depthMutex);
            uint16_t* depthPtr = static_cast<uint16_t*>(depth);
            std::copy(depthPtr, depthPtr + _frameSize, _bufferDepth.begin());
            _newDepthFrame = true;
        }

        /*****/
        bool getRGB(cv::Mat& rgbMat)
        {
            std::unique_lock<std::mutex> lock(_rgbMutex);
            if (!_newRgbFrame)
                return false;
            rgbMat = cv::Mat(_bufferVideo, true);
            rgbMat = rgbMat.reshape(3, _height);
            _newRgbFrame = false;
            return true;
        }

        /*****/
        bool getDepth(cv::Mat& depthMat)
        {
            std::unique_lock<std::mutex> lock(_depthMutex);
            if (!_newDepthFrame)
                return false;
            depthMat = cv::Mat(_bufferDepth, true);
            depthMat = depthMat.reshape(1, _height);
            _newDepthFrame = false;
            return true;
        }

    private:
        size_t _frameSize {0};
        size_t _width {0};
        size_t _height {0};
        std::vector<uint16_t> _bufferDepth;
        std::vector<uint8_t> _bufferVideo;
        std::mutex _rgbMutex;
        std::mutex _depthMutex;
        bool _newRgbFrame {false};
        bool _newDepthFrame {false};
};

/*************/
class RgbdCamera
{
    public:
        RgbdCamera();
        ~RgbdCamera();

        explicit operator bool() const
        {
            return _ready;
        }

        bool grab();
        bool isReady() const;
        cv::Mat retrieveRGB() {return _rgbMap.clone();}
        cv::Mat retrieveDisparity() {return _depthMap.clone();}
        cv::Mat retrieveDepthMask() {return _depthMask.clone();}
        void saveToDisk();

        void activateCalibration() {_activateCalibration = !_activateCalibration;}
        void showCalibrationLines() {_showCalibrationLines = !_showCalibrationLines;}
        void setWhiteBalance(float r, float g, float b) {_balanceRed = r; _balanceGreen = g; _balanceBlue = b;}

    private:
        bool _ready {false};
        Freenect::Freenect _freenectCtx;
        FreenectCamera* _camera;

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

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

#ifndef STEREOCAMERA_H
#define STEREOCAMERA_H

#include <chrono>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudastereo.hpp>

/*************/
class StereoCamera
{
    public:
        enum StereoMode
        {
            BM,
            CSBP
        };

        StereoCamera(StereoMode mode = BM);
        StereoCamera(std::vector<int> camIndices, StereoMode mode = BM);

        explicit operator bool() const
        {
            if (_calibrationLoaded && _cameras.size() >= 2)
                return true;
            else
                return false;
        }

        void compute();
        bool grab();
        bool isReady();
        bool loadConfiguration(std::string intrinsic, std::string extrinsic);
        std::vector<cv::Mat>& retrieve();
        std::vector<cv::Mat>& retrieveRemapped();
        cv::Mat retrieveDisparity() {return _disparityMap.clone();}
        cv::Mat retrieveDepthMask() {return _depthMask.clone();}
        void saveToDisk();

        void activateCalibration() {_activateCalibration = !_activateCalibration;}
        void showCalibrationLines() {_showCalibrationLines = !_showCalibrationLines;}
        void setBgLearningTime(float v) {_bgLearningTime = v;}
        void setWhiteBalance(float r, float g, float b) {_balanceRed = r; _balanceGreen = g; _balanceBlue = b;}

    private:
        std::chrono::system_clock::time_point _startTime;
        std::vector<cv::VideoCapture> _cameras;

        StereoMode _stereoMode;
        bool _showCalibrationLines {false};
        bool _activateCalibration {true};
        float _bgLearningTime {0.001};

        // White balance parameters
        float _balanceRed {1.f};
        float _balanceGreen {1.f};
        float _balanceBlue {1.f};

        // Frames, on host and client
        std::vector<cv::Mat> _frames;
        std::vector<cv::Mat> _remappedFrames;
        std::vector<cv::cuda::GpuMat> _d_frames;
        cv::cuda::GpuMat _d_disparity;
        cv::cuda::GpuMat _d_background;
        cv::cuda::GpuMat _d_depthMask;

        std::vector<std::vector<cv::Mat>> _rmaps;
        cv::Mat _disparityMap;
        cv::Mat _depthMask;

        cv::Ptr<cv::StereoMatcher> _stereoMatcher;
        cv::Ptr<cv::cuda::DisparityBilateralFilter> _disparityFilter;

        cv::Ptr<cv::BackgroundSubtractor> _bgSubtractor;
        cv::Ptr<cv::BackgroundSubtractor> _bgSubtractor2;
        cv::Ptr<cv::cuda::Filter> _closeFilter;
        cv::Ptr<cv::cuda::Filter> _dilateFilter;

        unsigned int _captureIndex {0};

        struct Calibration
        {
            cv::Mat cameraMatrix;
            cv::Mat distCoeffs;
            cv::Mat rotation;
            cv::Mat position;
        };
        std::vector<Calibration> _calibrations;
        bool _calibrationLoaded {false};

        bool correctImages();
        void computeDisparity();
        void computeBackground();
        void computeMask();

        void init(std::vector<int> camIndices);
};

#endif

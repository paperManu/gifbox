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
 * blobserver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with blobserver.  If not, see <http://www.gnu.org/licenses/>.
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
#include <opencv2/cudastereo.hpp>


/*************/
class StereoCamera
{
    public:
        StereoCamera();
        StereoCamera(std::vector<int> camIndices);

        void computeDisparity();
        bool grab();
        bool isReady();
        bool loadConfiguration(std::string intrinsic, std::string extrinsic);
        std::vector<cv::Mat>& retrieve();
        std::vector<cv::Mat>& retrieveRemapped();
        cv::Mat retrieveDisparity() {return _disparityMap.clone();}
        void saveToDisk();

    private:
        std::chrono::system_clock::time_point _startTime;
        std::vector<cv::VideoCapture> _cameras;

        // Frames, on host and client
        std::vector<cv::Mat> _frames;
        std::vector<cv::Mat> _remappedFrames;
        std::vector<cv::cuda::GpuMat> _d_frames;
        cv::cuda::GpuMat _d_disparity;

        std::vector<std::vector<cv::Mat>> _rmaps;
        cv::Mat _disparityMap;

        cv::Ptr<cv::StereoMatcher> _stereoMatcher;

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

        void init(std::vector<int> camIndices);
};

#endif

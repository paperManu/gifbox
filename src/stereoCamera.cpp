#include "stereoCamera.h"

#include <chrono>
#include <iostream>

using namespace std;
using namespace cv;

/*************/
StereoCamera::StereoCamera(StereoMode mode)
{
    _stereoMode = mode;
    vector<int> camIndices {0, 1};
    init(camIndices);
}

/*************/
StereoCamera::StereoCamera(vector<int> camIndices, StereoMode mode)
{
    _stereoMode = mode;
    init(camIndices);
}

/*************/
void StereoCamera::init(vector<int> camIndices)
{
    _startTime = chrono::system_clock::now();
    for (auto& idx : camIndices)
    {
        cout << "Opening camera " << idx << endl;
        _cameras.emplace_back(VideoCapture());
        _cameras[_cameras.size() - 1].open("v4l2:///dev/video" + to_string(idx));
        //_cameras[_cameras.size() - 1].set(cv::CAP_PROP_EXPOSURE, 300);
    }

    _frames.resize(camIndices.size());
    _calibrations.resize(camIndices.size());

    if (_stereoMode == BM)
        _stereoMatcher = cuda::createStereoBM(32, 9);
    else if (_stereoMode == CSBP)
        _stereoMatcher = cuda::createStereoConstantSpaceBP(32, 8, 4, 4, CV_16SC1);

    _disparityFilter = cuda::createDisparityBilateralFilter(32, 3, 3);

    _bgSubtractor = cuda::createBackgroundSubtractorMOG2(500);
    Mat element = getStructuringElement(cv::MORPH_ELLIPSE, Size(5, 5));
    _closeFilter = cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, element, Point(-1, -1), 1);
    _dilateFilter = cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, element, Point(-1, -1), 4);
}

/*************/
void StereoCamera::compute()
{
    if (correctImages())
    {
        computeDisparity();
        computeBackground();
        computeMask();
    }
}

/*************/
bool StereoCamera::correctImages()
{
    if (!_calibrationLoaded)
    {
        cout << "Stereo::computeDisparity - No valid calibration has been loaded" << endl;
        return false;
    }

    if (_frames.size() == 0)
    {
        cout << "Stereo::computeDisparity - No frame captured" << endl;
        return false;
    }

    for (auto& frame : _frames)
    {
        if (frame.empty())
        {
            cout << "Stereo::computeDisparity - Not all frames were grabbed" << endl;
            return false;
        }
    }

    if (_rmaps.size() == 0)
    {
        _rmaps.resize(_frames.size());
        Size imageSize = _frames[0].size();

        for (unsigned int i = 0; i < _frames.size(); ++i)
        {
            _rmaps[i].resize(2);
            initUndistortRectifyMap(_calibrations[i].cameraMatrix, _calibrations[i].distCoeffs,
                                    _calibrations[i].rotation, _calibrations[i].position,
                                    imageSize, CV_16SC2, _rmaps[i][0], _rmaps[i][1]);
        }

        _disparityMap = Mat(_frames[0].rows, _frames[0].cols, CV_16S);
    }

    _remappedFrames.resize(_frames.size());
    if (_activateCalibration)
    {
        for (unsigned int i = 0; i < _frames.size(); ++i)
            remap(_frames[i], _remappedFrames[i], _rmaps[i][0], _rmaps[i][1], cv::INTER_LINEAR);
    }
    else
    {
        for (unsigned int i = 0; i < _frames.size(); ++i)
            _remappedFrames[i] = _frames[i];
    }

    _d_frames.resize(_frames.size());

    return true;
}

/*************/
void StereoCamera::computeDisparity()
{
    for (unsigned int i = 0; i < _frames.size(); ++i)
    {
        cv::Mat gray, resizedGray;
        cv::cvtColor(_remappedFrames[i], gray, COLOR_BGR2GRAY);
        cv::resize(gray, resizedGray, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
        _d_frames[i].upload(resizedGray);
    }

    _stereoMatcher->compute(_d_frames[0], _d_frames[1], _d_disparity);
    cv::Mat disparityMap;
    _d_disparity.download(disparityMap);
    if (_stereoMode == CSBP)
    {
        disparityMap.convertTo(_disparityMap, CV_8U);
        cv::resize(_disparityMap, disparityMap, _frames[0].size());
        _disparityMap = disparityMap;
    }
    else
    {
        cv::resize(disparityMap, _disparityMap, _frames[0].size());
    }

    // Refine disparity
    _d_disparity.upload(_disparityMap);
    _d_frames[0].upload(_remappedFrames[1]);
    _disparityFilter->apply(_d_disparity, _d_frames[0], _d_frames[1]);

    // Filter it
    _dilateFilter->apply(_d_frames[1], _d_disparity);

    _d_disparity.download(_disparityMap);
}

/*************/
void StereoCamera::computeBackground()
{
    _d_frames[0].upload(_remappedFrames[0]);
    _bgSubtractor->apply(_d_frames[0], _d_frames[1], _bgLearningTime);
    cv::cuda::threshold(_d_frames[1], _d_frames[0], 1, 255, cv::THRESH_BINARY);
    _closeFilter->apply(_d_frames[0], _d_background);

    cv::Mat bgSegmentation;
    _d_background.download(bgSegmentation);
    cv::imshow("bgseg", bgSegmentation);
}

/*************/
void StereoCamera::computeMask()
{
    cv::cuda::multiply(_d_disparity, _d_frames[0], _d_depthMask, 1.0 / 255.0);
    _d_depthMask.download(_depthMask);
}

/*************/
bool StereoCamera::grab()
{
    bool state = true;
    for (unsigned int i = 0; i < _cameras.size(); ++i)
    {
        state &= _cameras[i].grab();
    }
    if (!state)
        return state;

    for (unsigned int i = 0; i < _cameras.size(); ++i)
    {
        cv::Mat rawFrame;
        state &= _cameras[i].retrieve(rawFrame, 0);
        cv::Mat bayerFrame(rawFrame.rows, rawFrame.cols, CV_8U, rawFrame.ptr());
        cv::cvtColor(bayerFrame, _frames[i], cv::COLOR_BayerGB2RGB);
    }

    if (_showCalibrationLines)
    {
        for (auto& frame : _frames)
        {
            cv::line(frame, cv::Point(frame.cols / 2, 0), cv::Point(frame.cols / 2, frame.rows), cv::Scalar(0, 255, 0));
            cv::line(frame, cv::Point(0, frame.rows / 2), cv::Point(frame.cols, frame.rows / 2), cv::Scalar(0, 255, 0));

            // This also outputs the mean value of both cameras
            auto mean = cv::mean(frame);
            string text;
            for (int i = 0; i < mean.rows; ++i)
                text += to_string(mean[i]) + " // ";
            cv::putText(frame, text, cv::Point(32, 32), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));
        }
    }

    return state;
}

/*************/
bool StereoCamera::isReady()
{
    bool ready = true;
    for (auto& cam : _cameras)
        ready &= cam.isOpened();

    return ready;
}

/*************/
bool StereoCamera::loadConfiguration(string intrinsic, string extrinsic)
{
    FileStorage fileIntrinsic, fileExtrinsic;
    if (!fileIntrinsic.open(intrinsic, FileStorage::READ))
        return false;
    if (!fileExtrinsic.open(extrinsic, FileStorage::READ))
        return false;
    
    try
    {
        for (unsigned int i = 1; i <= _cameras.size(); ++i)
        {
            fileIntrinsic["M" + to_string(i)] >> _calibrations[i - 1].cameraMatrix;
            fileIntrinsic["D" + to_string(i)] >> _calibrations[i - 1].distCoeffs;

            fileExtrinsic["R" + to_string(i)] >> _calibrations[i - 1].rotation;
            fileExtrinsic["P" + to_string(i)] >> _calibrations[i - 1].position;
        }
    }
    catch (int e)
    {
        _calibrationLoaded = false;
        cout << "Exception caught while opening configuration" << endl;
    }

    _calibrationLoaded = true;

    return true;
}

/*************/
vector<Mat>& StereoCamera::retrieve()
{
    return _frames;
}

/*************/
vector<Mat>& StereoCamera::retrieveRemapped()
{
    return _remappedFrames;
}

/*************/
void StereoCamera::saveToDisk()
{
    //auto now = chrono::system_clock::now();
    //int timestamp = (now - _startTime).count() / 1e6;
    for (unsigned int i = 0; i < _frames.size(); ++i)
    {
        string filename = "/tmp/camera_" + to_string(_captureIndex) + "_" + to_string(i) + ".jpg";
        imwrite(filename, _frames[i], {CV_IMWRITE_JPEG_QUALITY, 95});
        cout << "Save images in " << filename << endl;
    }
    _captureIndex++;
}

#include "stereoCamera.h"

#include <chrono>
#include <iostream>

using namespace std;
using namespace cv;

/*************/
StereoCamera::StereoCamera()
{
    vector<int> camIndices {0, 1};
    init(camIndices);
}

/*************/
StereoCamera::StereoCamera(vector<int> camIndices)
{
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
        _cameras[_cameras.size() - 1].open(idx);
        //_cameras[_cameras.size() - 1].set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        //_cameras[_cameras.size() - 1].set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    }

    _frames.resize(camIndices.size());
    _calibrations.resize(camIndices.size());

    _stereoMatcher = cuda::createStereoBM(64, 11);
    _d_frames.resize(2);
}

/*************/
void StereoCamera::computeDisparity()
{
    if (!_calibrationLoaded)
    {
        cout << "Stereo::computeDisparity - No valid calibration has been loaded" << endl;
        return;
    }

    if (_frames.size() == 0)
    {
        cout << "Stereo::computeDisparity - No frame captured" << endl;
        return;
    }

    for (auto& frame : _frames)
    {
        if (frame.empty())
        {
            cout << "Stereo::computeDisparity - Not all frames were grabbed" << endl;
            return;
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
    for (unsigned int i = 0; i < _frames.size(); ++i)
        remap(_frames[i], _remappedFrames[i], _rmaps[i][0], _rmaps[i][1], CV_INTER_LINEAR);

    _d_frames.resize(_frames.size());
    for (unsigned int i = 0; i < _frames.size(); ++i)
    {
        cv::Mat gray;
        cv::cvtColor(_remappedFrames[i], gray, COLOR_BGR2GRAY);
        _d_frames[i].upload(gray);
    }
    _d_disparity = cuda::GpuMat(_frames[0].size(), CV_8U);

    _stereoMatcher->compute(_d_frames[0], _d_frames[1], _d_disparity);
    _d_disparity.download(_disparityMap);
}

/*************/
bool StereoCamera::grab()
{
    bool state = true;
    for (unsigned int i = 0; i < _cameras.size(); ++i)
        state &= _cameras[i].grab();
    if (!state)
        return state;

    for (unsigned int i = 0; i < _cameras.size(); ++i)
        state &= _cameras[i].retrieve(_frames[i], 0);

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
        string filename = "grabs/camera_" + to_string(_captureIndex) + "_" + to_string(i) + ".jpg";
        imwrite(filename, _frames[i], {CV_IMWRITE_JPEG_QUALITY, 95});
    }
    _captureIndex++;
}

#include "rgbdCamera.h"

using namespace std;

/*************/
RgbdCamera::RgbdCamera()
{
    try
    {
        _camera = &_freenectCtx.createDevice<FreenectCamera>(0);
        _camera->startVideo();
        _camera->startDepth();

        auto depthFormat = FREENECT_DEPTH_REGISTERED;
        _camera->setDepthFormat(depthFormat);

        // Prepare filters
        _closeElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10));
        _dilateElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        _erodeElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        _bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 4, false);

        _ready = true;
    }
    catch (...)
    {
        _ready = false;
    }
}

/*************/
RgbdCamera::~RgbdCamera()
{
    //_camera->stopVideo();
    //_camera->stopDepth();
    //_freenectCtx.deleteDevice(0);
}

/*************/
bool RgbdCamera::grab()
{
    _camera->getRGB(_rgbMap);
    cv::cvtColor(_rgbMap, _rgbMap, cv::COLOR_BGR2RGB);
    _camera->getDepth(_depthMap);

    if (_depthMap.rows && _depthMap.cols)
    {
        if (!_depthMask.rows && !_depthMask.cols)
            _depthMask = cv::Mat(_depthMap.size(), CV_8U);

        _depthMap.convertTo(_depthMask, CV_8U, 1.0 / 32.0);
        cv::Mat unknownMask;
        cv::threshold(_depthMask, unknownMask, 1, 255, cv::THRESH_BINARY_INV);
        cv::Mat fgMask;
        _bgSubtractor->apply(_depthMask, fgMask, 0.00001);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_ERODE, _erodeElement);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_DILATE, _dilateElement);
        fgMask = 255 - fgMask;
        unknownMask += fgMask;
        _depthMask = _depthMask + unknownMask;
        
        cv::morphologyEx(_depthMask, _depthMask, cv::MORPH_OPEN, _closeElement);
    }
    
    return true;
}

/*************/
bool RgbdCamera::isReady() const
{
    return _ready;
}

/*************/
void RgbdCamera::saveToDisk()
{
}

#include "k2Camera.h"

#include <chrono>

#define ALPHA_WALL_WIDTH 48

using namespace std;

/*************/
K2Camera::K2Camera()
{
    try
    {
        // Prepare filters
        _closeElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        _dilateElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        _erodeElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        _bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);
        //_bgSubtractor->setVarThreshold(2);
        //_bgSubtractor->setVarThresholdGen(4);
        //_bgSubtractor->setDetectShadows(false);

        // Connect to the Kinect2
        libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Debug));

        if (_freenect2.enumerateDevices() == 0)
        {
            cout << "K2Camera: no Kinect2 device connected" << endl;
            _ready = false;
            return;
        }

        auto serial = _freenect2.getDefaultDeviceSerialNumber();
        _pipeline = unique_ptr<libfreenect2::PacketPipeline>(new libfreenect2::OpenGLPacketPipeline());
        _device = unique_ptr<libfreenect2::Freenect2Device>(_freenect2.openDevice(serial, _pipeline.get()));

        // Run the kinect thread
        _continueGrab = true;
        _grabThread = thread([&]() {
            libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
            libfreenect2::FrameMap frames;
            libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);

            _device->setColorFrameListener(&listener);
            _device->setIrAndDepthFrameListener(&listener);
            _device->start();

            cout << "K2Camera: device serial is " << _device->getSerialNumber() << endl;
            cout << "K2Camera: device firmware version is " << _device->getFirmwareVersion() << endl;

            auto registration = unique_ptr<libfreenect2::Registration>(new libfreenect2::Registration(_device->getIrCameraParams(), _device->getColorCameraParams()));
            auto bgMat = cv::Mat();

            auto updateFrameNumber = 0;

            while (_continueGrab)
            {
                listener.waitForNewFrame(frames);
                libfreenect2::Frame* rgb = frames[libfreenect2::Frame::Color];
                libfreenect2::Frame* depth = frames[libfreenect2::Frame::Depth];

                registration->apply(rgb, depth, &undistorted, &registered);

                unique_lock<mutex> lock(_grabMutex);
                _rgbMap = cv::Mat(cv::Size(registered.width, registered.height), CV_8UC4, registered.data).clone();
                cv::cvtColor(_rgbMap, _rgbMap, cv::COLOR_RGBA2RGB);
                
                // Process the RGB image to remove holes
                if (_rgbMap.total() != 0)
                {
                    cv::Mat rgbMask = cv::Mat(_rgbMap.size(), CV_8U);
                    cv::cvtColor(_rgbMap, rgbMask, cv::COLOR_RGB2GRAY);
                    cv::threshold(rgbMask, rgbMask, 1, 255, cv::THRESH_BINARY_INV);

                    cv::Mat rgbDilated = _rgbMap.clone();
                    cv::morphologyEx(rgbDilated, rgbDilated, cv::MORPH_DILATE, _dilateElement, cv::Point(), 3);

                    for (int y = 0; y < _rgbMap.rows; ++y)
                        for (int x = 0; x < _rgbMap.cols; ++x)
                        {
                            unsigned char maskValue = rgbMask.at<unsigned char>(y, x);
                            if (maskValue > 0)
                            {
                                _rgbMap.at<cv::Vec3b>(y, x)[0] = rgbDilated.at<cv::Vec3b>(y, x)[0];
                                _rgbMap.at<cv::Vec3b>(y, x)[1] = rgbDilated.at<cv::Vec3b>(y, x)[1];
                                _rgbMap.at<cv::Vec3b>(y, x)[2] = rgbDilated.at<cv::Vec3b>(y, x)[2];
                            }
                        }
                }

                // Process the depth map to convert tu 8U
                _depthMap = cv::Mat(cv::Size(depth->width, depth->height), CV_32F, depth->data).clone();
                if (_depthMap.rows && _depthMap.cols)
                {
                    if (!_depthMask.rows && !_depthMask.cols)
                        _depthMask = cv::Mat(_depthMap.size(), CV_8U);

                    _depthMap.convertTo(_depthMask, CV_8U, 1.0 / 32.0);
                    cv::Mat unknownMask;
                    cv::threshold(_depthMask, unknownMask, 1, 255, cv::THRESH_BINARY_INV);
                    cv::Mat fgMask;

                    // The background is updated only during the first few frames
                    if (updateFrameNumber++ < 500)
                        _bgSubtractor->apply(_depthMask, fgMask, -1.0);
                    else
                        _bgSubtractor->apply(_depthMask, fgMask, 0.0);

                    cv::morphologyEx(fgMask, fgMask, cv::MORPH_ERODE, _erodeElement);
                    cv::morphologyEx(fgMask, fgMask, cv::MORPH_DILATE, _dilateElement);
                    fgMask = 255 - fgMask;
                    unknownMask += fgMask;
                    _depthMask = _depthMask + unknownMask;
                    
                    //cv::morphologyEx(_depthMask, _depthMask, cv::MORPH_OPEN, _closeElement);
                }

                // Crop inside the depth and RGB images
                int topMargin = 16;
                int bottomMargin = 32;
                _rgbMap = cv::Mat(_rgbMap, cv::Rect(0, topMargin, _rgbMap.cols, _rgbMap.rows - topMargin - bottomMargin));
                _depthMask = cv::Mat(_depthMask, cv::Rect(0, topMargin, _depthMask.cols, _depthMask.rows - topMargin - bottomMargin));

                // Add some alpha on the vertical border of the depth mask, to handle
                // capture issues on the walls of the box
                for (int32_t y = 0; y < _depthMask.rows; ++y)
                {
                    for (int32_t x = 0; x < ALPHA_WALL_WIDTH; ++x)
                    {
                        float alpha = (float)x / (float)ALPHA_WALL_WIDTH;
                        _depthMask.at<uint8_t>(y, x) *= alpha;
                        _depthMask.at<uint8_t>(y, _depthMask.cols - x) *= alpha;
                    }
                }

                _ready = true;

                listener.release(frames);
            }

            _device->stop();
            _device->close();
        });

    }
    catch (...)
    {
        _ready = false;
    }
}

/*************/
K2Camera::~K2Camera()
{
    _continueGrab = false;

    if (_grabThread.joinable())
        _grabThread.join();
}

/*************/
bool K2Camera::grab()
{
    unique_lock<mutex> lock(_grabMutex);

    if (_rgbMap.total() == 0 || _depthMap.total() == 0 || _depthMask.total() == 0)
        return false;

    return true;
}

/*************/
bool K2Camera::isReady() const
{
    return _ready;
}

/*************/
cv::Mat K2Camera::retrieveRGB()
{
    unique_lock<mutex> lock(_grabMutex);
    return _rgbMap.clone();
}

/*************/
cv::Mat K2Camera::retrieveDisparity()
{
    unique_lock<mutex> lock(_grabMutex);
    return _depthMap.clone();
}

/*************/
cv::Mat K2Camera::retrieveDepthMask()
{
    unique_lock<mutex> lock(_grabMutex);
    return _depthMask.clone();
}

/*************/
void K2Camera::saveToDisk()
{
}

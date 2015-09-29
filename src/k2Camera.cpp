#include "k2Camera.h"

#include <chrono>

using namespace std;

/*************/
K2Camera::K2Camera()
{
    try
    {
        // Prepare filters
        _closeElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        _dilateElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        _erodeElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        _bgSubtractor = cv::createBackgroundSubtractorMOG2(1000, 2, false);
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

            while (_continueGrab)
            {
                listener.waitForNewFrame(frames);
                libfreenect2::Frame* rgb = frames[libfreenect2::Frame::Color];
                libfreenect2::Frame* depth = frames[libfreenect2::Frame::Depth];

                registration->apply(rgb, depth, &undistorted, &registered);

                unique_lock<mutex> lock(_grabMutex);
                _rgbMap = cv::Mat(cv::Size(registered.width, registered.height), CV_8UC4, registered.data).clone();
                cv::cvtColor(_rgbMap, _rgbMap, cv::COLOR_RGBA2RGB);
                _depthMap = cv::Mat(cv::Size(depth->width, depth->height), CV_32F, depth->data).clone();

                if (_depthMap.rows && _depthMap.cols)
                {
                    if (!_depthMask.rows && !_depthMask.cols)
                        _depthMask = cv::Mat(_depthMap.size(), CV_8U);

                    _depthMap.convertTo(_depthMask, CV_8U, 1.0 / 32.0);
                    cv::Mat unknownMask;
                    cv::threshold(_depthMask, unknownMask, 1, 255, cv::THRESH_BINARY_INV);
                    cv::Mat fgMask;
                    _bgSubtractor->apply(_depthMask, fgMask, 0.0000001);
                    cv::morphologyEx(fgMask, fgMask, cv::MORPH_ERODE, _erodeElement);
                    cv::morphologyEx(fgMask, fgMask, cv::MORPH_DILATE, _dilateElement);
                    fgMask = 255 - fgMask;
                    unknownMask += fgMask;
                    _depthMask = _depthMask + unknownMask;
                    
                    cv::morphologyEx(_depthMask, _depthMask, cv::MORPH_OPEN, _closeElement);
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

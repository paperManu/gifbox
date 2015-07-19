#include "v4l2output.h"

#include <errno.h>
#include <string.h>

using namespace std;

/*************/
V4l2Output::V4l2Output(int width, int height, string device)
{
    _device = device;
    _sink = open(device.c_str(), O_WRONLY);
    if (_sink < 0)
    {
        cout << "Unable to open v4l2 loopback device: " << _device << endl;
        return;
    }

    struct v4l2_format v4l2format;
    v4l2format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    if (ioctl(_sink, VIDIOC_G_FMT, &v4l2format))
    {
        cout << "Error while getting v4l2 loopback device format: " << strerror(errno) << endl;
        return;
    }

    v4l2format.fmt.pix.width = width;
    v4l2format.fmt.pix.height = height;
    v4l2format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    v4l2format.fmt.pix.sizeimage = width * height * 3;
    if (ioctl(_sink, VIDIOC_S_FMT, v4l2format) < 0)
    {
        cout << "Error while setting v4l2 loopback device format: " << strerror(errno) << endl;
        return;
    }

    _width = width;
    _height = height;
    
    cout << "V4L2 loopback device successfully opened" << endl;
}

/*************/
V4l2Output::~V4l2Output()
{
    if (_sink >= 0)
        close(_sink);
}

/*************/
bool V4l2Output::writeToDevice(void* data, size_t size)
{
    if (_sink < 0)
        return false;

    if ((int)size != write(_sink, data, size))
    {
        cout << "Error while sending frame" << endl;
        return false;
    }

    return true;
}

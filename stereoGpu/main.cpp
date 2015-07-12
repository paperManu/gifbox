#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/cudastereo.hpp>

#include <linux/videodev2.h>

using namespace std;
using namespace cv;

string v4l2dev = "/dev/video0";
int v4l2sink = -1;

/*************/
int main(int argc, char** argv)
{

    // Initialize cameras and stereo matcher
    vector<VideoCapture> cameras;
    cameras.emplace_back(300);
    cameras.emplace_back(301);

    if (cameras[0].isOpened())
        cout << "Camera 0 opened" << endl;
    if (cameras[1].isOpened())
        cout << "Camera 1 opened" << endl;

    auto bm = cuda::createStereoBM(128);

    vector<Mat> captures(2);
    cameras[0] >> captures[0];
    Mat dispColor(captures[0].size(), captures[0].type());

    // Initialize v4l2loopback device
    v4l2sink = open(v4l2dev.c_str(), O_WRONLY);
    if (v4l2sink < 0)
    {
        cout << "Unable to open v4l2 loopback device: " << v4l2dev << endl;
        exit(1);
    }
    struct v4l2_format v4l2format;
    v4l2format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    if (ioctl(v4l2sink, VIDIOC_G_FMT, &v4l2format))
    {
        cout << "Error while setting v4l2 loopback device" << endl;
        exit(1);
    }
    v4l2format.fmt.pix.width = captures[0].cols;
    v4l2format.fmt.pix.height = captures[0].rows;
    v4l2format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    v4l2format.fmt.pix.sizeimage = dispColor.total() * dispColor.elemSize();
    if (ioctl(v4l2sink, VIDIOC_S_FMT, v4l2format) < 0)
    {
        cout << "Error while setting v4l2 loopback device" << endl;
        exit(1);
    }
    cout << "V4L2 loopback device successfully opened" << endl;

    // Main loop
    while (true)
    {
        cameras[0] >> captures[0];
        cameras[1] >> captures[1];

        vector<Mat> gray(2);

        cvtColor(captures[0], gray[0], COLOR_BGR2GRAY);
        cvtColor(captures[1], gray[1], COLOR_BGR2GRAY);

        vector<cuda::GpuMat> d_captures(2);
        d_captures[0].upload(gray[0]);
        d_captures[1].upload(gray[1]);

        imshow("left", captures[0]);
        imshow("right", captures[1]);

        Mat disp(captures[0].size(), CV_8U);
        cuda::GpuMat d_disp(captures[0].size(), CV_8U);
        bm->compute(d_captures[0], d_captures[1], d_disp);

        d_disp.download(disp);
        imshow("disparity", disp);

        cvtColor(disp, dispColor, COLOR_GRAY2BGR);

        if (dispColor.total() * dispColor.elemSize() != write(v4l2sink, dispColor.data, dispColor.total() * dispColor.elemSize()))
        {
            cout << "Error while sending frame" << endl;
        }

        auto key = waitKey(16);
        if (key >= 0)
            break;
    }

    // Clean v4l2loopback
    close(v4l2sink);

    return 0;
}

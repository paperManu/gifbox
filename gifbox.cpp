#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "stereoCamera.h"
#include "v4l2output.h"

using namespace std;

/*************/
int main(int argc, char** argv)
{
    vector<int> camIndices {0, 1};
    StereoCamera stereoCamera(camIndices);
    stereoCamera.loadConfiguration("intrinsics.yml", "extrinsics.yml");

    V4l2Output v4l2sink(640, 480);
    if (!v4l2sink)
        exit(1);

    cv::Mat disparityColor;

    bool continueLoop = true;
    while(continueLoop)
    {
        stereoCamera.grab();

        vector<cv::Mat> frames = stereoCamera.retrieve();

        stereoCamera.computeDisparity();
        cv::Mat disparity = stereoCamera.retrieveDisparity();
        if (disparityColor.total() == 0)
            disparityColor = cv::Mat(frames[0].size(), frames[0].type());
        cv::cvtColor(disparity, disparityColor, cv::COLOR_GRAY2BGR);
        v4l2sink.writeToDevice(disparityColor.data, disparityColor.total() * disparityColor.elemSize());

        vector<cv::Mat> remappedFrames = stereoCamera.retrieveRemapped();

        unsigned int index = 0;
        for (auto& frame : remappedFrames)
        {
            string name = "Camera remapped" + to_string(index);
            cv::imshow(name, frame);
            index++;
        }
        cv::imshow("disparity", disparityColor);

        short key = cv::waitKey(16);
        switch (key)
        {
        default:
            if (key != -1)
                cout << "Pressed key: " << key << endl;
            break;
        case 27: // Escape
            continueLoop = false;
            break;
        case 's': // Save images to disk
            stereoCamera.saveToDisk();
        }
    }

    return 0;
}

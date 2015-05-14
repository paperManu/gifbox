#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "stereoCamera.h"

using namespace std;

/*************/
int main(int argc, char** argv)
{
    vector<int> camIndices {1, 0};
    StereoCamera stereoCamera(camIndices);
    stereoCamera.loadConfiguration("intrinsics.yml", "extrinsics.yml");

    cv::Mat background, sceneMask, loadBuffer;
    loadBuffer = cv::imread("assets/ours_bg.png");
    cv::resize(loadBuffer, background, cv::Size(640, 360), 0, 0, cv::INTER_LINEAR);
    loadBuffer = cv::imread("assets/ours_mask.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::resize(loadBuffer, sceneMask, cv::Size(640, 360), 0, 0, cv::INTER_LINEAR);

    bool continueLoop = true;
    while(continueLoop)
    {
        bool state;
        state = stereoCamera.grab();

        vector<cv::Mat> frames = stereoCamera.retrieve();
        if (argc > 1)
        {
            stereoCamera.computeDisparity();
            cv::Mat dispImg = stereoCamera.retrieveDisparity();
            double minVal, maxVal;
            cv::minMaxLoc(dispImg, &minVal, &maxVal);
            cv::Mat disparity8U(dispImg.rows, dispImg.cols, CV_8UC1);
            dispImg.convertTo(disparity8U, CV_8UC1, 255 / (maxVal - minVal));
            cv::imshow("Disparity", disparity8U);

            vector<cv::Mat> remappedImg = stereoCamera.retrieveRemapped();
            cv::Mat cameraImg;
            cv::resize(remappedImg[0], cameraImg, cv::Size(640, 360), 0, 0, cv::INTER_LINEAR);
            cv::Mat mergeImg = cv::Mat(dispImg.rows, dispImg.cols, CV_8UC3);
            for (unsigned int y = 0; y < disparity8U.rows; ++y)
                for (unsigned int x = 0; x < disparity8U.cols; ++x)
                {
                    unsigned char disparity = disparity8U.at<uchar>(y, x);
                    unsigned char maskSceneVal = sceneMask.at<uchar>(y, x);
                    if (disparity > 128 && disparity > maskSceneVal)
                    {
                        mergeImg.at<cv::Vec3b>(y, x) = cameraImg.at<cv::Vec3b>(y, x);
                    }
                    else
                    {
                        mergeImg.at<cv::Vec3b>(y, x) = background.at<cv::Vec3b>(y, x);
                    }
                }
            cv::resize(mergeImg, loadBuffer, cv::Size(1280, 720), 0, 0, cv::INTER_LINEAR);
            cv::flip(loadBuffer, mergeImg, 1);
            cv::imshow("mergeImg", mergeImg);
        }

        unsigned int index = 0;
        for (auto& frame : frames)
        {
            string name = "Camera " + to_string(index);
            cv::imshow(name, frame);
            index++;
        }

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

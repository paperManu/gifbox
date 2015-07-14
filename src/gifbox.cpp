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
 * along with GifBox.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "filmPlayer.h"
#include "httpServer.h"
#include "stereoCamera.h"
#include "v4l2output.h"

using namespace std;

/*************/
int main(int argc, char** argv)
{
    // Launch http server
    HttpServer server("127.0.0.1", "8888");
    auto requestHandler = server.getRequestHandler();
    thread serverThread = thread([&]() {
        server.run();
    });

    // Load films
    vector<FilmPlayer> films;
    films.emplace_back("./films/ALL_THE_RAGE/", 32, 2);
    bool isReady = true;
    for (auto& film : films)
        isReady = isReady && film;
    if (!isReady)
    {
        cout << "Could not load films. Exiting" << endl;
        return 1;
    }
    films[0].start();

    // Load cameras
    vector<int> camIndices {0, 1};
    StereoCamera stereoCamera(camIndices);
    stereoCamera.loadConfiguration("intrinsics.yml", "extrinsics.yml");

    // Prepare v4l2 loopback
    V4l2Output v4l2sink(640, 480, "/dev/video1");
    if (!v4l2sink)
        return 1;

    cv::Mat disparityColor;

    bool continueLoop = true;
    while(continueLoop)
    {
        // Do camera stuff
        stereoCamera.grab();
        vector<cv::Mat> frames = stereoCamera.retrieve();

        stereoCamera.computeDisparity();
        cv::Mat disparity = stereoCamera.retrieveDisparity();
        if (disparityColor.total() == 0)
            disparityColor = cv::Mat(frames[0].size(), frames[0].type());
        cv::applyColorMap(disparity, disparityColor, cv::COLORMAP_JET);
        v4l2sink.writeToDevice(disparityColor.data, disparityColor.total() * disparityColor.elemSize());

        vector<cv::Mat> remappedFrames = stereoCamera.retrieveRemapped();

        unsigned int index = 0;
        for (auto& frame : remappedFrames)
        {
            string name = "Camera remapped" + to_string(index);
            cv::imshow(name, frame);
            index++;
        }
        cv::imshow("disparity", disparity);

        // Get current film frame
        auto frame = films[0].getCurrentFrame();
        auto frameMask = films[0].getCurrentMask();

        // Merge images
        cv::Mat cameraMask;
        // Background layer
        cv::Mat finalImage = frame[1].clone();
        cv::threshold(disparity, cameraMask, 48, 255, cv::THRESH_BINARY);
        cv::Mat tmpMat, tmpMask;
        // Between BG and FG
        cv::resize(cameraMask, tmpMask, finalImage.size(), cv::INTER_LINEAR);
        cv::resize(remappedFrames[0], tmpMat, finalImage.size(), cv::INTER_LINEAR);
        tmpMat.copyTo(finalImage, tmpMask);
        // Foreground layer
        frame[0].copyTo(finalImage, frameMask[0]);
        // Foregrounder (!) layer
        cv::threshold(disparity, cameraMask, 60, 255, cv::THRESH_BINARY);
        cv::resize(cameraMask, tmpMask, finalImage.size(), cv::INTER_LINEAR);
        tmpMat.copyTo(finalImage, tmpMask);

        cv::imshow("Result", finalImage);

        // Handle HTTP requests
        RequestHandler::Command cmd = requestHandler->getNextCommand();
        if (cmd == RequestHandler::Command::quit)
            continueLoop = false;
        else if (cmd == RequestHandler::Command::shot)
            stereoCamera.saveToDisk();

        // Handle keyboard
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

    server.stop();
    serverThread.join();

    return 0;
}

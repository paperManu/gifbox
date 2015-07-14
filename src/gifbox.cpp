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
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "filmPlayer.h"
#include "httpServer.h"
#include "layerMerger.h"
#include "stereoCamera.h"
#include "v4l2output.h"

using namespace std;

struct State
{
    bool run {true};
    bool sendToV4l2 {false};
    bool record {false};
};

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
    unique_ptr<V4l2Output> v4l2sink;

    // And the layer merger
    LayerMerger layerMerger;

    State state;
    while(state.run)
    {
        // Do camera stuff
        stereoCamera.grab();
        vector<cv::Mat> frames = stereoCamera.retrieve();

        stereoCamera.computeDisparity();
        cv::Mat disparity = stereoCamera.retrieveDisparity();

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

        cv::Mat cameraMaskBG, cameraMaskFG;
        cv::threshold(disparity, cameraMaskBG, 48, 255, cv::THRESH_BINARY);
        cv::threshold(disparity, cameraMaskFG, 60, 255, cv::THRESH_BINARY);
        auto finalImage = layerMerger.mergeLayersWithMasks({frame[1], frames[0], frame[0], frames[0]},
                                                           {cameraMaskBG, frameMask[0], cameraMaskFG});

        cv::imshow("Result", finalImage);

        // Write the result to v4l2
        if (state.sendToV4l2)
        {
            if (!v4l2sink || finalImage.rows != v4l2sink->getHeight() || finalImage.cols != v4l2sink->getWidth())
                v4l2sink = unique_ptr<V4l2Output>(new V4l2Output(finalImage.cols, finalImage.rows, "/dev/video1"));
            if (*v4l2sink)
                v4l2sink->writeToDevice(finalImage.data, finalImage.total() * finalImage.elemSize());
        }

        // Handle HTTP requests
        pair<RequestHandler::Command, RequestHandler::ReturnFunction> cmd;
        while ((cmd = requestHandler->getNextCommand()).first != RequestHandler::Command::nop)
        {
            if (cmd.first == RequestHandler::Command::quit)
                state.run = false;
            else if (cmd.first == RequestHandler::Command::record)
                layerMerger.setSaveMerge(true, "/tmp/gifbox_result");
            else if (cmd.first == RequestHandler::Command::stop)
            {
                layerMerger.setSaveMerge(false);
                state.sendToV4l2 = false;
            }
            else if (cmd.first == RequestHandler::Command::start)
                state.sendToV4l2 = true;

            cmd.second(true);
        }

        // Handle keyboard
        short key = cv::waitKey(16);
        switch (key)
        {
        default:
            if (key != -1)
                cout << "Pressed key: " << key << endl;
            break;
        case 27: // Escape
            state.run = false;
            break;
        case 's': // Save images to disk
            stereoCamera.saveToDisk();
        }
    }

    server.stop();
    serverThread.join();

    return 0;
}

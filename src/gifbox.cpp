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
#include "values.h"

using namespace std;

/*************/
struct State
{
    bool run {true};
    bool sendToV4l2 {false};
    bool record {false};

    int cam1 {0};
    int cam2 {1};
    int camOut {2};

    string currentFilm {"ALL_THE_RAGE"};
    int frameNbr {0};
    float fps {5.f};

    int fgLimit {12};
    int bgLimit {10};

    float bgLearningTime {0.001};

    float balanceRed {1.f};
    float balanceBlue {1.f};
} _state;

/*************/
void parseArguments(int argc, char** argv)
{
    for (int i = 1; i < argc;)
    {
        if ("-cam1" == string(argv[i]) && i < argc - 1)
            _state.cam1 = stoi(argv[i + 1]);
        else if ("-cam2" == string(argv[i]) && i < argc - 1)
            _state.cam2 = stoi(argv[i + 1]);
        else if ("-out" == string(argv[i]) && i < argc - 1)
            _state.camOut = stoi(argv[i + 1]);
        else if ("-film" == string(argv[i]) && i < argc - 1)
            _state.currentFilm = string(argv[i + 1]);
        else if ("-frameNbr" == string(argv[i]) && i < argc - 1)
            _state.frameNbr = stoi(argv[i + 1]);
        else if ("-fps" == string(argv[i]) && i < argc - 1)
            _state.fps = stof(argv[i + 1]);
        else if ("-bgLearningTime" == string(argv[i]) && i < argc - 1)
            _state.bgLearningTime = stof(argv[i + 1]);
        ++i;
    }
}

/*************/
int main(int argc, char** argv)
{
    parseArguments(argc, argv);

    // Launch http server
    HttpServer server("127.0.0.1", "8080");
    auto requestHandler = server.getRequestHandler();
    thread serverThread = thread([&]() {
        server.run();
    });

    // Load films
    vector<FilmPlayer> films;
    films.emplace_back("./films/" + _state.currentFilm + "/", _state.frameNbr, 2, _state.fps);
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
    vector<int> camIndices {_state.cam1, _state.cam2};
    StereoCamera stereoCamera(camIndices, StereoCamera::StereoMode::CSBP);
    stereoCamera.loadConfiguration("intrinsics.yml", "extrinsics.yml");
    stereoCamera.setBgLearningTime(_state.bgLearningTime);

    // Prepare v4l2 loopback
    unique_ptr<V4l2Output> v4l2sink;

    // And the layer merger
    LayerMerger layerMerger;

    while(_state.run)
    {
        auto frameBegin = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // Do camera stuff
        stereoCamera.grab();

        if (stereoCamera)
        {
            stereoCamera.setWhiteBalance(_state.balanceRed, _state.balanceBlue);
            stereoCamera.compute();
            cv::Mat depthMask = stereoCamera.retrieveDepthMask();

            vector<cv::Mat> remappedFrames = stereoCamera.retrieveRemapped();

            unsigned int index = 0;
            for (auto& frame : remappedFrames)
            {
                string name = "Camera remapped" + to_string(index);
                cv::imshow(name, frame);
                index++;
            }
            cv::imshow("depthMask", depthMask);

            // Get current film frame
            auto frame = films[0].getCurrentFrame();
            auto frameMask = films[0].getCurrentMask();

            cv::Mat cameraMaskBG, cameraMaskFG;
            cv::threshold(depthMask, cameraMaskBG, _state.bgLimit, 255, cv::THRESH_BINARY);
            cv::threshold(depthMask, cameraMaskFG, _state.fgLimit, 255, cv::THRESH_BINARY);
            auto finalImage = layerMerger.mergeLayersWithMasks({frame[1], remappedFrames[0], frame[0], remappedFrames[0]},
                                                               {cameraMaskBG, frameMask[0], cameraMaskFG});

            //auto finalImage = layerMerger.mergeLayersWithMasks({frame[1], frame[0]},
            //                                                   {frameMask[0]});

            cv::Mat finalImageFlipped;
            cv::flip(finalImage, finalImageFlipped, 1);
            cv::imshow("Result", finalImageFlipped);

            // Write the result to v4l2
            if (_state.sendToV4l2)
            {
                if (!v4l2sink || finalImage.rows != v4l2sink->getHeight() || finalImage.cols != v4l2sink->getWidth())
                    v4l2sink = unique_ptr<V4l2Output>(new V4l2Output(finalImage.cols, finalImage.rows, "/dev/video" + to_string(_state.camOut)));
                if (*v4l2sink)
                    v4l2sink->writeToDevice(finalImage.data, finalImage.total() * finalImage.elemSize());
            }
        }
        else
        {
            auto rawFrames = stereoCamera.retrieve();
            int rawFrameIndex = 0;
            for (auto& frame : rawFrames)
                cv::imshow("Raw Frame " + to_string(rawFrameIndex++), frame);
        }

        // Handle HTTP requests
        pair<RequestHandler::Command, RequestHandler::ReturnFunction> message;
        while ((message = requestHandler->getNextCommand()).first.command != RequestHandler::CommandId::nop)
        {
            auto command = message.first;
            auto replyFunction = message.second;

            if (command.command == RequestHandler::CommandId::quit)
            {
                _state.run = false;
            }
            else if (command.command == RequestHandler::CommandId::record)
            {
                layerMerger.setSaveMerge(true, "/tmp/gifbox_result");
            }
            else if (command.command == RequestHandler::CommandId::stop)
            {
                layerMerger.setSaveMerge(false);
                _state.sendToV4l2 = false;
            }
            else if (command.command == RequestHandler::CommandId::start)
            {
                _state.sendToV4l2 = true;
            }

            message.second(true, {"Default reply"});
        }

        // Handle keyboard
        // TODO: more precise loop timing
        auto frameEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        auto frameDuration = static_cast<unsigned long>(1000 / _state.fps);
        auto sleepTime = std::min(1000ul, std::max(1ul, frameDuration - (frameEnd - frameBegin)));
        short key = cv::waitKey(sleepTime);
        switch (key)
        {
        default:
            break;
        case 27: // Escape
            _state.run = false;
            break;
        case 'c': // enable calibration
            stereoCamera.activateCalibration();
            break;
        case 'l': // show calibration lines
            stereoCamera.showCalibrationLines();
            break;
        case 's': // Save images to disk
            stereoCamera.saveToDisk();
            break;
        case 'u': // FG forward
            _state.fgLimit++;
            cout << "Foreground: " << _state.fgLimit << endl;
            break;
        case 'j': // FG backward
            _state.fgLimit--;
            cout << "Foreground: " << _state.fgLimit << endl;
            break;
        case 'i': // FG forward
            _state.bgLimit++;
            cout << "Background: " << _state.bgLimit << endl;
            break;
        case 'k': // FG backward
            _state.bgLimit--;
            cout << "Background: " << _state.bgLimit << endl;
            break;
        case 't': // WB Blue
            _state.balanceBlue += 0.05f;
            cout << "White balance red / blue : " << _state.balanceRed << " / " << _state.balanceBlue << endl;
            break;
        case 'g': // WB Blue
            _state.balanceBlue -= 0.05f;
            cout << "White balance red / blue : " << _state.balanceRed << " / " << _state.balanceBlue << endl;
            break;
        case 'r': // WB Red
            _state.balanceRed += 0.05f;
            cout << "White balance red / blue : " << _state.balanceRed << " / " << _state.balanceBlue << endl;
            break;
        case 'f': // WB Red
            _state.balanceRed -= 0.05f;
            cout << "White balance red / blue : " << _state.balanceRed << " / " << _state.balanceBlue << endl;
            break;
        }
    }

    server.stop();
    serverThread.join();

    return 0;
}

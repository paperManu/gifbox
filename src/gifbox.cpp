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

#include "gifbox.h"

using namespace std;

/*************/
void GifBox::parseArguments(int argc, char** argv)
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
GifBox::GifBox(int argc, char** argv)
{
    parseArguments(argc, argv);

    // Launch http server
    _httpServer = unique_ptr<HttpServer>(new HttpServer("127.0.0.1", "8080"));
    _httpServerThread = thread([&]() {
        _httpServer->run();
    });

    // Load films
    _films.emplace_back("./films/" + _state.currentFilm + "/", _state.frameNbr, 2, _state.fps);
    for (auto filmIt = _films.begin(); filmIt != _films.end();)
    {
        auto film = *filmIt;
        if (!film)
            filmIt = _films.erase(filmIt);
        else
            filmIt++;
    }

    if (_films.size() == 0)
        cout << "Could not load films. Exiting" << endl;
    else
        _films[0].start();

    // Load cameras
    vector<int> camIndices {_state.cam1, _state.cam2};
    _stereoCamera = unique_ptr<StereoCamera>(new StereoCamera(camIndices, StereoCamera::StereoMode::CSBP));
    _stereoCamera->loadConfiguration("intrinsics.yml", "extrinsics.yml");
    _stereoCamera->setBgLearningTime(_state.bgLearningTime);

    // And the layer merger
    _layerMerger = unique_ptr<LayerMerger>(new LayerMerger());
}

/*************/
GifBox::~GifBox()
{
}

/*************/
void GifBox::run()
{
    while(_state.run)
    {
        auto frameBegin = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // Do camera stuff
        _stereoCamera->grab();

        if (_stereoCamera)
        {
            _stereoCamera->setWhiteBalance(_state.balanceRed, _state.balanceBlue);
            _stereoCamera->compute();
            cv::Mat depthMask = _stereoCamera->retrieveDepthMask();

            vector<cv::Mat> remappedFrames = _stereoCamera->retrieveRemapped();
            if (remappedFrames.size() == 0)
                remappedFrames = _stereoCamera->retrieve();

            unsigned int index = 0;
            for (auto& frame : remappedFrames)
            {
                string name = "Camera remapped" + to_string(index);
                cv::imshow(name, frame);
                index++;
            }

            if (depthMask.rows > 0 && depthMask.cols > 0)
            {
                cv::imshow("depthMask", depthMask);

                if (_films.size() != 0)
                {
                    // Get current film frame
                    auto frame = _films[0].getCurrentFrame();
                    auto frameMask = _films[0].getCurrentMask();

                    cv::Mat cameraMaskBG, cameraMaskFG;
                    cv::threshold(depthMask, cameraMaskBG, _state.bgLimit, 255, cv::THRESH_BINARY);
                    cv::threshold(depthMask, cameraMaskFG, _state.fgLimit, 255, cv::THRESH_BINARY);
                    auto finalImage = _layerMerger->mergeLayersWithMasks({frame[1], remappedFrames[0], frame[0], remappedFrames[0]},
                                                                       {cameraMaskBG, frameMask[0], cameraMaskFG});

                    //auto finalImage = _layerMerger->mergeLayersWithMasks({frame[1], frame[0]},
                    //                                                   {frameMask[0]});

                    cv::Mat finalImageFlipped;
                    cv::flip(finalImage, finalImageFlipped, 1);
                    cv::imshow("Result", finalImageFlipped);

                    // Write the result to v4l2
                    if (_state.sendToV4l2)
                    {
                        if (!_v4l2Sink || finalImage.rows != _v4l2Sink->getHeight() || finalImage.cols != _v4l2Sink->getWidth())
                            _v4l2Sink = unique_ptr<V4l2Output>(new V4l2Output(finalImage.cols, finalImage.rows, "/dev/video" + to_string(_state.camOut)));
                        if (*_v4l2Sink)
                            _v4l2Sink->writeToDevice(finalImage.data, finalImage.total() * finalImage.elemSize());
                    }
                }
            }
        }
        else
        {
            auto rawFrames = _stereoCamera->retrieve();
            int rawFrameIndex = 0;
            for (auto& frame : rawFrames)
                cv::imshow("Raw Frame " + to_string(rawFrameIndex++), frame);
        }

        // Handle HTTP requests
        auto requestHandler = _httpServer->getRequestHandler();
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
                _layerMerger->setSaveMerge(true, "/tmp/gifbox_result");
            }
            else if (command.command == RequestHandler::CommandId::stop)
            {
                _layerMerger->setSaveMerge(false);
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
        auto sleepTime = std::max(1l, (long)frameDuration - ((long)frameEnd - (long)frameBegin));

        short key = cv::waitKey(sleepTime);
        processKeyEvent(key);
    }

    _httpServer->stop();
    _httpServerThread.join();

    return;
}

/*************/
void GifBox::processKeyEvent(short key)
{
    switch (key)
    {
    default:
        //cout << "Pressed key: " << key << endl;
        break;
    case 27: // Escape
        _state.run = false;
        break;
    case 'c': // enable calibration
        _stereoCamera->activateCalibration();
        break;
    case 'l': // show calibration lines
        _stereoCamera->showCalibrationLines();
        break;
    case 's': // Save images to disk
        _stereoCamera->saveToDisk();
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

/*************/
int main(int argc, char** argv)
{
    GifBox gifbox(argc, argv);
    gifbox.run();
    return 0;
}

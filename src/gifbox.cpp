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

#include <spawn.h>

using namespace std;

/*************/
void GifBox::parseArguments(int argc, char** argv)
{
    if (argc == 1)
    {
        cout << "Gifbox engine - Where the fun is assembled!" << endl;
        cout << "Basic usage: gifengine -film FILENAME -frameNbr FRAMENBR -fps FPS" << endl;
        cout << "Parameters:" << endl;
        cout << "  -film: specify the name of the directory in which the film is stored" << endl;
        cout << "  -frameNbr: set the number of frames for the given film" << endl;
        cout << "  -fps: set the framerate" << endl;
        cout << "  -maxRecordTime: set the maximum number of frames recorded" << endl;
        cout << "  -out: set the output v4l2 device, defaults to 0" << endl;
        exit(0);
    }
    for (int i = 1; i < argc;)
    {
        if ("-film" == string(argv[i]) && i < argc - 1)
        {
            _state.currentFilm = string(argv[i + 1]);
            ++i;
        }
        else if ("-frameNbr" == string(argv[i]) && i < argc - 1)
        {
            _state.frameNbr = stoi(argv[i + 1]);
            ++i;
        }
        else if ("-fps" == string(argv[i]) && i < argc - 1)
        {
            _state.fps = stof(argv[i + 1]);
            ++i;
        }
        else if ("-maxRecordTime" == string(argv[i]) && i < argc - 1)
        {
            _state.recordTimeMax = stoi(argv[i + 1]);
            ++i;
        }
        else if ("-out" == string(argv[i]) && i < argc - 1)
        {
            _state.camOut = stoi(argv[i + 1]);
            ++i;
        }
        else
        {
            cout << "Unrecognized argument: " << argv[i] << endl;
        }

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
        cout << "Could not load films." << endl;
    else
        _films[0].start();

    // Load camera
    _camera = unique_ptr<RgbdCamera>(new RgbdCamera());

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
        _camera->grab();

        if (_camera)
        {
            _camera->setWhiteBalance(_state.balanceRed, _state.balanceGreen, _state.balanceBlue);
            cv::Mat depthMask = _camera->retrieveDepthMask();

            auto rgbFrame = _camera->retrieveRGB();
            //cv::imshow("RGB Camera", rgbFrame);

            if (depthMask.rows > 0 && depthMask.cols > 0)
            {
                //cv::imshow("depthMask", depthMask);

                if (_films.size() != 0)
                {
                    // Get current film frame
                    auto frame = _films[0].getCurrentFrame();
                    auto frameMask = _films[0].getCurrentMask();

                    // If we just changed frame in the film, we save the previous merge result
                    bool recordEnded = false;
                    if (_films[0].hasChangedFrame())
                        recordEnded = _layerMerger->saveFrame();

                    cv::Mat cameraMaskBG, cameraMaskFG;
                    cv::threshold(depthMask, cameraMaskBG, _state.bgLimit, 255, cv::THRESH_BINARY_INV);
                    cv::threshold(depthMask, cameraMaskFG, _state.fgLimit, 255, cv::THRESH_BINARY_INV);
                    auto finalImage = _layerMerger->mergeLayersWithMasks({frame[1], rgbFrame, frame[0], rgbFrame},
                                                                       {cameraMaskBG, frameMask[0], cameraMaskFG});

                    cv::Mat finalImageFlipped;
                    cv::flip(finalImage, finalImageFlipped, 1);
                    cv::imshow("Result", finalImageFlipped);

                    // Write the result to v4l2
                    //if (_state.sendToV4l2)
                    //{
                        if (!_v4l2Sink || finalImage.rows != _v4l2Sink->getHeight() || finalImage.cols != _v4l2Sink->getWidth())
                            _v4l2Sink = unique_ptr<V4l2Output>(new V4l2Output(finalImage.cols, finalImage.rows, "/dev/video" + to_string(_state.camOut)));
                        if (*_v4l2Sink)
                        {
                            cv::Mat rgbImage;
                            cv::cvtColor(finalImage, rgbImage, cv::COLOR_BGR2RGB);
                            _v4l2Sink->writeToDevice(rgbImage.data, rgbImage.total() * rgbImage.elemSize());
                        }
                    //}
                }
            }
        }
        else
        {
            auto rgbFrame = _camera->retrieveRGB();
            cv::imshow("Raw Frame", rgbFrame);
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
                message.second(true, {"Default reply"});
            }
            else if (command.command == RequestHandler::CommandId::record)
            {
                _state.record = _layerMerger->isRecording();
                if (!_state.record)
                {
                    _layerMerger->setSaveMerge(true, "/tmp/gifbox_result", _state.recordTimeMax);
                    _state.record = true;
                }
                message.second(true, {"Default reply"});
            }
            else if (command.command == RequestHandler::CommandId::setFilm)
            {
                if (command.args.size() < 4)
                {
                    message.second(true, {"Need to specify film name, frame number and framerate"});
                }
                else
                {
                    auto filename = command.args[1].asString();
                    int frameNbr = command.args[2].asInt();
                    float frameRate = command.args[3].asFloat();
                    FilmPlayer film("./films/" + filename + "/", frameNbr, 2, frameRate);
                    if (film)
                    {
                        _films.clear();
                        _films.push_back(film);
                        _films[0].start();
                        _state.currentFilm = filename;
                        _state.frameNbr = frameNbr;
                        _state.fps = frameRate;
                        message.second(true, {"Success"});
                    }
                    else
                    {
                        message.second(true, {"Failed"});
                    }
                }
            }
            else if (command.command == RequestHandler::CommandId::stop)
            {
                _layerMerger->setSaveMerge(false);
                _state.sendToV4l2 = false;
                _state.record = false;
                message.second(true, {"Default reply"});
            }
            else if (command.command == RequestHandler::CommandId::start)
            {
                _state.sendToV4l2 = true;
                message.second(true, {"Default reply"});
            }
        }

        // Handle keyboard
        // TODO: more precise loop timing
        auto frameEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        long frameDuration = 33; // 33ms per frame, so 30fps
        auto sleepTime = std::max(1l, frameDuration - ((long)frameEnd - (long)frameBegin));

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
    case 32: // Space
        _state.record = _layerMerger->isRecording();
        if (!_state.record)
        {
            _layerMerger->setSaveMerge(true, "/tmp/gifbox_result", _state.recordTimeMax);
            _state.record = true;
            cv::waitKey(1000);
            //this_thread::sleep_for(chrono::seconds(1));
        }
        //else
        //{
        //    _layerMerger->setSaveMerge(false);
        //    _state.record = false;
        //}
        break;
    case 'c': // enable calibration
        _camera->activateCalibration();
        break;
    case 'l': // show calibration lines
        _camera->showCalibrationLines();
        break;
    case 's': // Save images to disk
        _camera->saveToDisk();
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
        cout << "White balance red / green / blue : " << _state.balanceRed << " / " << _state.balanceGreen << " / " << _state.balanceBlue << endl;
        break;
    case 'g': // WB Blue
        _state.balanceBlue -= 0.05f;
        cout << "White balance red / green / blue : " << _state.balanceRed << " / " << _state.balanceGreen << " / " << _state.balanceBlue << endl;
        break;
    case 'r': // WB Green
        _state.balanceGreen += 0.05f;
        cout << "White balance red / green / blue : " << _state.balanceGreen << " / " << _state.balanceGreen << " / " << _state.balanceBlue << endl;
        break;
    case 'f': // WB Green
        _state.balanceGreen -= 0.05f;
        cout << "White balance red / green / blue : " << _state.balanceGreen << " / " << _state.balanceGreen << " / " << _state.balanceBlue << endl;
        break;
    case 'e': // WB Red
        _state.balanceRed += 0.05f;
        cout << "White balance red / green / blue : " << _state.balanceRed << " / " << _state.balanceGreen << " / " << _state.balanceBlue << endl;
        break;
    case 'd': // WB Red
        _state.balanceRed -= 0.05f;
        cout << "White balance red / green / blue : " << _state.balanceRed << " / " << _state.balanceGreen << " / " << _state.balanceBlue << endl;
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

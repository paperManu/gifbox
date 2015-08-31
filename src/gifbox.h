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

#include "./filmPlayer.h"
#include "./httpServer.h"
#include "./layerMerger.h"
#include "./rgbdCamera.h"
#include "./stereoCamera.h"
#include "./v4l2output.h"
#include "./values.h"

/*************/
class GifBox
{
    public:
        GifBox(int argc, char** argv);
        ~GifBox();

        void run();

    private:
        struct State
        {
            bool run {true};
            bool sendToV4l2 {false};

            bool record {false};
            int recordTimeMax {120};
        
            int cam1 {1};
            int cam2 {2};
            int camOut {0};
        
            std::string currentFilm {"ALL_THE_RAGE"};
            int frameNbr {0};
            float fps {5.f};
        
            int fgLimit {30};
            int bgLimit {45};
        
            float bgLearningTime {0.001};
        
            float balanceRed {1.f};
            float balanceGreen {1.f};
            float balanceBlue {1.f};
        } _state;

        std::unique_ptr<HttpServer> _httpServer;
        std::thread _httpServerThread;
        std::vector<FilmPlayer> _films;
        //std::unique_ptr<StereoCamera> _stereoCamera;
        std::unique_ptr<RgbdCamera> _camera;
        std::unique_ptr<V4l2Output> _v4l2Sink;
        std::unique_ptr<LayerMerger> _layerMerger;

        void parseArguments(int argc, char** argv);
        void processKeyEvent(short key);
};

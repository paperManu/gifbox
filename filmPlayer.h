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
 * GifBox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GifBox.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FILMPLAYER_H
#define FILMPLAYER_H

#include <chrono>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#define PLANE_BASENAME "plan"
#define FRAME_BASENAME "Frame"

/*************/
class FilmPlayer
{
    public:
        FilmPlayer(std::string path, int frameNbr, int planeNbr, float fps = 10.f);

        explicit operator bool() const
        {
            if (_ready < 0)
                return false;
            else
                return true;
        }

        void start();
        std::vector<cv::Mat>& getCurrentFrame();

    private:
        std::string _path {};
        int _frameNbr {0};
        int _planeNbr {0};
        float _fps {10.f};

        bool _ready {false};
        std::vector<std::vector<cv::Mat>> _frames;
        std::chrono::milliseconds _startTime;
};

#endif

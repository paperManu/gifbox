/*
 *
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
 * along with blobserver.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef V4L2OUTPUT_H
#define V4L2OUTPUT_H

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include <linux/videodev2.h>

/*************/
class V4l2Output
{
    public:
        V4l2Output(int width, int height, std::string device = "/dev/video0");
        ~V4l2Output();

        explicit operator bool() const
        {
            if (_sink < 0)
                return false;
            else
                return true;
        }

        bool writeToDevice(void* data, size_t size);

    private:
        std::string _device {};
        int _sink {-1};
};

#endif

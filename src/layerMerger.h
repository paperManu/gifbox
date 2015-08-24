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

#ifndef LAYERMERGER_H
#define LAYERMERGER_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/*************/
class LayerMerger
{
    public:
        LayerMerger();

        // Layers from back to front, with one mask between each of them
        // Everything is resized to the size of the first layer
        cv::Mat mergeLayersWithMasks(const std::vector<cv::Mat>& layers, const std::vector<cv::Mat>& masks);
        void setSaveMerge(bool save, std::string basename = "", int maxRecordTime =  0);

        bool isRecording() {return _saveMergerResult;}

    private:
        cv::Mat _mergeResult;

        std::string _saveBasename {""};
        unsigned int _saveIndex {0};

        bool _saveMergerResult {false};
        unsigned int _saveImageIndex {0};

        unsigned int _maxRecordTime {0};
};

#endif

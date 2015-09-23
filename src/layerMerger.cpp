#include "layerMerger.h"

#include <iostream>
#include <limits>
#include <spawn.h>
#include <wait.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

/*************/
LayerMerger::LayerMerger()
{
    _maxRecordTime = numeric_limits<unsigned int>::max();

    _logoONF = cv::imread("logoONF.png", cv::IMREAD_COLOR);
    if (_logoONF.total() == 0)
        cout << "LayerMerger: could not load logo file logoONF.png" << endl;
}

/*************/
cv::Mat LayerMerger::mergeLayersWithMasks(const vector<cv::Mat>& layers, const vector<cv::Mat>& masks)
{
    if (layers.size() != masks.size() + 1)
    {
        cout << "LayerMerger: wrong number of layers and masks (" << layers.size() << " and " << masks.size() << ")" << endl;
        return {};
    }

    auto frameSize = layers[0].size();
    cv::Mat mergeResult = layers[0].clone();

    for (unsigned int i = 1; i < layers.size(); ++i)
    {
        cv::Mat tmpLayer = layers[i].clone();
        cv::Mat tmpAlpha = masks[i - 1];

        if (layers[i].size() != frameSize)
            cv::resize(layers[i], tmpLayer, frameSize, cv::INTER_LINEAR);
        if (masks[i - 1].size() != frameSize)
            cv::resize(masks[i - 1], tmpAlpha, frameSize, cv::INTER_LINEAR);

        cv::Mat alpha;
        cv::cvtColor(tmpAlpha, alpha, cv::COLOR_GRAY2BGR);

        tmpLayer = tmpLayer.mul(alpha, 1.0 / 255.0);

        for (int y = 0; y < mergeResult.rows; ++y)
            for (int x = 0; x < mergeResult.cols; ++x)
            {
                auto alphaValue = alpha.at<cv::Vec3b>(y, x)[0];
                if (alphaValue == 0)
                {
                    continue;
                }
                else if (alphaValue == 255)
                {
                    mergeResult.at<cv::Vec3b>(y, x)[0] = tmpLayer.at<cv::Vec3b>(y, x)[0];
                    mergeResult.at<cv::Vec3b>(y, x)[1] = tmpLayer.at<cv::Vec3b>(y, x)[1];
                    mergeResult.at<cv::Vec3b>(y, x)[2] = tmpLayer.at<cv::Vec3b>(y, x)[2];
                }
                else
                {
                    mergeResult.at<cv::Vec3b>(y, x)[0] = ((255 - alphaValue) * mergeResult.at<cv::Vec3b>(y, x)[0] + alphaValue * tmpLayer.at<cv::Vec3b>(y, x)[0]) / 255;
                    mergeResult.at<cv::Vec3b>(y, x)[1] = ((255 - alphaValue) * mergeResult.at<cv::Vec3b>(y, x)[1] + alphaValue * tmpLayer.at<cv::Vec3b>(y, x)[1]) / 255;
                    mergeResult.at<cv::Vec3b>(y, x)[2] = ((255 - alphaValue) * mergeResult.at<cv::Vec3b>(y, x)[2] + alphaValue * tmpLayer.at<cv::Vec3b>(y, x)[2]) / 255;
                }
            }
    }

    // Add the logo
    if (_logoONF.total() > 0)
    {
        cv::Mat tmpMat = _logoONF.clone();
        cv::Mat logo;
        if (tmpMat.size() != frameSize)
            cv::resize(tmpMat, logo, frameSize, cv::INTER_LINEAR);
        mergeResult += logo * 0.35;

        _mergeResult = mergeResult.clone();
    }

    if (_saveMergerResult)
    {
        // Add a record progress bar
        // The frame is inverted, we draw it from the other side
        cv::Mat layer(frameSize, CV_8UC3);
        auto lowerLeft = cv::Point(frameSize.width, frameSize.height - 16);
        auto upperRight = cv::Point(frameSize.width - (frameSize.width * _saveImageIndex) / _maxRecordTime, frameSize.height);
        cv::rectangle(layer, lowerLeft, upperRight, cv::Scalar(255, 64, 64), CV_FILLED);

        mergeResult += layer;
    }


    return mergeResult;
}

/*************/
bool LayerMerger::saveFrame()
{
    if (_mergeResult.total() == 0)
        return false;

    if (_saveMergerResult)
    {
        auto filename = getFilename();
        cv::imwrite(filename, _mergeResult, {cv::IMWRITE_PNG_COMPRESSION, 9});
        _saveImageIndex++;

        if (_saveImageIndex >= _maxRecordTime)
        {
            _saveMergerResult = false;
            _saveImageIndex = 0;
            convertSequenceToGif();
        }

        return true;
    }

    return false;
}

/*************/
void LayerMerger::setSaveMerge(bool save, string basename, int maxRecordTime)
{
    if (save)
        _saveIndex++;

    _saveMergerResult = save;
    _saveBasename = basename;
    _saveImageIndex = 0;

    if (maxRecordTime == 0)
        _maxRecordTime = numeric_limits<unsigned int>::max();
    else
        _maxRecordTime = maxRecordTime;
}

/*************/
string LayerMerger::getFilename()
{
    string filename;
    if (_saveImageIndex < 10)
        filename = _saveBasename + "_" + to_string(_saveIndex) + "_0" + to_string(_saveImageIndex) + ".png";
    else
        filename = _saveBasename + "_" + to_string(_saveIndex) + "_" + to_string(_saveImageIndex) + ".png";
    return filename;
}

/*************/
void LayerMerger::convertSequenceToGif()
{
    auto basename = "gifbox_result_" + to_string(_saveIndex);
    _lastRecordName = basename;
    string cmd = "convertToGif";
    char* argv[] = {(char*)"convertToGif", (char*)basename.c_str(), nullptr};

    int pid;
    posix_spawn(&pid, cmd.c_str(), nullptr, nullptr, argv, nullptr);
    //waitpid(pid, nullptr, 0);
}

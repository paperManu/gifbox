#include "layerMerger.h"

#include <iostream>
#include <limits>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

/*************/
LayerMerger::LayerMerger()
{
    _maxRecordTime = numeric_limits<unsigned int>::max();
}

/*************/
cv::Mat LayerMerger::mergeLayersWithMasks(const vector<cv::Mat>& layers, const vector<cv::Mat>& masks)
{
    if (layers.size() != masks.size() + 1)
    {
        cout << "LayerMerger: wrong number of layers and masks (" << layers.size() << " and " << masks.size() << ")" << endl;
        return {};
    }

    cv::Mat mergeResult = layers[0].clone();

    for (unsigned int i = 1; i < layers.size(); ++i)
    {
        cv::Mat tmpLayer = layers[i].clone();
        cv::Mat tmpAlpha = masks[i - 1];

        if (layers[i].size() != layers[0].size())
            cv::resize(layers[i], tmpLayer, layers[0].size(), cv::INTER_LINEAR);
        if (masks[i - 1].size() != layers[0].size())
            cv::resize(masks[i - 1], tmpAlpha, layers[0].size(), cv::INTER_LINEAR);

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

    _mergeResult = mergeResult;

    if (_saveMergerResult)
    {
        string filename = _saveBasename + "_" + to_string(_saveIndex) + "_" + to_string(_saveImageIndex) + ".png";
        cv::imwrite(filename, mergeResult, {cv::IMWRITE_PNG_COMPRESSION, 9});
        _saveImageIndex++;

        if (_saveImageIndex >= _maxRecordTime)
        {
            _saveMergerResult = false;
            _saveImageIndex = 0;
        }
    }

    return mergeResult;
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

#include "layerMerger.h"

#include <iostream>

using namespace std;

/*************/
cv::Mat LayerMerger::mergeLayersWithMasks(vector<cv::Mat> layers, vector<cv::Mat> masks)
{
    if (layers.size() != masks.size() + 1)
    {
        cout << "LayerMerger: wrong number of layers and masks (" << layers.size() << " and " << masks.size() << ")" << endl;
        return {};
    }

    cv::Mat mergeResult = layers[0].clone();

    for (unsigned int i = 1; i < layers.size(); ++i)
    {
        cv::Mat tmpLayer = layers[i];
        cv::Mat tmpMask = masks[i - 1];

        if (layers[i].size() != layers[0].size())
            cv::resize(layers[i], tmpLayer, layers[0].size(), cv::INTER_LINEAR);
        if (masks[i - 1].size() != layers[0].size())
            cv::resize(masks[i - 1], tmpMask, layers[0].size(), cv::INTER_LINEAR);

        tmpLayer.copyTo(mergeResult, tmpMask);
    }

    _mergeResult = mergeResult;

    return mergeResult;
}

/*************/
void LayerMerger::setSaveMerge(bool save, string basename)
{
    _saveMergerResult = save;
    _saveBasename = basename;
}

#include "filmPlayer.h"

#include <iostream>

using namespace std;

/*************/
FilmPlayer::FilmPlayer(string path, int frameNbr, int planeNbr, float fps)
{
    _path = path;
    _frameNbr = frameNbr;
    _planeNbr = planeNbr;
    _fps = fps;

    _frames.clear();
    for (int i = 1; i <= frameNbr; ++i)
    {
        vector<cv::Mat> frames;
        vector<cv::Mat> masks;

        for (int p = 1; p <= planeNbr; ++p)
        {
            string filename = path + "/" + string(PLANE_BASENAME) + to_string(p) + "/" + string(FRAME_BASENAME) + to_string(i) + ".png";
            cv::Mat frame = cv::imread(filename, cv::IMREAD_UNCHANGED);
            if (frame.data == nullptr)
            {
                cout << "FilmPlayer: could not load frame " << filename << ". Exiting." << endl;
                _frames.clear();
                _ready = false;
                return;
            }
            frames.emplace_back(frame);

            if (p < planeNbr)
            {
                // If there is no alpha channel, we create the mask from the white value
                if (frame.channels() < 4)
                {
                    cv::Mat gray;
                    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                    cv::Mat mask;
                    cv::threshold(gray, mask, 254, 255, cv::THRESH_BINARY_INV);
                    masks.emplace_back(mask);
                }
                else
                {
                    cv::Mat alpha(frame.size(), CV_8UC1);
                    cv::mixChannels(frame, alpha, {3, 0});
                    masks.emplace_back(alpha);

                    cv::cvtColor(frame, frames[frames.size() - 1], cv::COLOR_RGBA2RGB);
                }
            }
            else if (frame.channels() == 4)
            {
                cv::cvtColor(frame, frames[frames.size() - 1], cv::COLOR_RGBA2RGB);
            }
        }

        _frames.emplace_back(frames);
        _masks.emplace_back(masks);
    }

    if (_frames.size() && _frames.size() == _frameNbr)
    {
        _ready = true;
        cout << "FilmPlayer: successfully loaded film in path " << path << " with " << frameNbr << " frames and " << planeNbr << " planes." << endl;
    }
    else
    {
        _ready = false;
        cout << "FilmPlayer: error while loading film in path " << path << endl;
    }
}

/*************/
FilmPlayer::~FilmPlayer()
{
}

/*************/
void FilmPlayer::start()
{
    _startTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch());
}

/*************/
vector<cv::Mat>& FilmPlayer::getCurrentFrame()
{
    auto currentTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch());
    auto elapsed = (currentTime - _startTime).count();
    int frameIndex = static_cast<int>(elapsed * _fps / 1000.f) % _frameNbr;

    // Store whether we changed frame
    _frameChanged = (frameIndex != _lastIndex);

    _lastIndex = frameIndex;
    return _frames[frameIndex];
}

/*************/
bool FilmPlayer::hasChangedFrame()
{
    bool changed = _frameChanged;
    _frameChanged = false;
    return changed;
}

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
            cv::Mat frame = cv::imread(filename, cv::IMREAD_COLOR);
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
                cv::Mat gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                cv::Mat mask;
                cv::threshold(gray, mask, 254, 255, cv::THRESH_BINARY_INV);
                masks.emplace_back(mask);
            }
        }

        _frames.emplace_back(frames);
        _masks.emplace_back(masks);
    }

    _ready = true;
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
    _lastIndex = frameIndex;
    return _frames[frameIndex];
}

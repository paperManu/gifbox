// Gifomaton draft stereoCamera class

#include <chrono>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

/*************/
class StereoCamera
{
    public:
        StereoCamera();
        StereoCamera(std::vector<int> camIndices);
        ~StereoCamera();

        void computeDisparity();
        bool grab();
        bool isReady();
        bool loadConfiguration(std::string intrinsic, std::string extrinsic);
        std::vector<cv::Mat>& retrieve();
        std::vector<cv::Mat>& retrieveRemapped();
        cv::Mat retrieveDisparity() {return _disparityMap.clone();}
        void saveToDisk();

    private:
        std::chrono::system_clock::time_point _startTime;
        std::vector<cv::VideoCapture> _cameras;
        std::vector<cv::Mat> _frames;
        std::vector<cv::Mat> _remappedFrames;

        std::vector<std::vector<cv::Mat>> _rmaps;
        cv::Mat _disparityMap;

        //std::shared_ptr<cv::StereoBM> _stereoMatcher;
        std::shared_ptr<cv::StereoSGBM> _stereoMatcher;

        unsigned int _captureIndex {0};

        struct Calibration
        {
            cv::Mat cameraMatrix;
            cv::Mat distCoeffs;
            cv::Mat rotation;
            cv::Mat position;
        };
        std::vector<Calibration> _calibrations;
        bool _calibrationLoaded {false};

        void init(std::vector<int> camIndices);
};

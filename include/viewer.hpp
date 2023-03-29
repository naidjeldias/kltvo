#include <thread>
#include "tracking.h"
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/pangolin.h>
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>

class Viewer 
{

public:
    Viewer(const string &strSettingPath, Tracking* tracker);
    void run();
    void shutdown();
private:
    Tracking* trackerPtr_;
    bool finishRequested_;
    int imageWidth_, imageHeight_;
    float updateRate_, viewpointX_, viewpointY_, viewpointZ_, viewpointF_;
    std::mutex data_buffer_mutex_;
    std::vector<cv::Mat> cameraPoses_;
    cv::Mat computeGlobalPose();
    void computeOpenGLCameraMatrix(const cv::Mat& cameraPose, pangolin::OpenGlMatrix& Twc);
    void drawTrajectory();
    void renderCamera(const pangolin::OpenGlMatrix& camtMat);
    cv::Mat convertToOpenGLFrame(const cv::Mat& camMat);
};

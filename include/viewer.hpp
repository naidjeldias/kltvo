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
    Viewer(const string &strSettingPath);
    ~Viewer();
    void run();
    void shutdown();
    void update(Tracking* tracker);
private:
    int trackingState_;
    bool finishRequested_;
    int imageWidth_, imageHeight_;
    float updateRate_, viewpointX_, viewpointY_, viewpointZ_, viewpointF_;
    std::mutex data_buffer_mutex_;
    std::vector<cv::Mat> cameraPoses_;
    cv::Mat imLeft0_;
    std::vector<cv::Point2f> features_, keypoints_;

    cv::Mat computeGlobalPose();
    void computeOpenGLCameraMatrix(const cv::Mat& cameraPose, pangolin::OpenGlMatrix& Twc);
    void drawTrajectory();
    void renderCamera(const pangolin::OpenGlMatrix& camtMat);
    cv::Mat convertToOpenGLFrame(const cv::Mat& camMat);
    void drawPointsImage(cv::Mat &im, const std::vector<cv::Point2f> &pts, cv::Scalar color);
};

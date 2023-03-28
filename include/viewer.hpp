#include <thread>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/pangolin.h>
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>

class Viewer 
{

public:
    Viewer();
    void run();
    void setCameraPoses(const std::vector<cv::Mat>& cameraPoses);
    void shutdown();
private:
    bool finishRequested_;
    std::mutex data_buffer_mutex_;
    std::vector<cv::Mat> cameraPoses_;
    cv::Mat computeGlobalPose();
    void computeOpenGLCameraMatrix(const cv::Mat& cameraPose, pangolin::OpenGlMatrix& Twc);
    void drawTrajectory();
    void renderCamera(const pangolin::OpenGlMatrix& camtMat);
};

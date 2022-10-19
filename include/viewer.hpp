#include "tracking.h"
#include <pangolin/pangolin.h>
#include <pangolin/pangolin.h>
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>

class Viewer 
{

public:
    Viewer(Tracking* trackerPtr);
    void run();
    cv::Mat global_pose_ = cv::Mat::eye(4,4,CV_32F);
private:
    std::mutex tracker_;
    Tracking* trackerPtr_;
    cv::Mat computeGlobalPose();
    void computeOpenGLCameraMatrix(const cv::Mat& cameraPose, pangolin::OpenGlMatrix& Twc);
};

//
// Created by nigel on 21/01/19.
//

#ifndef KLTVO_TRACKING_H
#define KLTVO_TRACKING_H

#include "utils.h"
#include "opencv2/features2d/features2d.hpp"

class Tracking{

public:

    Tracking();

    cv::Mat K;
    double  baseline;
    cv::Mat P1, P2;
    bool initPhase;


    void start(const cv::Mat &imLeft, const cv::Mat &imRight);

private:

    //----------local mapping
    int max_iter_3d         = 10;
    double th_3d            = 0.5;
    cv::Mat imLeft0, imRight0;

    void bucketFeatureExtraction (cv::Mat &image, cv::Size block, std::vector<cv::KeyPoint> &keypoints);
    void localMapping (const std::vector<cv::Point2d> &pts_l, const std::vector<cv::Point2d> &pts_r,
                       std::vector<cv::Point3d> &pts3D, const std::vector<cv::DMatch> &macthes);
    void stereoMatching(std::vector<cv::Point2f>& pts_l, std::vector<cv::Point2f>& pts_r, const cv::Mat& imLeft,
                        const cv::Mat& imRight, const std::vector<bool>& inliers,  std::vector<cv::DMatch> &matches);


};

#endif //KLTVO_TRACKING_H

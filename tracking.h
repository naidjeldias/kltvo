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


    bool once;


    void start(cv::Mat &imLeft, cv::Mat &imRight);

private:

    void bucketFeatureExtraction (cv::Mat &image, cv::Size block, std::vector<cv::KeyPoint> &keypoints);


};

#endif //KLTVO_TRACKING_H

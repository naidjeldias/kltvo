//
// Created by nigel on 06/12/18.
//

#ifndef RANSAC_EIGHTPOINT_H
#define RANSAC_EIGHTPOINT_H
#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>
#include "utils.h"

using namespace cv;

namespace kltvo
{
class EightPoint{

public:

    EightPoint(double probability, int minSet, int maxIteration, double maxError);
    //----------------Ransac Parameters
    std::vector<int> generateRandomIndices(const unsigned long &maxIndice, const unsigned int &vecSize);
    void setRansacParameters(double probability, int minSet, int maxIteration, double maxError);
    double ransacProb, ransacTh;
    int ransacMaxIt, ransacNumit;
    unsigned int ransacMinSet;
    void operator() (const std::vector<Point2f> &kpt_l, const std::vector<Point2f> &kpt_r,
                                      std::vector<DMatch> &finalMatches, std::vector<bool> &inliers2D, bool normalize,
                                      int method, cv::Mat &bestFmat);
    //using normalized 8-point algorithm
    cv::Mat computeFundamentalMatrix(const std::vector<Point2f> &kpt_l, const std::vector<Point2f> &kpt_r,
                                     const std::vector<int> &indices, const cv::Mat &leftScalingMat, const cv::Mat &rightScalingMat, bool normalize);

    //normalize data before compute fundamental matrix - translation and scaling of each umage so that
    //the centroid of the reference points is at the origin of the coordinates and the RMS distance from the origin is equal to sqrt(2)
    void computeMatNormTransform(const std::vector<Point2f> &kpt_l, const std::vector<Point2f> &kpt_r, unsigned long nPts, cv::Mat &leftScalingMat, cv::Mat &rightScalingMat);
    double sampsonError(cv::Mat fmat, cv::Mat left_pt, cv::Mat right_pt);
    double euclideanDist(const cv::Point2d &p, const cv::Point2d &q);

    int getRansacNumit();


};

} // namespace kltvo

#endif //RANSAC_EIGHTPOINT_H

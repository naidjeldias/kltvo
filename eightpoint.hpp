//
// Created by nigel on 06/12/18.
//

#ifndef RANSAC_EIGHTPOINT_H
#define RANSAC_EIGHTPOINT_H
#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>

using namespace cv;

class EightPoint{

public:

    EightPoint();
    //----------------Ransac Parameters
    std::vector<int> generateRandomIndices(const unsigned long &maxIndice, const int &vecSize);
    void setRansacParameters(double probability, int minSet, int maxIteration, double maxError);
    double ransacProb, ransacTh;
    int ransacMinSet, ransacMaxIt;

    cv::Mat ransacEightPointAlgorithm(const std::vector<Point2f> &kpt_l, const std::vector<Point2f> &kpt_r,
                                      std::vector<DMatch> &finalMatches, std::vector<bool> &inliers2D, bool normalize, int method);
    //using normalized 8-point algorithm
    cv::Mat computeFundamentalMatrix(const std::vector<Point2f> &kpt_l, const std::vector<Point2f> &kpt_r,
                                     const std::vector<int> &indices, const cv::Mat &leftScalingMat, const cv::Mat &rightScalingMat, bool normalize);

    //normalize data before compute fundamental matrix - translation and scaling of each umage so that
    //the centroid of the reference points is at the origin of the coordinates and the RMS distance from the origin is equal to sqrt(2)
    void computeMatNormTransform(const std::vector<Point2f> &kpt_l, const std::vector<Point2f> &kpt_r, unsigned long nPts, cv::Mat &leftScalingMat, cv::Mat &rightScalingMat);
    double sampsonError(cv::Mat fmat, cv::Mat left_pt, cv::Mat right_pt);
    void drawEpLines(const std::vector<Point2f> &pts_l, const std::vector<Point2f> &pts_r, const cv::Mat &F,
                     const std::vector<bool> &inliers, int rightFlag, const cv::Mat &image, const cv::Mat &image1,
                     const std::vector<cv::DMatch> &matches);

    void drawMatches_(const cv::Mat &left_image, const cv::Mat &right_image, const std::vector<Point2f> &kpts_l,
                      const std::vector<Point2f> &kpts_r, const std::vector<cv::DMatch> &matches, bool hold);

    double euclideanDist(const cv::Point2d &p, const cv::Point2d &q);


};



#endif //RANSAC_EIGHTPOINT_H

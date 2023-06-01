//
// Created by nigel on 21/01/19.
//

#ifndef KLTVO_UTILS_H
#define KLTVO_UTILS_H


#include <opencv2/opencv.hpp>
#include<Eigen/Dense>


using namespace std;
namespace kltvo
{
namespace utils
{
void load_camCalib_yaml(string path, cv::Mat &K, double &baseline);

std::vector<double> getQuaternion(cv::Mat &R);
std::vector<float> mRot2Quat(const cv::Mat& m);

void saveTrajectoryKitti(const string &filename, std::list<cv::Mat> &relativeFramePoses);
void saveTrajectoryEuroc(const string &filename, std::list<cv::Mat> &relativeFramePoses, std::vector<double> frameTimeStamps);
} // namespace utils
} // namespace kltvo

#endif //KLTVO_UTILS_H

//
// Created by nigel on 21/01/19.
//

#ifndef KLTVO_UTILS_H
#define KLTVO_UTILS_H


#include <opencv2/opencv.hpp>


using namespace std;

void load_camCalib_yaml(string path, cv::Mat &K, double &baseline);

std::vector<double> getQuaternion(cv::Mat &R);
std::vector<float> mRot2Quat(const cv::Mat& m);


#endif //KLTVO_UTILS_H

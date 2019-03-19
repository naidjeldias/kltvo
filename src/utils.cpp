//
// Created by nigel on 21/01/19.
//

#include "utils.h"


void load_camCalib_yaml(string path, cv::Mat &K, double &baseline) {

    cv::FileStorage fsSettings(path, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["Camera.fx"] >> K.at<double>(0,0);
    fsSettings["Camera.fy"] >> K.at<double>(1,1);
    fsSettings["Camera.cx"] >> K.at<double>(0,2);
    fsSettings["Camera.cy"] >> K.at<double>(1,2);
    fsSettings["Camera.bf"] >> baseline;

    baseline = baseline / K.at<double>(0,0);

    //std::cout << K << std::endl;
//    std::cout << baseline << std::endl;

}
#include <iostream>
#include <chrono>
#include "tracking.h"
#include "utils.h"

using namespace std;

int main() {

    //kitti dataset
//    string path_data    = string("/media/nigel/Dados/Documents/Projetos/CLionProjects/kltVO/kitti");
//    string path_left    = string ("/video_0.avi");
//    string path_right   = string ("/video_1.avi");


    //full kitti dataset
    string path_data = string("/media/nigel/Dados/Documents/Projetos/KITTI_DATASET/dataset/sequences/00");
    string path_left    = string ("/image_0/%06d.png");
    string path_right   = string ("/image_1/%06d.png");


    Tracking tracking;

    string path_calib   = string("/media/nigel/Dados/Documents/Projetos/CLionProjects/kltVO/kitti/KITTI00-02.yaml");
    tracking.K           = cv::Mat::eye(3,3, CV_64F);

    load_camCalib_yaml(path_calib, tracking.K, tracking.baseline);

    tracking.P1 = cv::Mat::eye(4,4, CV_32F);
    tracking.P2 = cv::Mat::eye(4,4, CV_32F);

    tracking.K.convertTo(tracking.K, CV_32F);

    tracking.K.copyTo(tracking.P1.rowRange(0,3).colRange(0,3));
    tracking.K.copyTo(tracking.P2.rowRange(0,3).colRange(0,3));

    tracking.P2.at<float>(0,3) = - tracking.baseline * tracking.K.at<float>(0,0);

    //std::cout << tracking.P1 << std::endl;
    //std::cout << tracking.P2 << std::endl;
    //std::cout << tracking.K << std::endl;

    cv::VideoCapture left_vd, right_vd;

    bool isleft_vd  = left_vd.open(path_data+path_left);
    bool isright_vd = right_vd.open(path_data+path_right);

    int count = 0;
    while( isleft_vd && isright_vd /* && count < 2 */){

        cv::Mat imleft, imright;

        left_vd.read(imleft);
        right_vd.read(imright);

        if(imleft.empty() || imright.empty())
            break;

        if(imleft.channels() == 3)
            cvtColor(imleft, imleft, cv::COLOR_RGB2GRAY);
        if(imright.channels() == 3)
            cvtColor(imright, imright, cv::COLOR_RGB2GRAY);

        auto startTime = std::chrono::steady_clock::now();

        tracking.start(imleft,imright);

        auto endTime = std::chrono::steady_clock::now();

        std::cout << "Time elapsed: "<< std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()
                  << " ms" << std::endl;

        cv::imshow("Left Frame", imleft);
//        cv::imshow("Right Frame", imright);
//
////         Press  ESC on keyboard to exit
        char c=(char) cv::waitKey(30);
        if(c==27)
            break;
        count ++;

    }

    left_vd.release();
    right_vd.release();

    cv::destroyAllWindows();

    return 0;
}
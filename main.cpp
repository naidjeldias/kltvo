#include <iostream>
#include <chrono>
#include "tracking.h"
#include "utils.h"


class time_point;

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

    tracking.P1 = cv::Mat::eye(3,4, CV_64F);
    tracking.P2 = cv::Mat::eye(3,4, CV_64F);


    tracking.fu = tracking.K.at<double>(0,0);
    tracking.uc = tracking.K.at<double>(0,2);
    tracking.fv = tracking.K.at<double>(1,1);
    tracking.vc = tracking.K.at<double>(1,2);


    tracking.K.copyTo(tracking.P1.rowRange(0,3).colRange(0,3));
    tracking.K.copyTo(tracking.P2.rowRange(0,3).colRange(0,3));


    tracking.P2.at<double>(0,3) = - tracking.baseline * tracking.K.at<double>(0,0);

//    std::cout << tracking.fu << std::endl;
//    std::cout << tracking.fv << std::endl;
//    std::cout << tracking.uc << std::endl;

    cv::VideoCapture left_vd, right_vd;

    bool isleft_vd  = left_vd.open(path_data+path_left);
    bool isright_vd = right_vd.open(path_data+path_right);

    int count = 0;
    while( isleft_vd && isright_vd && count < 4  ){

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
        std::cout << "Frame: "<< count <<  std::endl;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
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

//    tracking.myfile.close();
    tracking.f.close();

    left_vd.release();
    right_vd.release();

    cv::destroyAllWindows();

    return 0;
}
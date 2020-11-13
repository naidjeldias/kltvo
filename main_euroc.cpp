//
// Created by nigel on 15/05/19.
//
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <zconf.h>
#include "tracking.h"

void LoadImages(const string &strPathToSequence, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{

    string strPathLeft  = strPathToSequence + "/cam0/data";
    string strPathRight = strPathToSequence + "/cam1/data";

    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            double t;
            string stamp;
//            ss >> stamp;
//            std::cout << stamp << std::endl;
            ss >> t;
            vTimeStamps.push_back(t/1e9);

        }
    }
}

int main(){

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimeStamp;
    string path_data  = string("../../EuRoc_Dataset/mav0");
    //change de sequence txt in order to use others sequences
    string path_times = string("euroc/times/MH01.txt");

    LoadImages(path_data, path_times, vstrImageLeft, vstrImageRight, vTimeStamp);

    if(vstrImageLeft.empty() || vstrImageRight.empty())
    {
        cerr << "ERROR: No images in provided path." << endl;
        return 1;
    }

    if(vstrImageLeft.size()!=vstrImageRight.size())
    {
        cerr << "ERROR: Different number of left and right images." << endl;
        return 1;
    }

    // Read rectification parameters
    string path_calib   = string("euroc/EuRoC.yaml");
    string path_config  = string("config/euroc.yaml");


    cv::FileStorage fsSettings(path_calib, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
       rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return -1;
    }


    cv::Mat M1l,M2l,M1r,M2r;
    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l,cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r,cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);

    double fu, fv, uc, vc, bf;

    std::cout << "Pl: " << P_l << std::endl;
    std::cout << "Pr: " << P_r << std::endl;


    fu = P_l.at<double>(0,0);
    fv = P_l.at<double>(1,1);
    uc = P_l.at<double>(0,2);
    vc = P_l.at<double>(1,2);

    bf = -P_r.at<double>(0,3);

    Tracking tracking(path_config, fu, fv, uc, vc, bf);

    const int nImages = vstrImageLeft.size();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    int current_ni;
    for(int ni=0; ni<nImages; ni++)
//    for(int ni=0; ni<4; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],IMREAD_UNCHANGED);

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

        if(imRight.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageRight[ni]) << endl;
            return 1;
        }

        cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);

        double tframe = vTimeStamp[ni];


        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        tracking.start(imLeftRect,imRightRect, tframe);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimeStamp[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimeStamp[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);

        current_ni = ni;

        cv::imshow("Left Frame", imLeft);
        char c=(char) cv::waitKey(1);
        if(c==27)
            break;

    }

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    float meanTime = totaltime/current_ni;
    cout << "-------" << endl << endl;
    cout << "mean tracking time: " << totaltime/current_ni << endl;


    string resultFile = "Euroc_MH01_KLTVO.txt";
//    tracking.saveTrajectoryKitti("results/euroc/"+resultFile);
    tracking.saveTrajectoryEuroc("results/euroc/"+resultFile);
#if LOG
    tracking.saveStatistics("stats/euroc/Euroc_MH01_STATS.csv", meanTime, true);

#endif
    cv::destroyAllWindows();

    return 0;
}
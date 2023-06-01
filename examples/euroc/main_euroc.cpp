//
// Created by nigel on 15/05/19.
//
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "tracking.h"
#include <unistd.h>
#include "viewer.hpp"
#include <yaml-cpp/yaml.h>

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

int main(int argc, char *argv[]){

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimeStamp;

    string seq = "MH01";

    //    string path_data  = string("../../EuRoc_Dataset/MH01/mav0");
    string path_data  = "../../EuRoc_Dataset/"+seq+"/mav0";

    if (argc == 3){
        path_data = argv[1];
        seq       = argv[2];
        cout << "Using sequence " << seq << " on path: " << path_data << endl;
    }else
    {
        cout << "Usage: ./stereo_euroc <SEQUENCE_PATH> <SEQUENCE_ID>" << endl;
        cout << "Example: ./stereo_euroc ~/MH_02_easy/mav0/ MH02" << endl;
        return 0;
    }

    ifstream file(path_data);
    if(!file)
    {
        cout << path_data << " path does not exist" << endl;
        return 0;
    }

    string resultPath = "examples/euroc/results/";
//    string resultFile = "Euroc_MH01_KLTVO.txt";
    string resultFile = "Euroc_" + seq + "_KLTVO.txt";


    //change de sequence txt in order to use others sequences
//    string path_times = string("euroc/times/MH    01.txt");
    string path_times = "examples/euroc/times/"+seq+".txt";

    string statsPath = "examples/euroc/stats/";
    string statsFile = "Euroc_" + seq + "_STATS.csv";

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

    string path_config  = string("examples/euroc/config/euroc.yaml");
    bool viz = true;
    YAML::Node odometry_params = YAML::LoadFile(path_config);
    viz = odometry_params["Viewer.enabled"].as<bool>();
    kltvo::Tracking* trackerPtr = new kltvo::Tracking(odometry_params);
    
    // Read rectification parameters
    string path_calib   = string("examples/euroc/calib/EuRoC.yaml");
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

    fu = P_l.at<double>(0,0);
    fv = P_l.at<double>(1,1);
    uc = P_l.at<double>(0,2);
    vc = P_l.at<double>(1,2);

    bf = -P_r.at<double>(0,3);
    
    trackerPtr->setCalibrationParameters(fu, fv, uc, vc, bf);

    // starting visualizer thread
    kltvo::Viewer* viewer_;
    std::thread* viewer_thd_;
    if (viz)
    {
        viewer_ = new kltvo::Viewer(path_config);
        viewer_thd_ = new thread(&kltvo::Viewer::run, viewer_);
    }


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
   //for(int ni=0; ni<1644; ni++)
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

        trackerPtr->start(imLeftRect,imRightRect, tframe);
        if (viz)
            viewer_->update(trackerPtr);

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



//    trackerPtr->saveTrajectoryKitti("results/euroc/"+resultFile);
    trackerPtr->saveTrajectoryEuroc(resultPath+resultFile);
#if LOG
    trackerPtr->saveStatistics(statsPath+statsFile, meanTime, true);

#endif
    

    if(viz)
    {
        cv::waitKey(0);
        viewer_->shutdown();
        viewer_thd_->join();
        delete viewer_thd_;
        delete viewer_;
    }
    delete trackerPtr;
    return 0;
}

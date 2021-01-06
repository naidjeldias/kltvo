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


void LoadParameters (const string &strSettingPath, int &frameGridRows, int &frameGridCols,  double &maxDisp, double &minDisp, 
                    double &thDepth, double &sadMinValue, double &halfBlockSize, int &winSize, int &pyrMaxLevel, 
                    int &nFeatures, float &fScaleFactor, int &nLevels, int &fIniThFAST, int &fMinThFAST,  
                    double &ransacProb, int &ransacMinSet, int &ransacMaxIt, double &ransacTh, int &max_iter_3d, double &th_3d, 
                    double &ransacProbGN, double &ransacThGN, int &ransacMinSetGN, int &ransacMaxItGN, double &minIncTh, 
                    int &maxIteration, int &finalMaxIteration, bool &reweigh, double &adjustValue)
{

    cv::FileStorage fsSettings(strSettingPath, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }


    frameGridRows = fsSettings["FeaturExtrac.frameGridRows"];
    frameGridCols = fsSettings["FeaturExtrac.frameGridCols"];

    minDisp         = fsSettings["Disparity.mindisp"];
    maxDisp         = fsSettings["Disparity.maxdisp"];

    thDepth         = fsSettings["ThDepth"];
    sadMinValue     = fsSettings["SAD.minValue"];
    halfBlockSize   = fsSettings["SAD.winHalfBlockSize"];

    winSize         = fsSettings["KLT.winSize"];
    pyrMaxLevel     = fsSettings["KLT.pyrMaxLevel"];

    nFeatures       = fsSettings["ORBextractor.nFeatures"];
    fScaleFactor    = fsSettings["ORBextractor.scaleFactor"];
    nLevels         = fsSettings["ORBextractor.nLevels"];
    fIniThFAST      = fsSettings["ORBextractor.iniThFAST"];
    fMinThFAST      = fsSettings["ORBextractor.minThFAST"];

    ransacProb      = fsSettings["EightPoint.ransacProb"];
    ransacMinSet    = fsSettings["EightPoint.ransacSet"];
    ransacMaxIt     = fsSettings["EightPoint.ransacMaxInt"];
    ransacTh        = fsSettings["EightPoint.ransacTh"];

    max_iter_3d     = fsSettings["Triangulation.maxIt"];    // max iteration for 3D estimation
    th_3d           = fsSettings["Triangulation.reproTh"];  // max reprojection error for 3D estimation

    ransacProbGN          = fsSettings["GN.ransacProb"];
    ransacThGN            = fsSettings["GN.ransacTh"];        // th for RANSAC inlier selection using reprojection error
    ransacMinSetGN        = fsSettings["GN.ransacMinSet"];
    ransacMaxItGN         = fsSettings["GN.ransacMaxIt"];     // RANSAC iteration for pose estimation
    minIncTh            = 10E-5;                            // min increment for pose optimization
    maxIteration        = fsSettings["GN.maxIt"];           // max iteration for minimization into RANSAC routine
    finalMaxIteration   = fsSettings["GN.finalMaxIt"];      // max iterations for minimization final refinement
    reweigh             = true;                             // reweight in optimization
    adjustValue         = fsSettings["GN.weightAdjustVal"];

}
int main(int argc, char *argv[]){

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimeStamp;

    string seq = "MH01";

    //    string path_data  = string("../../EuRoc_Dataset/MH01/mav0");
    string path_data  = "../../EuRoc_Dataset/"+seq+"/mav0";

    if(argc <= 1){
        cout << "No argument, the default path will be used. Dataset Path: "<< path_data << endl;
    }
    else if (argc == 3){
        path_data = argv[1];
        seq       = argv[2];
        cout << "Using sequence" << seq << "on path: " << path_data << endl;
    }else
    {
        cout << "Usage: <SEQUENCE_PATH> <SEQUENCE_ID>" << endl;
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

    int frameGridRows, frameGridCols, winSize, pyrMaxLevel, nFeatures, nLevels, fIniThFAST, fMinThFAST,
        ransacMinSet, ransacMaxIt, max_iter_3d, ransacMinSetGN, ransacMaxItGN, maxIteration, finalMaxIteration;
    double maxDisp, minDisp, thDepth, sadMinValue, halfBlockSize, ransacProb,ransacTh, th_3d, ransacProbGN,
           ransacThGN, minIncTh, adjustValue;
 
    float fScaleFactor; 
    bool reweigh;

    string path_config  = string("examples/euroc/config/euroc.yaml");

    LoadParameters(path_config, frameGridRows, frameGridCols,  maxDisp, minDisp, thDepth, sadMinValue, halfBlockSize, 
                winSize, pyrMaxLevel, nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, ransacProb, ransacMinSet, 
                ransacMaxIt, ransacTh, max_iter_3d, th_3d, ransacProbGN, ransacThGN, ransacMinSetGN, ransacMaxItGN, minIncTh, 
                maxIteration, finalMaxIteration, reweigh, adjustValue);
    Tracking tracking(path_config);
    

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
    
    tracking.setCalibrationParameters(fu, fv, uc, vc, bf);

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



//    tracking.saveTrajectoryKitti("results/euroc/"+resultFile);
    tracking.saveTrajectoryEuroc(resultPath+resultFile);
#if LOG
    tracking.saveStatistics(statsPath+statsFile, meanTime, true);

#endif
    cv::destroyAllWindows();

    return 0;
}

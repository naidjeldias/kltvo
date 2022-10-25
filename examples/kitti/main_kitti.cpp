#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "tracking.h"
#include "utils.h"
#include "viewer.hpp"
#include <unistd.h>


class time_point;

using namespace std;

cv::Mat global_pose = cv::Mat::eye(4,4,CV_32F);

void draw_trajectory(const cv::Mat &current_pose, cv::Mat &image)
{
    //Compute global pose
    //Compute the inverse of relative pose estimation inv(current_pose) = [R' | C]
    //where C = -1 * R' * t

    cv::Mat R = current_pose.rowRange(0,3).colRange(0,3);
    cv::Mat t = current_pose.col(3).rowRange(0,3);
    
    cv::Mat Rt  = R.t();
    cv::Mat C   = -1 * Rt * t; 
    
    cv::Mat inv_pose = cv::Mat::eye(4,4,CV_32F);
    Rt.copyTo(inv_pose.rowRange(0,3).colRange(0,3));
    C.copyTo(inv_pose.rowRange(0,3).col(3));

    global_pose = global_pose * inv_pose;

    int x = int(global_pose.at<float>(0,3)) + 300;;
    int z = int(global_pose.at<float>(2,3)) + 100;;
    
    cv::circle(image, Point(x, z) ,1, CV_RGB(255,0,0), 2);
    rectangle( image, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), cv::FILLED);
    char text[100];
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", global_pose.at<float>(0,3), global_pose.at<float>(1,3), global_pose.at<float>(2,3));
    putText(image, text, cv::Point (10, 50), FONT_HERSHEY_PLAIN, 1, Scalar::all(255), 1, 8);

}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps){
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
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

int main(int argc, char *argv[]) {

    // Retrieve paths to images
    //full kitti dataset
    string seq = "03";

    cout << endl << "-------" << endl;

    string path_data = string("../../KITTI_DATASET/dataset/sequences/"+seq);

    if (argc == 3){
        path_data = argv[1];
        seq       = argv[2];
        cout << "Using sequence" << seq << "on path: " << path_data << endl;
    }else
    {
        cout << "Usage: ./stereo_kitti <SEQUENCE_PATH> <SEQUENCE_ID>" << endl;
        cout << "Example: ./stereo_kitti ~/dataset/sequences/00/ 00" << endl;
        return 0;
    }

    ifstream file(path_data);
    if(!file)
    {
        cout << path_data << " path does not exist" << endl;
        return 0;
    }

    string resultPath = "examples/kitti/results/";
    string resultFile = "KITTI_" + seq + "_KLTVO.txt";

    string statsPath = "examples/kitti/stats/";
    string statsFile = "KITTI_" + seq + "_STATS.csv";


    
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(path_data, vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence "<< seq << "..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    string yamlFile = "KITTI" + seq + ".yaml";
    string path_calib       = string("examples/kitti/calib/"+yamlFile);
    string path_config      = string("examples/kitti/config/kitti.yaml");
//    string path_calib   = string("kitti/KITTI00-02.yaml");

    int frameGridRows, frameGridCols, winSize, pyrMaxLevel, nFeatures, nLevels, fIniThFAST, fMinThFAST,
        ransacMinSet, ransacMaxIt, max_iter_3d, ransacMinSetGN, ransacMaxItGN, maxIteration, finalMaxIteration;
    double maxDisp, minDisp, thDepth, sadMinValue, halfBlockSize, ransacProb,ransacTh, th_3d, ransacProbGN,
           ransacThGN, minIncTh, adjustValue;
 
    float fScaleFactor; 
    bool reweigh;

    LoadParameters(path_config, frameGridRows, frameGridCols,  maxDisp, minDisp, thDepth, sadMinValue, halfBlockSize, 
            winSize, pyrMaxLevel, nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, ransacProb, ransacMinSet, 
            ransacMaxIt, ransacTh, max_iter_3d, th_3d, ransacProbGN, ransacThGN, ransacMinSetGN, ransacMaxItGN, minIncTh, 
            maxIteration, finalMaxIteration, reweigh, adjustValue);

    Tracking* trackerPtr = new Tracking(frameGridRows, frameGridCols,  maxDisp, minDisp, sadMinValue, halfBlockSize, 
            winSize, pyrMaxLevel, nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, ransacProb, ransacMinSet, 
            ransacMaxIt, ransacTh, max_iter_3d, th_3d, ransacProbGN, ransacThGN, ransacMinSetGN, ransacMaxItGN, 
            maxIteration, finalMaxIteration, reweigh, adjustValue);
    
    Viewer* viewerPtr = new Viewer(trackerPtr);
    std::thread viewer (&Viewer::run, viewerPtr);

    cv::FileStorage fsSettings(path_calib, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    double fu, fv, uc, vc, bf;

    fu = fsSettings["Camera.fx"];
    fv = fsSettings["Camera.fy"];
    uc = fsSettings["Camera.cx"];
    vc = fsSettings["Camera.cy"];

    bf = fsSettings["Camera.bf"];

    trackerPtr->setCalibrationParameters(fu, fv, uc, vc, bf);

    // Main loop
    cv::Mat imLeft, imRight;
    int current_ni;
    cv::Mat traj_img = Mat::zeros(600, 600, CV_8UC3);
    for(int ni=0; ni<nImages; ni++)
//    for(int ni=0; ni<2; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],IMREAD_UNCHANGED);

        if(imLeft.channels() == 3)
            cvtColor(imLeft,imLeft,COLOR_RGB2GRAY);
        if(imRight.channels() == 3)
            cvtColor(imRight,imRight,COLOR_RGB2GRAY);

        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // std::thread tracker (&Tracking::start, trackerPtr, imLeft,imRight, tframe);
        // tracker.join();
        trackerPtr->start(imLeft,imRight, tframe);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        cv::Mat current_pose = trackerPtr->getCurrentPose();

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);

        current_ni = ni;

        draw_trajectory(current_pose, traj_img);

        cv::imshow( "Trajectory", traj_img );

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
    cout << "total time in seconds: "   << totaltime            << endl;
    cout << "mean tracking time: "      << meanTime << endl;


    trackerPtr->saveTrajectoryKitti(resultPath+resultFile);
#if LOG
    trackerPtr->saveStatistics(statsPath+statsFile, meanTime);

//    tracking.saveTrajectoryKitti8point(resultPath+"8point_"+resultFile);
#endif
//    tracking.saveTrajectoryTUM("KLTVO_KITTI_TUM.txt");
    cout << "-------" << endl << endl;
    cv::destroyAllWindows();

    return 0;
}

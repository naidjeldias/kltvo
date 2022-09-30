#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "tracking.h"
#include "utils.h"


class time_point;

using namespace std;

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
    
    // if(argc <= 2)
    // {
    //     cout << "Sequence "<< seq << " selected!"<< endl;
    // }else
    //     cout << "No sequence passed as argument default sequence "<< seq << " will be selected!"<< seq << endl;

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

    Tracking tracking(frameGridRows, frameGridCols,  maxDisp, minDisp, thDepth, sadMinValue, halfBlockSize, 
            winSize, pyrMaxLevel, nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, ransacProb, ransacMinSet, 
            ransacMaxIt, ransacTh, max_iter_3d, th_3d, ransacProbGN, ransacThGN, ransacMinSetGN, ransacMaxItGN, minIncTh, 
            maxIteration, finalMaxIteration, reweigh, adjustValue);

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

    tracking.setCalibrationParameters(fu, fv, uc, vc, bf);

    // Main loop
    cv::Mat imLeft, imRight;
    int current_ni;
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

        tracking.start(imLeft,imRight, tframe);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

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


    tracking.saveTrajectoryKitti(resultPath+resultFile);
#if LOG
    tracking.saveStatistics(statsPath+statsFile, meanTime);

//    tracking.saveTrajectoryKitti8point(resultPath+"8point_"+resultFile);
#endif
//    tracking.saveTrajectoryTUM("KLTVO_KITTI_TUM.txt");
    cout << "-------" << endl << endl;
    cv::destroyAllWindows();

    return 0;
}

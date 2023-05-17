#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "tracking.h"
#include "utils.h"
#include <unistd.h>
#include "viewer.hpp"


class time_point;

using namespace std;

cv::Mat global_pose = cv::Mat::eye(4,4,CV_32F);

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


int main(int argc, char *argv[]) {

    // Retrieve paths to images
    //full kitti dataset
    string seq = "03";

    cout << endl << "-------" << endl;

    string path_data = string("../../KITTI_DATASET/dataset/sequences/"+seq);

    if (argc == 3){
        path_data = argv[1];
        seq       = argv[2];
        cout << "Using sequence " << seq << " on path: " << path_data << endl;
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
    bool viz = true;
    YAML::Node odometry_params = YAML::LoadFile(path_config);
    viz = odometry_params["Viewer.enabled"].as<bool>();
    Tracking* trackerPtr = new Tracking(odometry_params);

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

    // starting visualizer thread
    Viewer* viewer_;
    std::thread* viewer_thd_;
    if (viz)
    {
        viewer_ = new Viewer(path_config);
        viewer_thd_ = new thread(&Viewer::run, viewer_);
    }
    
    


    // Main loop
    cv::Mat imLeft, imRight;
    int current_ni;
   for(int ni=0; ni<nImages; ni++)
//    for(int ni=0; ni<20; ni++)
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
        if (viz)
            viewer_->update(trackerPtr);

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

    cv::waitKey(0);

    if(viz)
    {
        viewer_->shutdown();
        viewer_thd_->join();
        delete viewer_thd_;
        delete viewer_;
    }
    delete trackerPtr;
    cout << "-------" << endl << endl;

    return 0;
}

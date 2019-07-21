#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <zconf.h>
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

int main() {

    //kitti dataset
//   string path_data    = string("kitti");
//   string path_left    = string ("/video_0.avi");
//   string path_right   = string ("/video_1.avi");


    // Retrieve paths to images
    //full kitti dataset
    string path_data = string("../../KITTI_DATASET/dataset/sequences/03");
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(path_data, vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    string path_calib   = string("kitti/KITTI03.yaml");
//    string path_calib   = string("kitti/KITTI00-02.yaml");
    Tracking tracking(path_calib);

    // Main loop
    cv::Mat imLeft, imRight;
    int current_ni;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],IMREAD_UNCHANGED);

        if(imLeft.channels() == 3)
            cvtColor(imLeft,imLeft,CV_RGB2GRAY);
        if(imRight.channels() == 3)
            cvtColor(imRight,imRight,CV_RGB2GRAY);

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
    cout << "-------" << endl << endl;
    cout << "mean tracking time: " << totaltime/current_ni << endl;

    tracking.saveTrajectoryKitti("KLTVO_KITTI.txt");
//    tracking.saveTrajectoryTUM("KLTVO_KITTI_TUM.txt");

    cv::destroyAllWindows();

    return 0;
}
//
// Created by nigel on 21/01/19.
//

#ifndef KLTVO_TRACKING_H
#define KLTVO_TRACKING_H

#define MAX_DELTAY 0
#define MAX_DELTAX 62

#define FRAME_GRID_COLS 48
#define FRAME_GRID_ROWS 48

#include "utils.h"
#include "epnp.h"
#include <iostream>
#include <stdio.h>
#include <list>
#include <iterator>
#include <thread>
#include <mutex>
#include <functional>

#include "opencv2/features2d/features2d.hpp"

#include "eightpoint.hpp"
#include "ORBextractor.h"

class Tracking{

public:

    enum status {CONVERGED, UPDATE, FAILED};

    Tracking(const string &strSettingPath);

    ~Tracking();

    cv::Mat K;
    double  baseline;
    cv::Mat P1, P2;
    double uc, vc, fu, fv;

    //Current relative pose
    cv::Mat Tcw;

    cv::Mat getCurrentPose();

    void saveTrajectoryKitti(const string &filename);

    void saveTrajectoryTUM(const string &filename);

    void start(const cv::Mat &imLeft, const cv::Mat &imRight, const double timestamp);

    //create log file for debug
    bool debug_;
    ofstream logFile;


private:

    //--------------ORB extractor variables
    int nFeatures;
    float fScaleFactor;
    int nLevels ;
    int fIniThFAST;
    int fMinThFAST;
    std::mutex mtxORB;



    //--------------Stereo matching
    double maxDisp, minDisp, initTimestamp;

    //-------------KLT threads
    std::mutex mtx1, mtx2, mtx3;


    //--------------ORB extractor
    ORBextractor* mpORBextractorLeft;
    ORBextractor* mpORBextractorRight;

    bool initPhase;

    int numFrame;

    //--------------Eight Point Algorithm
    EightPoint* mEightPointLeft;



    std::list<cv::Mat> relativeFramePoses;
    std::list<double>  frameTimeStamp;


    //----------local mapping
    int max_iter_3d;
    double th_3d;
    cv::Mat imLeft0, imRight0;

    //----------Pose estimation
    double ransacProb, ransacTh;
    int ransacMinSet, ransacMaxIt;
    std::vector<int> generateRandomIndices(const unsigned long &maxIndice, const int &vecSize);
    double minIncTh;            // min increment for pose optimization
    int maxIteration;           // max number of iteration for pose update
    int finalMaxIteration;      // max iterations for minimization final refinement
    bool reweigh;               // reweight in optimization


    void localMapping (const std::vector<cv::Point2f> &pts_l, const std::vector<cv::Point2f> &pts_r,
                       std::vector<cv::Point3f> &pts3D, const std::vector<cv::DMatch> &macthes, double &meanError);

    void stereoMatching(const std::vector<cv::Point2f>& pts_l, const std::vector<cv::Point2f>& pts_r, const cv::Mat& imLeft,
                        const cv::Mat& imRight,  std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &new_pts_l,
                        std::vector<cv::Point2f> &new_pts_r);

    bool findMatchingSAD(const cv::Point2f &pt_l, const cv::Mat& imLeft, const cv::Mat& imRight,
                         std::vector<cv::Point2f>& pts_r, cv::Point2f &ptr_m, int &index);

    int poseEstimationRansac(const std::vector<cv::Point2f> &pts2dl, const std::vector<cv::Point2f> &pts2dr, const std::vector<cv::Point3f> &pts3d
            , std::vector<double> &p0, std::vector<bool> &inliers, std::vector<double> &p, bool reweigh);

    int poseEstimation(const std::vector<cv::Point2d> &pts2dl, const std::vector<cv::Point2d> &pts2dr, const std::vector<cv::Point3d> &pts3d
            , std::vector<double> &p0, const int numPts, bool reweigh);

    void computeJacobian(const int numPts, const std::vector<cv::Point3d> &pts3D, const std::vector<cv::Point2d> &pts2d_l,
                         const std::vector<cv::Point2d> &pts2d_r, std::vector<double> &p0, cv::Mat &J, cv::Mat &res, bool reweigh);

    int checkInliers(const std::vector<cv::Point3f> &pts3d, const std::vector<cv::Point2f> &pts2dl, const std::vector<cv::Point2f> &pts2dr,
                     const std::vector<int> &index, const std::vector<double> &p0, std::vector<bool> &inliers, long double &sumErr, bool reweigh);

    double euclideanDist(const cv::Point2d &p, const cv::Point2d &q);

    void quadMatching(const std::vector<cv::Point3f> &pts3D, const std::vector<cv::Point2f> &pts2D_l, const std::vector<cv::Point2f> &pts2D_r
            , std::vector<bool> &inliers, const cv::Mat &imLeft, const cv::Mat &imRight, std::vector<cv::Point3f> &new_pts3D,
                      std::vector<cv::Point2f> &new_pts2D_l, std::vector<cv::Point2f> &new_pts2D_r, std::vector<cv::DMatch> &matches);

    void essentialMatrixDecomposition(const cv::Mat &F, const cv::Mat &K, const cv::Mat &K_l, const std::vector<cv::Point2f> &pts_l,
                                      const std::vector<cv::Point2f> &pts_r, const std::vector<bool> &inliers , cv::Mat &R_est, cv::Mat &t_est);

    void checkSolution(const cv::Mat &R1, const cv::Mat &R2, const cv::Mat &u3, const cv::Mat &K, const cv::Mat &K_l, const cv::Point2f &pt_l
            , const cv::Point2f &pt_r, cv::Mat &R_est, cv::Mat &t_est);

    bool pointFrontCamera(cv::Mat &R, const cv::Mat &u3, const cv::Mat &pt_l, const cv::Mat &pt_r, const cv::Mat &P, cv::Mat &P_l,
            const cv::Mat &K, const cv::Mat &K_l);


    void extractORB(int flag, cv::Mat &im, std::vector<KeyPoint> &kpt, std::vector<cv::Point2f> &pts);

    void opticalFlowFeatureTrack(cv::Mat &imT0, const cv::Mat &imT1, Size win, int maxLevel, cv::Mat &status, cv::Mat &error,
                                 std::vector<Point2f> &prevPts, std::vector<Point2f> &nextPts, std::vector <Mat> imT0_pyr, std::vector <Mat> imT1_pyr);

    void gridNonMaximumSuppression(std::vector<cv::Point2f> &pts, const std::vector<cv::KeyPoint> &kpts, const cv::Mat &im);

    bool assignFeatureToGrid(const cv::KeyPoint &kp, int &posX, int &posXY, const cv::Mat &im, const int &nBucketX, const int &nBucketY);


    //----------------------debug functions
    void drawPointfImage(const cv::Mat &im, const std::vector<Point2f> pts, const string &filename);
    void writeOnLogFile(const string &name, const string &value);
    void drawGridAndPoints(const cv::Mat &im, const std::vector<Point2f> &pts, const string &fileName);
    void logFeatureExtraction(const std::vector<cv::KeyPoint> &kpts_l, const std::vector<cv::KeyPoint> &kpts_r,
                              const std::vector<Point2f> &pts, const cv::Mat &im);
    void logStereoMatching(const cv::Mat &im_r, const cv::Mat &im_l, const std::vector<cv::DMatch> &mrl,
                           const std::vector<Point2f> &pts_r, const std::vector<Point2f> &pts_l);
    void logLocalMaping(const std::vector<Point3f> &pts3D, double &meanError);
    void logFeatureTracking(const std::vector<Point2f> &pts_l0, const std::vector<Point2f> &pts_r1, const cv::Mat &fmat,
                            const std::vector<Point2f> &pts_l1, const std::vector<bool> &inliers, const cv::Mat &im_l0,
                            const cv::Mat &im_l1, const std::vector<cv::DMatch> &mll, const cv::Mat &R);
    void logQuadMatching(const cv::Mat &im_l1, const cv::Mat &im_r1, const std::vector<Point2f> &pts_l1,
                         const std::vector<Point2f> &pts_r1, const std::vector<cv::DMatch> &mlr1, int numPts);
    void logPoseEstimation();



};

#endif //KLTVO_TRACKING_H

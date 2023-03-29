//
// Created by nigel on 21/01/19.
//

#ifndef KLTVO_TRACKING_H
#define KLTVO_TRACKING_H

#define MAX_DELTAY 0
//#define MAX_DELTAX 69
#define MAX_DELTAX 721

#define LOG             true
#define ENABLE_PRINT    false
#define LOG_DRAW        false

#define FRAME_GRID_COLS 24
#define FRAME_GRID_ROWS 24

#include "utils.h"
#include <iostream>
#include <stdio.h>
#include <list>
#include <iterator>
#include <thread>
#include <mutex>
#include <functional>
#include <math.h>
#include<Eigen/Dense>


#include "opencv2/features2d/features2d.hpp"

#include "eightpoint.hpp"
#include "ORBextractor.h"

class Tracking{

public:
    
    struct{
        std::vector<cv::Point2f> features;
        std::vector<cv::Point2f> keypoints;
    }keyframe_;

    enum status {CONVERGED, UPDATE, FAILED};

    Tracking(int &frameGridRows, int &frameGridCols,  double &maxDisp, double &minDisp, double &sadMinValue, 
        double &halfBlockSize, int &winSize, int &pyrMaxLevel, int &nFeatures, float &fScaleFactor, int &nLevels, 
        int &fIniThFAST, int &fMinThFAST, double &ransacProbTrack, int &ransacMinSetTrack, int &ransacMaxItTrack, 
        double &ransacThTrack, int &max_iter_3d, double &th_3d, double &ransacProbGN, double &ransacThGN, int &ransacMinSetGN, 
        int &ransacMaxItGN, int &maxIteration, int &finalMaxIteration, bool &reweigh, double &adjustValue);

    ~Tracking();

    cv::Mat K;
    double  baseline;
    cv::Mat P1, P2;
    double uc, vc, fu, fv;
    cv::Mat imLeft0_, imRight0_;

    // Camera poses
    cv::Mat cameraCurrentPose_;
    std::vector<cv::Mat> cameraPoses_;

    void setCalibrationParameters(const double &mFu, const double &mFv, const double &mUc, const double &mVc,
                   const double &mbf);

    void saveTrajectoryKitti(const string &filename);

    void saveTrajectoryKitti8point(const string &filename);


    void saveTrajectoryEuroc(const string &filename);

    void saveStatistics (const string &filename, float &meanTime, bool withTime= false);

    cv::Mat start(const cv::Mat &imLeft, const cv::Mat &imRight, const double timestamp);

    //create log file for debug
    bool debug_;


private:

    std::list<cv::Mat> relativeFramePoses;
    std::list<double>  frameTimeStamp;

#if LOG
    std::list<int > gnIterations, leftPtsDetec, ptsNMS, ptsStereoMatch, ptsTracking,
                    ptsQuadMatch, numInliersGN, maxItGN, ransacIt_8point;
    std::list<double > gnMeanIterations, rep_err_3d;
    std::list<cv::Mat> relativeFramePoses_;
#endif


    bool initPhase;
    int numFrame;

    double euclideanDist(const cv::Point2d &p, const cv::Point2d &q);

    cv::Mat computeGlobalPose(const cv::Mat &current_pose);

    //-------------- feature extraction
    int nFeatures;
    float fScaleFactor;
    int nLevels ;
    int fIniThFAST;
    int fMinThFAST;
    std::mutex mtxORB;
    int frameGridCols;
    int frameGridRows;

    ORBextractor* mpORBextractorLeft;
    ORBextractor* mpORBextractorRight;

    void featureExtraction (const cv::Mat &im0, const cv::Mat &im1, std::vector<KeyPoint> &kpts0,
            std::vector<KeyPoint> &kpts1, std::vector<Point2f> &pts0, std::vector<Point2f> &pts1);
    void extractORB(int flag, const cv::Mat &im, std::vector<KeyPoint> &kpt, std::vector<cv::Point2f> &pts);

    void gridNonMaximumSuppression(std::vector<cv::Point2f> &pts, const std::vector<cv::KeyPoint> &kpts, const cv::Mat &im);

    bool assignFeatureToGrid(const cv::KeyPoint &kp, int &posX, int &posXY, const cv::Mat &im, const int &nBucketX, const int &nBucketY);



    //-------------- stereo matching
    double maxDisp, minDisp, initTimestamp, thDepth, sadMinValue, halfBlockSize;


    void stereoMatching(const std::vector<cv::Point2f>& pts_l, const std::vector<cv::Point2f>& pts_r, const cv::Mat& imLeft,
                        const cv::Mat& imRight,  std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &new_pts_l,
                        std::vector<cv::Point2f> &new_pts_r, std::vector<cv::Point3f> &pointCloud, double &meanError, std::vector<bool> &ptsClose);

    bool findMatchingSAD(const cv::Point2f &pt_l, const cv::Mat& imLeft, const cv::Mat& imRight,
                         std::vector<cv::Point2f>& pts_r, cv::Point2f &ptr_m, int &index, const std::vector<std::vector<std::size_t>> &vecRowIndices);

    int sign(double value);


    //------------- feature tracking
    std::mutex mtx1, mtx2, mtx3, mtx4;
    int winSize, pyrMaxLevel;


    void featureTracking (const cv::Mat &imL0, const cv::Mat &imL1, const cv::Mat &imR0, const cv::Mat &imR1, std::vector<Point2f> &ptsL0,
            std::vector<Point2f> &ptsL1, std::vector<Point2f> &ptsR0, std::vector<Point2f> &ptsR1, std::vector<Point3f> &pts3D, std::vector<bool> &ptsClose );

    void opticalFlowFeatureTrack(const cv::Mat &imT0, const cv::Mat &imT1, Size win, int maxLevel, std::vector<uchar> &status, std::vector<float> &error,
                                 std::vector<Point2f> &prevPts, std::vector<Point2f> &nextPts, std::vector <Mat> imT0_pyr,
                                 std::vector <Mat> imT1_pyr, int flag, std::vector<Point3f> &pts3D, std::vector<bool> &ptsClose);

    void checkPointOutBounds(std::vector<Point2f> &prevPts, std::vector<Point2f> &nextPts,
                             const cv::Mat &imT1, const  std::vector<uchar> &status, int flag, std::vector<Point3f> &pts3D, std::vector<bool> &ptsClose);

    //-------------- Outliers removal and motion estimation
    EightPoint* mEightPointLeft;

    void outlierRemovalAndMotionEstimation(const cv::Mat &imL0, const std::vector<Point2f> &ptsL0
            , const cv::Mat &imL1 ,const std::vector<Point2f> &ptsL1, const cv::Mat &imR0, const std::vector<Point2f> &ptsR0,
            const cv::Mat &imR1, const std::vector<Point2f> &ptsR1, std::vector<bool> &inliers, std::vector<double> &rvec_est, cv::Mat &t_est);

    void essentialMatrixDecomposition(const cv::Mat &F, const cv::Mat &K, const cv::Mat &K_l, const std::vector<cv::Point2f> &pts_l,
                                      const std::vector<cv::Point2f> &pts_r, std::vector<bool> &inliers , cv::Mat &R_est, cv::Mat &t_est);

    void checkSolution(const cv::Mat &R1, const cv::Mat &R2, const cv::Mat &u3, const cv::Mat &K, const cv::Mat &K_l, const std::vector<cv::Point2f> &pts_l
            , const std::vector<cv::Point2f> &pts_r, cv::Mat &R_est, cv::Mat &t_est, std::vector<bool> &inliers);

    bool pointFrontCamera(cv::Mat &R, const cv::Mat &u3, const cv::Mat &pt_l, const cv::Mat &pt_r, const cv::Mat &P, cv::Mat &P_l,
                          const cv::Mat &K, const cv::Mat &K_l);


    //----------local mapping
    int max_iter_3d;
    double th_3d;

    void localMapping (const std::vector<cv::Point2f> &pts_l, const std::vector<cv::Point2f> &pts_r,
                       std::vector<cv::Point3f> &pts3D, const std::vector<cv::DMatch> &macthes, double &meanError);

    bool triangulation (const cv::Point2f &pt_l, const cv::Point2f &pt_r, cv::Point3f &pt3D, double &error, double &depth);


    //----------- quad Matching
    void quadMatching(const std::vector<cv::Point3f> &pts3D, const std::vector<cv::Point2f> &pts2D_l, const std::vector<cv::Point2f> &pts2D_r
            , std::vector<bool> &inliers, const cv::Mat &imLeft, const cv::Mat &imRight, std::vector<cv::Point3f> &new_pts3D,
                      std::vector<cv::Point2f> &new_pts2D_l, std::vector<cv::Point2f> &new_pts2D_r,
                      std::vector<cv::DMatch> &matches, const std::vector<bool> &ptsClose);


    //----------Pose estimation
    double ransacProb, ransacTh;
    int ransacMinSet, ransacMaxIt;
    std::vector<int> generateRandomIndices(const unsigned long &maxIndice, const int &vecSize);
    double minIncTh;            // min increment for pose optimization
    int maxIteration;           // max number of iteration for pose update
    int finalMaxIteration;      // max iterations for minimization final refinement
    bool reweigh;               // reweight in optimization
    double adjustValue;


    void relativePoseEstimation(const std::vector<cv::Point2f> &pts2DL, const std::vector<cv::Point2f> &pts2DR,
            const std::vector<cv::Point3f> &pts3D, const std::vector<double> &rvec_est, const cv::Mat &t_est ,cv::Mat &Tcw);

    void poseRefinment(const std::vector<Point2f> &pts2DL, const std::vector<Point2f> &pts2DR,
            const std::vector<Point3f> &pts3D, const std::vector<bool> &inliers, std::vector<double> &p ,cv::Mat &rot_vec,
            cv::Mat &tr_vec, const int &bestNumInliers);

    void poseEstimationRansac(const std::vector<cv::Point2f> &pts2dl, const std::vector<cv::Point2f> &pts2dr, const std::vector<cv::Point3f> &pts3d
            , std::vector<double> &p0, std::vector<bool> &inliers, std::vector<double> &p, bool reweigh, int &bestNumInliers);

    int poseEstimation(const std::vector<cv::Point2d> &pts2dl, const std::vector<cv::Point2d> &pts2dr, const std::vector<cv::Point3d> &pts3d
            , std::vector<double> &p0, const int numPts, bool reweigh);

    void computeJacobian(const int numPts, const std::vector<cv::Point3d> &pts3D, const std::vector<cv::Point2d> &pts2d_l,
                         const std::vector<cv::Point2d> &pts2d_r, std::vector<double> &p0, cv::Mat &J, cv::Mat &res, bool reweigh);

    int checkInliers(const std::vector<cv::Point3f> &pts3d, const std::vector<cv::Point2f> &pts2dl, const std::vector<cv::Point2f> &pts2dr,
                     const std::vector<int> &index, const std::vector<double> &p0, std::vector<bool> &inliers, long double &sumErr, bool reweigh, long double &stdDev);

    //----------------------Tools functions
    std::vector<float > toQuaternion(const cv::Mat &R);


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
                            const cv::Mat &im_l1, const std::vector<cv::DMatch> &mll);
    void logQuadMatching(const cv::Mat &im_l1, const cv::Mat &im_r1, const std::vector<Point2f> &pts_l1,
                         const std::vector<Point2f> &pts_r1, const std::vector<cv::DMatch> &mlr1, int numPts);
    void logPoseEstimation();

    void drawFarAndClosePts (const cv::Point2f &pt, const cv::Scalar &color, cv::Mat &im);



};

#endif //KLTVO_TRACKING_H

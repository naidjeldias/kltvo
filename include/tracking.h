//
// Created by nigel on 21/01/19.
//

#ifndef KLTVO_TRACKING_H
#define KLTVO_TRACKING_H

#define MAX_DELTAY 0
#define MAX_DELTAX 32

#include "utils.h"
#include "epnp.h"
#include <iostream>
#include <stdio.h>
#include <list>
#include <iterator>

#include "opencv2/features2d/features2d.hpp"

#include "eightpoint.hpp"
#include "ORBextractor.h"

class Tracking{

public:

    enum status {CONVERGED, UPDATE, FAILED};

    Tracking();

    ~Tracking();

    cv::Mat K;
    double  baseline;
    cv::Mat P1, P2;
    double uc, vc, fu, fv;

    //Current relative pose
    cv::Mat Tcw;

    cv::Mat getCurrentPose();

    void saveTrajectoryKitti(const string &filename);


    void start(const cv::Mat &imLeft, const cv::Mat &imRight);

private:

    bool initPhase;

    EightPoint eightPoint;

    std::list<cv::Mat> relativeFramePoses;


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
                       std::vector<cv::Point3f> &pts3D, const std::vector<cv::DMatch> &macthes);

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


};

#endif //KLTVO_TRACKING_H

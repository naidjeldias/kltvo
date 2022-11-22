//
// Created by nigel on 21/01/19.
//


#include "tracking.h"



using namespace cv;

Tracking::Tracking(int &frameGridRows, int &frameGridCols,  double &maxDisp, double &minDisp, double &sadMinValue, double &halfBlockSize, int &winSize, int &pyrMaxLevel, 
                    int &nFeatures, float &fScaleFactor, int &nLevels, int &fIniThFAST, int &fMinThFAST,  
                    double &ransacProbTrack, int &ransacMinSetTrack, int &ransacMaxItTrack, double &ransacThTrack, int &max_iter_3d, double &th_3d, 
                    double &ransacProbGN, double &ransacThGN, int &ransacMinSetGN, int &ransacMaxItGN, 
                    int &maxIteration, int &finalMaxIteration, bool &reweigh, double &adjustValue) : 
frameGridRows(frameGridRows), frameGridCols(frameGridCols),  maxDisp(maxDisp), minDisp(minDisp), thDepth(35.0), sadMinValue(sadMinValue), halfBlockSize(halfBlockSize), 
winSize(winSize), pyrMaxLevel(pyrMaxLevel), nFeatures(nFeatures), fScaleFactor(fScaleFactor), nLevels(nLevels), fIniThFAST(fIniThFAST), fMinThFAST(fMinThFAST), max_iter_3d(max_iter_3d), 
th_3d(th_3d), ransacProb(ransacProbGN), ransacTh(ransacThGN), ransacMinSet(ransacMinSetGN), ransacMaxIt(ransacMaxItGN), minIncTh(10E-5), 
maxIteration(maxIteration), finalMaxIteration(finalMaxIteration), reweigh(reweigh), adjustValue(adjustValue), Tcw(cv::Mat::eye(4,4,CV_32F)), cameraCurrentPose_(cv::Mat::eye(4,4,CV_32F))

{
    srand(time(0));

    //-----Feature extraction
    std::cout << "NMS parameters: \n";

    std::cout << "- Num Grid rows : "                  << frameGridRows           << std::endl;
    std::cout << "- Num Grid cols:  "                  << frameGridCols             << std::endl;

    //----Stereo Matching
    std::cout << "Estereo Matching parameters: \n";

    std::cout << "- Min disparity: "                  << minDisp           << std::endl;
    std::cout << "- Max disparity: "                  << maxDisp             << std::endl;
    std::cout << "- Threshold depth: "                << thDepth         << std::endl;

    std::cout << "- SAD min value: "                  << sadMinValue             << std::endl;
    std::cout << "- SAD halfBlockSize: "              << halfBlockSize         << std::endl;

    //-----KLT feature tracker
    std::cout << "KLT parameters: \n";

    std::cout << "- Search Windows Size : "                  << winSize           << std::endl;
    std::cout << "- Pyramid max level:  "                    << pyrMaxLevel             << std::endl;

    //-----ORB extractor

    mpORBextractorLeft  = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //-----Eight Point algorithm

    std::cout << "- Ransac prob: "                  << ransacProb          << std::endl;
    std::cout << "- Ransac Th: "                    << ransacTh            << std::endl;
    std::cout << "- Ransac min set: "               << ransacMinSet         << std::endl;
    std::cout << "- Ransac max it: "                << ransacMaxIt          << std::endl;

    mEightPointLeft         = new EightPoint(ransacProbTrack, ransacMinSetTrack, ransacMaxItTrack, ransacThTrack);

    
    //----local mapping
    std::cout << "Triangulation parameters: \n";

    std::cout << "- Max it: "   << max_iter_3d  << std::endl;
    std::cout << "- Th 3d: "    << th_3d        << std::endl;

    //----Pose estimation
    std::cout << "Pose estimation parameters: \n";

    std::cout << "- Ransac prob: "                         << ransacProb           << std::endl;
    std::cout << "- Ransac Th: "                           << ransacTh             << std::endl;
    std::cout << "- Ransac min set: "                      << ransacMinSet         << std::endl;
    std::cout << "- Ransac max it: "                       << ransacMaxIt          << std::endl;
    std::cout << "- Min increment th: "                    << minIncTh             << std::endl;
    std::cout << "- Max iteration: "                       << maxIteration         << std::endl;
    std::cout << "- Refinment max iteration: "             << finalMaxIteration    << std::endl;
    std::cout << "- reprojection weight adjust value: "    << adjustValue          << std::endl;

    initPhase = true;

#if ENABLE_VIZ
    // starting visualizer thread
    viewer_ = new Viewer();
    viewer_thd_ = new thread(&Viewer::run, viewer_);
#endif
}


Tracking::~Tracking() {
}


void Tracking::setCalibrationParameters(const double &mFu, const double &mFv, const double &mUc, const double &mVc,
                   const double &mbf)
{
    fu = mFu; fv= mFv; uc = mUc; vc = mVc;

    baseline = mbf / fu;

    std::cout << "Camera parameters: \n";

    std::cout << "- fu: " << fu << std::endl;
    std::cout << "- fv: " << fv << std::endl;
    std::cout << "- uc: " << uc << std::endl;
    std::cout << "- vc: " << vc << std::endl;

    cv::Mat mK = cv::Mat::eye(3,3,CV_64F);
    mK.at<double>(0,0) = fu;
    mK.at<double>(1,1) = fv;
    mK.at<double>(0,2) = uc;
    mK.at<double>(1,2) = vc;
    mK.copyTo(K);

    std::cout << "- b: " << baseline << std::endl;

    Mat mP1 = cv::Mat::eye(3,4, CV_64F);
    Mat mP2 = cv::Mat::eye(3,4, CV_64F);

    mK.copyTo(mP1.rowRange(0,3).colRange(0,3));
    mK.copyTo(mP2.rowRange(0,3).colRange(0,3));

    mP2.at<double>(0,3) = -mbf;

    mP1.copyTo(P1);
    mP2.copyTo(P2);

}
cv::Mat Tracking::start(const Mat &imLeft, const Mat &imRight, const double timestamp) {


    if (initPhase){
        imLeft0         = imLeft.clone();
        imRight0        = imRight.clone();

        initPhase       = false;
        numFrame        = 0;
        initTimestamp   = timestamp;
    }else{

        numFrame ++;
#if LOG
        writeOnLogFile("Frame:", std::to_string(numFrame+1));
#endif

        //---------------------------------detect features
        std::vector<KeyPoint> kpts_l, kpts_r;
        kpts_l.reserve(nFeatures);
        kpts_r.reserve(nFeatures);

        std::vector<Point2f> pts_l0, pts_r0;
        pts_l0.reserve(nFeatures);
        pts_r0.reserve(nFeatures);

        featureExtraction(imLeft0, imRight0, kpts_l, kpts_r, pts_l0, pts_r0);

//        std::vector<cv::KeyPoint> kp_;
//        cv::Mat imOut;
//        cv::FAST(imLeft0, kp_, 100, true, cv::FastFeatureDetector::TYPE_9_16);
//        cv::drawKeypoints(imLeft0,kp_,imOut, cv::Scalar(0,255,0));
//        cv::imwrite("kptsFAST.png", imOut);

        //convert vector of keypoints to vector of Point2f
        for (auto& kpt:kpts_r)
            pts_r0.push_back(kpt.pt);

        //Free memory
        std::vector<KeyPoint>().swap(kpts_r);
        std::vector<KeyPoint>().swap(kpts_l);

        //-----------------------------------stereo matching
        std::vector<Point2f> new_pts_l0, new_pts_r0;
        new_pts_l0.reserve(pts_l0.size());
        new_pts_r0.reserve(pts_r0.size());

        std::vector<DMatch> mlr0;
        mlr0.reserve(pts_l0.size());

        std::vector<Point3f> pts3D;
        pts3D.reserve(pts_l0.size());

        double meanError;

        std::vector<bool> ptsClose;
        ptsClose.reserve(pts_l0.size());

        stereoMatching(pts_l0, pts_r0, imLeft0, imRight0, mlr0, new_pts_l0, new_pts_r0, pts3D, meanError, ptsClose);
#if LOG
        logStereoMatching(imRight0, imLeft0, mlr0, new_pts_r0, new_pts_l0);
        logLocalMaping(pts3D, meanError);
        rep_err_3d.push_back(meanError);
#endif


        //-----------------------------------tracking features from previous frames to current frames
        std::vector<Point2f> pts_l1, pts_r1;
        pts_l1.reserve(new_pts_l0.size());
        pts_r1.reserve(new_pts_r0.size());

        featureTracking(imLeft0, imLeft, imRight0, imRight, new_pts_l0, pts_l1, new_pts_r0, pts_r1, pts3D, ptsClose);


        //-----------------------------------outliers removal and 2D motion estimation
        std::vector<bool>       inliers;
        std::vector<double>     rvec_est;
        Mat t_est;

        outlierRemovalAndMotionEstimation(imLeft0, new_pts_l0, imLeft, pts_l1, imRight0,
                new_pts_r0, imRight, pts_r1, inliers, rvec_est, t_est);



        //------------------------------------circular matching
        std::vector<Point3f> new_pts3D;
        std::vector<Point2f> new_pts_l1, new_pts_r1;
        new_pts_l1.reserve(pts_l1.size());
        new_pts_r1.reserve(pts_r1.size());
        new_pts3D.reserve(pts3D.size());

        std::vector<DMatch> mlr1;
        mlr1.reserve(pts_l1.size());

        quadMatching(pts3D, pts_l1, pts_r1, inliers, imLeft, imRight, new_pts3D, new_pts_l1, new_pts_r1, mlr1, ptsClose);
#if LOG
        logQuadMatching(imLeft, imRight, new_pts_l1, new_pts_r1, mlr1, new_pts3D.size());
#endif

        //free memory
        std::vector<Point2f>().swap(pts_l1);
        std::vector<Point2f>().swap(pts_r1);
        std::vector<Point3f>().swap(pts3D);

        //------------------------------------relative pose estimation
        Mat Tcw_ = cv::Mat::eye(3,4,CV_64F);
        
        // pose covariance matrix
        cv::Mat cov_mat = cv::Mat::zeros(6,6, CV_64F);
        
        relativePoseEstimation(new_pts_l1, new_pts_r1, new_pts3D, rvec_est, t_est, Tcw_, cov_mat);
        double pose_entropy = - std::log(cv::determinant(cov_mat));
        //saving relative pose estimated
        relativeFramePoses.push_back(Tcw_.clone());
        frameTimeStamp.push_back(timestamp);

        Tcw = Tcw_.clone();

        imLeft0     = imLeft.clone();
        imRight0    = imRight.clone();

#if LOG
        writeOnLogFile("Pose entropy: ", std::to_string(pose_entropy));
        poses_entropy_.push_back(pose_entropy);
        writeOnLogFile("----------------------------", " ");
#endif
#if ENABLE_VIZ
        cameraPoses_.push_back(computeGlobalPose(Tcw_));
        viewer_->setCameraPoses(cameraPoses_);
#endif
    }


    return Tcw;
}

void Tracking::extractORB(int flag, const cv::Mat &im, std::vector<KeyPoint> &kpt, std::vector<cv::Point2f> &pts) {

    if(flag == 0){
        (*mpORBextractorLeft) (im, cv::Mat(), kpt);
//        std::cout << "Num kpt extracted: " << kpt.size() << std::endl;
        gridNonMaximumSuppression(pts,kpt,im);

    } else
        (*mpORBextractorRight)(im, cv::Mat(), kpt);

//    mtxORB.lock();
//    gridNonMaximumSuppression(pts,kpt,im);
//    mtxORB.unlock();



}

void Tracking::gridNonMaximumSuppression(std::vector<cv::Point2f> &pts, const std::vector<cv::KeyPoint> &kpts, const cv::Mat &im) {

    int nBucketX = im.cols / frameGridCols;
    int nBucketY = im.rows / frameGridRows;

    // Image grid for non-maximum suppression
    std::vector<size_t > imageGrids[nBucketY][nBucketX];

    int numFeatures = kpts.size();

    int nReserve =  numFeatures/(nBucketX*nBucketY);

    for(unsigned int i=0; i<nBucketY;i++)
        for (unsigned int j=0; j<nBucketX;j++)
            imageGrids[i][j].reserve(nReserve);

    //assigning each feature to a bucket
    for(int i=0; i<numFeatures; i++){

        const cv::KeyPoint &kp = kpts.at(i);

        int gridPosX, gridPosY;
        if(assignFeatureToGrid(kp,gridPosX,gridPosY,im, nBucketX, nBucketY))
            imageGrids[gridPosY][gridPosX].push_back(i);
    }


    for(unsigned int i=0; i<nBucketY;i++) {
        for (unsigned int j = 0; j < nBucketX; j++) {

            const vector<size_t> bucket = imageGrids[i][j];
            if (bucket.empty())
                continue;

            int bestIndex;
            double bestScore = 0;
            for (size_t k = 0; k < bucket.size(); k++) {
                const cv::KeyPoint &kp = kpts[bucket[k]];
                if (kp.response > bestScore) {
                    bestScore = kp.response;
                    bestIndex = k;
                }
            }

            pts.push_back(kpts[bucket[bestIndex]].pt);
        }
    }


}

bool Tracking::assignFeatureToGrid(const cv::KeyPoint &kp, int &posX, int &posY, const cv::Mat &im, const int &nBucketX, const int &nBucketY) {


    double posX_ = std::round(kp.pt.x * ((double)nBucketX/(double)(im.cols-1)));
    posX = (int) posX_;
    double posY_ = std::round(kp.pt.y * ((double) nBucketY/(double)(im.rows-1)));
    posY = (int) posY_;

    //check if coordinates are inside the image dimension
    if(posX < 0 || posX >= nBucketX || posY < 0 || posY >= nBucketY)
        return false;

    return true;

}

void Tracking::drawPointfImage(const cv::Mat &im, const std::vector<Point2f> pts, const string &filename) {
    std::vector<KeyPoint> kpts;
    cv::Mat imOut;
    for (int i = 0; i < pts.size(); i++){
        KeyPoint kpt;
        kpt.pt = pts.at(i);

        kpts.push_back(kpt);
    }

    drawKeypoints( im, kpts, imOut, cv::Scalar(0,225,0), DrawMatchesFlags::DEFAULT );
    imwrite(filename, imOut);
}


void Tracking:: localMapping(const std::vector<cv::Point2f> &pts_l, const std::vector<cv::Point2f> &pts_r,
                            std::vector<cv::Point3f> &pts3D, const std::vector<DMatch> &matches, double &meanError) {

    Point2f kp_l, kp_r;

    double w0, w1;
    double dist;

    double sum = 0;
    for( int i = 0; i < pts_l.size() ; i++ ){

        kp_l = pts_l.at(i);
        kp_r = pts_r.at(i);

        w0 = w1 = 1.0;

        Mat point3d;

        for(int j = 0; j < max_iter_3d; j++){

            Mat A   = Mat::zeros(4,4,CV_64F);
            Mat D, U, Vt;

            A.row(0) = w0*(kp_l.x*P1.row(2)-P1.row(0));
            A.row(1) = w0*(kp_l.y*P1.row(2)-P1.row(1));
            A.row(2) = w1*(kp_r.x*P2.row(2)-P2.row(0));
            A.row(3) = w1*(kp_r.y*P2.row(2)-P2.row(1));

            SVD::compute(A,D,U,Vt, SVD::MODIFY_A| SVD::FULL_UV);

            point3d = Vt.row(3).t();

            point3d = point3d.rowRange(0,4)/point3d.at<double>(3);

            Mat p0 = P1*point3d;
            Mat p1 = P2*point3d;

            w0 = 1.0/p0.at<double>(2);
            w1 = 1.0/p1.at<double>(2);

            double dx0 = kp_l.x - p0.at<double>(0)/p0.at<double>(2);
            double dy0 = kp_l.y - p0.at<double>(1)/p0.at<double>(2);
            double dx1 = kp_r.x - p1.at<double>(0)/p1.at<double>(2);
            double dy1 = kp_r.y - p1.at<double>(1)/p1.at<double>(2);

            dist = sqrt(dx0*dx0+dy0*dy0) + sqrt(dx1*dx1+dy1*dy1);


            if(dist < 2*th_3d){
                break;
            }
        }
        sum += dist;

        Point3f pt3d;
        pt3d.x           = (float) point3d.at<double>(0);
        pt3d.y           = (float) point3d.at<double>(1);
        pt3d.z           = (float) point3d.at<double>(2);

        pts3D.push_back(pt3d);

    }

    meanError = sum/(pts3D.size() * 2);

}


void Tracking::poseEstimationRansac(const std::vector<cv::Point2f> &pts2dl, const std::vector<cv::Point2f> &pts2dr,
                              const std::vector<cv::Point3f> &pts3d, std::vector<double> &p0, std::vector<bool> &bestInliers,
                              std::vector<double> &p, bool reweigh, int &bestNumInliers, cv::Mat &cov_mat) {


//    std::vector<bool> bestInliers;
//    int bestNumInliers = ransacMinSet;

    long double minSumErr = FLT_MAX;
    long double bestStdDev  = LDBL_MAX;
    p = p0;

#if LOG
    int nIt     = 0;
    int sumIt   = 0; //sum iterations GN inside RANSAC
    int n_      = 0; //check number of iterations
#endif

    for (int n = 0; n < ransacMaxIt; n++){
#if LOG
//        n_ ++;
#endif
        //compute rand index
        std::vector<int> randIndex (0, ransacMinSet);    //vector of rand index
        randIndex      = generateRandomIndices(pts3d.size(), ransacMinSet);

        std::vector<Point3d> aux_pt3d;
        std::vector<Point2d> aux_pt2dl, aux_pt2dr;
//        std::vector<double > aux_vDepth;
        aux_pt3d.reserve(ransacMinSet);
        aux_pt2dl.reserve(ransacMinSet);
        aux_pt2dr.reserve(ransacMinSet);
//        aux_vDepth.reserve(ransacMinSet);


        //selecting the random points
        for(auto &index:randIndex){
            aux_pt3d.push_back(pts3d.at(index));
            aux_pt2dl.push_back(pts2dl.at(index));
            aux_pt2dr.push_back(pts2dr.at(index));
        }

        // initialize p0_ for pose otimization iteration
        std::vector<double> p0_ = p0;

        // pose covariance matrix
        cv::Mat aux_cov_mat = cv::Mat::zeros(6,6, CV_64F);

        int status  = 0;
#if LOG
        int nIt_    = 0;
#endif

        for (int i = 0; i < maxIteration; i++){
#if LOG
            nIt_ ++;

#endif
            status = poseEstimation(aux_pt2dl, aux_pt2dr, aux_pt3d, p0_, randIndex.size(), reweigh, aux_cov_mat);
            if(status != UPDATE)
                break;
        }

#if LOG
        sumIt += nIt_;
#endif

        //free memory
        std::vector<Point2d>().swap(aux_pt2dl);
        std::vector<Point2d>().swap(aux_pt2dr);
        std::vector<Point3d>().swap(aux_pt3d);
//        std::vector<double >().swap(aux_vDepth);

        if(status == FAILED)
            continue;

        std::vector<bool> inliers (pts3d.size(), false);

        long double sumErr = 0;
        long double stdDev = 0;
        //validate model againts the init set
        int numInliers = checkInliers(pts3d, pts2dl, pts2dr, randIndex, p0_, inliers, sumErr, true, stdDev);

//        int totalInliers = numInliers + ransacMinSet;

        if((numInliers > bestNumInliers) || (numInliers == bestNumInliers && stdDev < bestStdDev)){
#if LOG
            nIt = nIt_;
#endif
            bestInliers     = inliers;
            p = p0_;
            bestNumInliers  = numInliers;
//            sumErr_         = sumErr;
            bestStdDev      = stdDev;
            cov_mat = aux_cov_mat;
        }
    }


#if LOG
    writeOnLogFile("Num inliers pose estimation: ", std::to_string(bestNumInliers));
    gnIterations.push_back(nIt);
    writeOnLogFile("Num iterations best pose GN: ", std::to_string(nIt));
    numInliersGN.push_back(bestNumInliers);
    float meanIt = sumIt/ransacMaxIt;
    gnMeanIterations.push_back(meanIt);
    writeOnLogFile("Mean GN iterations inside RANSAC: ", std::to_string(meanIt));

#endif

}

int Tracking::poseEstimation(const std::vector<cv::Point2d> &pts2dl, const std::vector<cv::Point2d> &pts2dr, const std::vector<cv::Point3d> &pts3d, 
                            std::vector<double> &p0, const int numPts, bool reweigh, cv::Mat &cov_mat) {

    // 6 parameters rx, ry, rz, tx, ty, tz
    // 2 equation for each point, 2 parameters (u, v) for each 2Dpoint (left and right)
    // Jacobian matrix
    Mat J   = cv::Mat::zeros(4*numPts, 6, CV_64F);
    // residual matrix
    Mat res = cv::Mat::zeros(4*numPts, 1, CV_64F);

    cv::Mat I = cv::Mat::eye(4*numPts,4*numPts,CV_64F);

    computeJacobian(numPts, pts3d, pts2dl, pts2dr, p0, J, res, reweigh);
    cov_mat = (J.t()* I *J).inv();

    cv::Mat S = cv::Mat(6,1,CV_64F);
    bool status = cv::solve(J, res, S, DECOMP_NORMAL);

    if(status){
        bool converged = true;
        //compute increments
        for(int j = 0; j < 6; j++){
            p0.at(j) += S.at<double>(j);
            if(fabs(S.at<double>(j)) > minIncTh)
                converged = false;
        }
        if(converged){
            return CONVERGED;
        } else
            return UPDATE;
    } else
        return FAILED;

}

void Tracking::computeJacobian(const int numPts, const std::vector<cv::Point3d> &pts3D,
                               const std::vector<cv::Point2d> &pts2d_l, const std::vector<cv::Point2d> &pts2d_r,
                               std::vector<double> &p0, cv::Mat &J, cv::Mat &res, bool reweigh) {
    //6 parameters to be estimated
    double rx, ry, rz, tx, ty, tz;
    //sin and cossine of the angles
    double sx, cx, sy, cy, sz, cz;

    // extract motion parameters
    rx=p0.at(0); ry=p0.at(1); rz=p0.at(2);
    tx=p0.at(3); ty=p0.at(4); tz=p0.at(5);

    // precompute sine/cosine
    sx=sin(rx); cx=cos(rx); sy=sin(ry);
    cy=cos(ry); sz=sin(rz); cz=cos(rz);

    // compute rotation matrix and derivatives
    double r00=+cy*cz;
    double r01=-cy*sz;
    double r02=+sy;
    double r10=+sx*sy*cz+cx*sz;
    double r11=-sx*sy*sz+cx*cz;
    double r12=-sx*cy;
    double r20=-cx*sy*cz+sx*sz;
    double r21=+cx*sy*sz+sx*cz;
    double r22=+cx*cy;

    double rdrx10=+cx*sy*cz-sx*sz;
    double rdrx11=-cx*sy*sz-sx*sz;
    double rdrx12=-cx*cy;
    double rdrx20=+sx*sy*cz+cx*sz;
    double rdrx21=-sx*sy*sz+cx*cz;
    double rdrx22=-sx*cy;
    double rdry00=-sy*cz;
    double rdry01=+sy*sz;
    double rdry02=+cy;
    double rdry10=+sx*cy*cz;
    double rdry11=-sx*cy*sz;
    double rdry12=+sx*sy;
    double rdry20=-cx*cy*cz;
    double rdry21=+cx*cy*sz;
    double rdry22=-cx*sy;
    double rdrz00=-cy*sz;
    double rdrz01=-cy*cz;
    double rdrz10=-sx*sy*sz+cx*cz;
    double rdrz11=-sx*sy*cz-cx*sz;
    double rdrz20=+cx*sy*sz+sx*cz;
    double rdrz21=+cx*sy*cz-sx*sz;

    //aux variables
    //3D point computed in previous camera coordinate
    double X1p,Y1p,Z1p;
    //3D point in each next camera coordinate
    double X1c,Y1c,Z1c,X2c, Y2c, Z2c;
    //
    double X1cd,Y1cd,Z1cd;


    for (int i=0; i<numPts; i++){

        // get 3d point in previous coordinate system
        X1p=pts3D.at(i).x;
        Y1p=pts3D.at(i).y;
        Z1p=pts3D.at(i).z;

        // compute 3d point in current left coordinate system
        X1c = r00*X1p+r01*Y1p+r02*Z1p+tx;
        Y1c = r10*X1p+r11*Y1p+r12*Z1p+ty;
        Z1c = r20*X1p+r21*Y1p+r22*Z1p+tz;

        // compute 3d point in current right coordinate system
        X2c = X1c-baseline;
        Y2c = Y1c;
        Z2c = Z1c;

        // weighting
        double weight = 1.0;

        //give more significance to features located closer to the image center in horizontal direction
        //the value 0.05 depends on the stereo camera and lens setup, was empirically set
        if (reweigh)
            weight = 1.0/(fabs(pts2d_l.at(i).x - uc)/fabs(uc) + 0.05);

        // for all parameters: 3 for rotation, 3 for traslation
        for (int j=0; j<6; j++) {

            // derivatives of Xc, Yc, Zc with respect to all parameters
            switch (j) {

                //theta x
                case 0:
                    X1cd = 0;
                    Y1cd = rdrx10*X1p+rdrx11*Y1p+rdrx12*Z1p;
                    Z1cd = rdrx20*X1p+rdrx21*Y1p+rdrx22*Z1p;
                    break;
                //theta y
                case 1:
                    X1cd = rdry00*X1p+rdry01*Y1p+rdry02*Z1p;
                    Y1cd = rdry10*X1p+rdry11*Y1p+rdry12*Z1p;
                    Z1cd = rdry20*X1p+rdry21*Y1p+rdry22*Z1p;
                    break;
                 // theta z
                case 2:
                    X1cd = rdrz00*X1p+rdrz01*Y1p;
                    Y1cd = rdrz10*X1p+rdrz11*Y1p;
                    Z1cd = rdrz20*X1p+rdrz21*Y1p;
                    break;
                //tx
                case 3: X1cd = 1; Y1cd = 0; Z1cd = 0; break;
                //ty
                case 4: X1cd = 0; Y1cd = 1; Z1cd = 0; break;
                //tz
                case 5: X1cd = 0; Y1cd = 0; Z1cd = 1; break;
            }

            // set jacobian entries (project via K)
            J.at<double>(4*i+0,j)   = weight*fu*(X1cd*Z1c-X1c*Z1cd)/(Z1c*Z1c);  // left u
            J.at<double>(4*i+1,j)   = weight*fu*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c);  // left v

            J.at<double>(4*i+2,j)   = weight*fv*(X1cd*Z2c-X2c*Z1cd)/(Z2c*Z2c);  // right u'
            J.at<double>(4*i+3,j)   = weight*fv*(Y1cd*Z2c-Y2c*Z1cd)/(Z2c*Z2c);  // right v'

        }

        // set prediction (project via K)
        double pred_u1 = fu*X1c/Z1c+uc; //  left u;
        double pred_v1 = fv*Y1c/Z1c+vc; //  left v

        double pred_u2 = fu*X2c/Z2c+uc; // right u
        double pred_v2 = fv*Y2c/Z2c+vc; // right v

        // set residuals
        res.at<double>(4*i+0)   = weight*(pts2d_l.at(i).x - pred_u1);
        res.at<double>(4*i+1)   = weight*(pts2d_l.at(i).y - pred_v1);

        res.at<double>(4*i+2)   = weight*(pts2d_r.at(i).x - pred_u2);
        res.at<double>(4*i+3)   = weight*(pts2d_r.at(i).y - pred_v2);

    }

}


int Tracking::checkInliers(const std::vector<cv::Point3f> &pts3d, const std::vector<cv::Point2f> &pts2dl,
                           const std::vector<cv::Point2f> &pts2dr, const std::vector<int> &index,
                           const std::vector<double> &p0, std::vector<bool> &inliers, long double &sumErr, bool reweigh, long double &stdDev) {

    //6 parameters to be estimated
    double rx, ry, rz, tx, ty, tz;
    //sin and cossine of the angles
    double sx, cx, sy, cy, sz, cz;

    // extract motion parameters
    rx=p0.at(0); ry=p0.at(1); rz=p0.at(2);
    tx=p0.at(3); ty=p0.at(4); tz=p0.at(5);

    // precompute sine/cosine
    sx=sin(rx); cx=cos(rx); sy=sin(ry);
    cy=cos(ry); sz=sin(rz); cz=cos(rz);

    // compute rotation matrix and derivatives
    double r00=+cy*cz;
    double r01=-cy*sz;
    double r02=+sy;
    double r10=+sx*sy*cz+cx*sz;
    double r11=-sx*sy*sz+cx*cz;
    double r12=-sx*cy;
    double r20=-cx*sy*cz+sx*sz;
    double r21=+cx*sy*sz+sx*cz;
    double r22=+cx*cy;

    //aux variables
    //3D point computed in previous camera coordinate
    double X1p,Y1p,Z1p;
    //3D point in each next camera coordinate
    double X1c,Y1c,Z1c,X2c, Y2c, Z2c;

    int numInliers = 0;

    std::vector<double > errorVect;
    errorVect.reserve(pts3d.size());
    long double meanError = 0;
    for (int i = 0; i < pts3d.size(); i++){

        //validadte against other elements
        if(!(std::find(index.begin(), index.end(), i) != index.end())){

            // weighting
            double weight = 1.0;

            //give more significance to features located closer to the image center in horizontal direction
            //the value 0.05 depends on the stereo camera and lens setup, was empirically set
            if (reweigh)
                weight = 1.0/(fabs(pts2dl.at(i).x - uc)/fabs(uc) + adjustValue);

            // get 3d point in previous coordinate system
            X1p=pts3d.at(i).x;
            Y1p=pts3d.at(i).y;
            Z1p=pts3d.at(i).z;

            // compute 3d point in current left coordinate system
            X1c = r00*X1p+r01*Y1p+r02*Z1p+tx;
            Y1c = r10*X1p+r11*Y1p+r12*Z1p+ty;
            Z1c = r20*X1p+r21*Y1p+r22*Z1p+tz;

            // compute 3d point in current right coordinate system
            X2c = X1c-baseline;
            Y2c = Y1c;
            Z2c = Z1c;


            double pred_u1 = fu*X1c/Z1c+uc; //  left u;
            double pred_v1 = fv*Y1c/Z1c+vc; //  left v

            double pred_u2 = fu*X2c/Z2c+uc; // right u
            double pred_v2 = fv*Y2c/Z2c+vc; // right v

            // set residuals
            double rx0 = weight*(pts2dl.at(i).x - pred_u1);
            double ry0 = weight*(pts2dl.at(i).y - pred_v1);

            double rx1 = weight*(pts2dr.at(i).x - pred_u2);
            double ry1 = weight*(pts2dr.at(i).y - pred_v2);

            double d     = rx0*rx0+ry0*ry0+rx1*rx1+ry1*ry1;
            sumErr      += d;

            if( d < ransacTh*ransacTh){
                errorVect.push_back(d) ;
                meanError += d;
                inliers[i] = true;
                numInliers++;
            }
        }else{
            inliers[i] = true;
            numInliers++;
        }
    }

    if(numInliers != 0){
        meanError /= (long double) numInliers;

        for (unsigned int p=0; p<errorVect.size(); p++)
        {
            long double delta = errorVect[p]-meanError;
            stdDev += delta*delta;
        }

        stdDev /= (double)(numInliers);
    } else
        stdDev = 0;


//    std::cout << "Num inliers: "    << numInliers   << std::endl;
//    std::cout << "Sum error: "      << sumErr       << std::endl;
//    std::cout << "Error Vec size: " << errorVect.size() << std::endl;

    std::vector<double >().swap(errorVect);

    return numInliers;


}

std::vector<int> Tracking::generateRandomIndices(const unsigned long &maxIndice, const int &vecSize){
    std::vector<int> randValues;
    int index;

    do{
        index = rand() % maxIndice;
        if(!(std::find(randValues.begin(), randValues.end(), index) != randValues.end())){
            randValues.push_back(index);
        }

    }while(randValues.size() < vecSize);

    return randValues;
}

double Tracking::euclideanDist(const cv::Point2d &p, const cv::Point2d &q) {
    Point2d diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}


void Tracking::stereoMatching(const std::vector<cv::Point2f> &pts_l, const std::vector<cv::Point2f> &pts_r,
                              const cv::Mat &imLeft, const cv::Mat &imRight, std::vector<cv::DMatch> &matches,
                              std::vector<cv::Point2f> &new_pts_l, std::vector<cv::Point2f> &new_pts_r,
                              std::vector<cv::Point3f> &pointCloud, double &meanError, std::vector<bool> &ptsClose) {

    std::vector<Point2f> aux_pts_r(pts_r);
    Mat im;
#if LOG_DRAW
    im = imLeft.clone();
    cv::cvtColor(im, im, cv::COLOR_GRAY2RGB);
#endif

    //Assign keypoints on right image to a row table
    std::vector<std::vector<std::size_t>> vecRowIndices (imRight.rows, std::vector<std::size_t>());

    for (int i=0; i<imRight.rows; i++)
        vecRowIndices[i].reserve(pts_l.size());

    const int nRpts = pts_r.size();

    for(int iR=0; iR < nRpts; iR++){

        const Point2f &pt   = pts_r[iR];
        const float pt_y    = pt.y;

        const int yi = round(pt_y);
        //push the point index on the vector of points in right image by it's y coordinate
        vecRowIndices[yi].push_back(iR);

    }

    int pos = 0;
//    int index_l = 0;
    double sum = 0;
    int numPtsClose= 0;
    int numMatches = 0;
    for (auto &pt_l:pts_l) {

        Point2f ptr;
        Point3f pt3D;
        int index;
        //find point correspondece in the right image using epipolar constraints
        bool found = findMatchingSAD(pt_l, imLeft, imRight, aux_pts_r, ptr, index, vecRowIndices);
        if(found){
            numMatches ++;
            //check if the point have a good triangulation
            double error = 0.0;
            double depth = 0.0;
            if(triangulation(pt_l, ptr, pt3D, error, depth)){


                new_pts_l.push_back(pt_l);
                new_pts_r.push_back(ptr);

                pointCloud.push_back(pt3D);

                if (depth > thDepth*baseline ){
                    ptsClose.push_back(false);
                }
                else{
                    numPtsClose ++;
                    ptsClose.push_back(true);
                }

                double dst = euclideanDist(pt_l, ptr);
                DMatch match(pos, pos, dst);
                matches.push_back(match);
                pos++;


            }

            sum += error;
        }

    }

#if LOG_DRAW

    for (int i = 0 ; i < ptsClose.size() ; i++){
        if(ptsClose[i])
            drawFarAndClosePts(new_pts_l[i], Scalar(0, 255, 0), im);
        else
            drawFarAndClosePts(new_pts_l[i], Scalar(0, 0, 255), im);
    }

    imwrite("dstPts.png", im);
#endif

#if LOG
    writeOnLogFile("Num points close:", std::to_string(numPtsClose));
#endif

    //Free memory
    std::vector<Point2f>().swap(aux_pts_r);

    meanError = sum/(numMatches);


}


bool Tracking::findMatchingSAD(const cv::Point2f &pt_l, const cv::Mat &imLeft, const cv::Mat &imRight,
                               std::vector<cv::Point2f> &pts_r, cv::Point2f &ptr_m, int &index, const std::vector<std::vector<std::size_t>> &vecRowIndices) {


    int blockSize = 2 * halfBlockSize + 1;

    int width = imRight.size().width;
    int height = imRight.size().height;

    double meanP    = 0; //mean of intensities on base template matching
    double stdP     = 0;

    Mat template_(blockSize, blockSize, CV_64F);
    //get pixel neighbors
    //        Mat template_ = imLeft(Rect ((int)pt.x - halfBlockSize, (int)pt.y - halfBlockSize, halfBlockSize, halfBlockSize)).clone();
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < blockSize; j++) {
            int x = (int) pt_l.x - (halfBlockSize - i);
            int y = (int) pt_l.y - (halfBlockSize - j);
            //check frame limits
            if (x >= 0 && x < width && y >= 0 && y < height) {
                Scalar intensity = imLeft.at<uchar>(y, x);
                template_.at<float>(j, i) = (int) intensity[0];
                meanP += (int) intensity[0];
            } else {
                template_.at<float>(j, i) = 0;
                meanP += 0;
            }
        }
    }

    meanP /= (blockSize*blockSize);

    for (int i=0; i < blockSize; i++){
        for (int j = 0; j < blockSize; j++) {

            double delta = (template_.at<float>(j, i) - meanP);
            stdP += (delta * delta);
        }
    }



    const float &vL = pt_l.y;
    const float &uL = pt_l.x;

    const int yi = round(vL);

    const std::vector<std::size_t> &vecCandidates = vecRowIndices[yi];
    if(vecCandidates.empty())
        return false;

    double minSAD = sadMinValue;
    Point2f bestPt;

    //flag to know when the point has no matching
    bool noMatching = true;

    int bestIndex_r = 0;
    //find que point with the lowest SAD
    for (size_t index=0; index < vecCandidates.size(); index++) {

        const size_t iR = vecCandidates[index];

        const Point2f &pt_r = pts_r[iR];


        //check if the point was matched before
        if(pt_r.x == -1 && pt_r.y == -1)
            continue;

        int deltay = (int) abs(pt_l.y - pt_r.y);
        int deltax = (int) pt_l.x - (int) pt_r.x;



        //epipolar constraints, the correspondent keypoint must be at the same row and disparity should be positive
        if (deltax >= minDisp && deltax <= maxDisp) {

            //compute SAD
            double meanC    = 0; //mean of intensities of current template
            double stdC     = 0; // standard deviation

            Mat templateC (blockSize, blockSize, CV_64F);
            for (int i = 0; i < blockSize; i++) {
                for (int j = 0; j < blockSize; j++) {
                    int x = (int) pt_r.x - (halfBlockSize - i);
                    int y = (int) pt_r.y - (halfBlockSize - j);
                    //check frame limits
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        Scalar intensity = imRight.at<uchar>(y, x);
                        templateC.at<float>(j, i) = (int) intensity[0];
                        meanC += (int) intensity[0];
                    } else {
                        templateC.at<float>(j, i) = 0;
                        meanC += 0;
                    }
                }
            }

            meanC /= (blockSize*blockSize);

            // compute standard deviation
            for (int i=0; i < blockSize; i++){
                for (int j = 0; j < blockSize; j++) {
                    double delta =  (templateC.at<float>(j, i) - meanC);
                    stdC += (delta*delta);
                }
            }

            double nSAD = 0;
            //compute NSAD
            for (int i=0; i < blockSize; i++){
                for (int j = 0; j < blockSize; j++) {
                    double nIp = (template_.at<float>(j,i) - meanP) / stdP;
                    double nIc = (templateC.at<float>(j,i) - meanC) / stdC;

                    nSAD += abs(nIp - nIc);
                }
            }

            if (nSAD < minSAD) {
                noMatching = false;
                minSAD = nSAD;
                bestPt = pt_r;
                bestIndex_r = iR;
            }
        }


    }


    if (!noMatching) {
        // a way to not compare with points already matched
        pts_r[bestIndex_r] = Point (-1,-1);
        ptr_m = bestPt;
        index = bestIndex_r;

        return true;
    }else{
        return false;
    }
}

void Tracking::quadMatching(const std::vector<cv::Point3f> &pts3D, const std::vector<cv::Point2f> &pts2D_l,
                            const std::vector<cv::Point2f> &pts2D_r, std::vector<bool> &inliers, const cv::Mat &imLeft,
                            const cv::Mat &imRight, std::vector<cv::Point3f> &new_pts3D, std::vector<cv::Point2f> &new_pts2D_l,
                            std::vector<cv::Point2f> &new_pts2D_r, std::vector<cv::DMatch> &matches, const std::vector<bool> &ptsClose) {

    std::vector<Point2f> aux_pts_r(pts2D_r);

    //Assign keypoints on right image to a row table
    std::vector<std::vector<std::size_t>> vecRowIndices (imRight.rows+1, std::vector<std::size_t>());

    for (int i=0; i<imRight.rows; i++)
        vecRowIndices[i].reserve(pts2D_r.size());


    const int nRpts = pts2D_r.size();

    for(int iR=0; iR < nRpts; iR++){


        const Point2f &pt   = pts2D_r[iR];

        const float pt_y    = pt.y;

        const int yi = round(pt_y);


        //push the point index on the vector of points in right image by it's y coordinate
        vecRowIndices[yi].push_back(iR);

    }


    int pos = 0;
    for (int i = 0; i < inliers.size(); i++){

        if(inliers.at(i) /*&& ptsClose.at(i)*/){

            Point2f pt2Dr;

            int index;

            bool found = findMatchingSAD(pts2D_l.at(i), imLeft, imRight, aux_pts_r, pt2Dr, index, vecRowIndices);

            if(found){
                new_pts3D.push_back(pts3D.at(i));
                new_pts2D_l.push_back(pts2D_l.at(i));
                new_pts2D_r.push_back(pts2D_r.at(index));

                double dst = euclideanDist(pts2D_l.at(i), pts2D_r.at(index));
                DMatch match(pos, pos, dst);
                matches.push_back(match);
                pos++;

            }

        }

    }


}

void Tracking::essentialMatrixDecomposition(const cv::Mat &F, const cv::Mat &K, const cv::Mat &K_l,
                                            const std::vector<cv::Point2f> &pts_l,
                                            const std::vector<cv::Point2f> &pts_r, std::vector<bool> &inliers, cv::Mat &R_est, cv::Mat &t_est) {


    Mat E = K_l.t() * F * K;

    double mW [3][3];

    mW[0][0] = 0;  mW[0][1] = -1; mW[0][2] = 0;
    mW[1][0] = 1;  mW[1][1] = 0 ; mW[1][2] = 0;
    mW[2][0] = 0;  mW[2][1] = 0 ; mW[2][2] = 1;

    Mat W (3,3, CV_64F, mW);

    double mDiag [3][3];

    mDiag[0][0] = 1;  mDiag[0][1] = 0; mDiag[0][2] = 0;
    mDiag[1][0] = 0;  mDiag[1][1] = 1; mDiag[1][2] = 0;
    mDiag[2][0] = 0;  mDiag[2][1] = 0; mDiag[2][2] = 0;

    Mat Diag (3,3, CV_64F, mDiag);

    Mat D, U, Vt;

    SVD::compute(E, D, U, Vt, SVD::MODIFY_A| SVD::FULL_UV);

//    Mat newE = U*Diag*Vt;
//    SVD::compute(newE, D, U, Vt, SVD::MODIFY_A| SVD::FULL_UV);

    Mat R1 =  U * W * Vt;
    if(determinant(R1) < 0)
        R1 = -R1;

    Mat R2 = U * W.t() * Vt;
    if(determinant(R2) < 0)
        R2 = -R2;

    checkSolution(R1,R2, U.col(2), K, K, pts_l, pts_r, R_est, t_est, inliers);

#if LOG
    writeOnLogFile("det(R) of E:", std::to_string(determinant(R_est)));
#endif

}

void Tracking::checkSolution(const cv::Mat &R1, const cv::Mat &R2, const cv::Mat &u3, const cv::Mat &K, const cv::Mat &K_l,
        const std::vector<cv::Point2f> &pts_l, const std::vector<cv::Point2f> &pts_r, cv::Mat &R_est,
        cv::Mat &t_est, std::vector<bool> &inliers) {


    cv::Mat P    = cv::Mat::eye(3,4,CV_64F);
    cv::Mat P_l  = cv::Mat::eye(3,4,CV_64F);

//    std::vector<bool> tmpInliers (inliers.size());
//    tmpInliers.swap(inliers);

    cv::Mat R;
    cv::Mat u3_;

    cv::Mat R_best, t_best;

    int bestNumPts = 0;

    //compute the 4 possible solutions
    for (int i = 0; i < 4; i++){

        int numPts = 0; //number of points in front of the cameras

        switch (i){

            //------- solution 1
            // [UWVt | +u3]
            case 0:
                R   = R1;
                u3_ = u3.clone();
                break;
            //------- solution 2
            // [UWVt | -u3]
            case 1:
                R   = R1;
                u3_  = -1 * u3.clone();
                break;
            //------ solution 3
            // [UWtVt | +u3]
            case 2:
                R = R2;
                u3_ = 1 * u3.clone();
                break;
            //------ solution 4
            // [UWtVt | -u3]
            case 3:
                R = R2;
                u3_  = -1 * u3.clone();
                break;
        }

//        std::vector<bool> bestSetInliers (tmpInliers.size(), false);

        for (int j = 0; j < inliers.size() /*tmpInliers.size()*/; j++ ){

            if(/*tmpInliers[j]*/inliers[j]){

                Mat x_r = cv::Mat::zeros(3,1, CV_64F);
                x_r.at<double>(0) = pts_r[j].x;
                x_r.at<double>(1) = pts_r[j].y;
                x_r.at<double>(2) = 1.0;

                Mat x_l = cv::Mat::zeros(3,1, CV_64F);
                x_l.at<double>(0) = pts_l[j].x;
                x_l.at<double>(1) = pts_l[j].y;
                x_l.at<double>(2) = 1.0;

                if(pointFrontCamera(R,u3_,x_l, x_r, P, P_l, K, K_l)){
//                    bestSetInliers[j] = true;
                    numPts ++;
                }
            }
        }

        if (numPts > bestNumPts){
            R_best      = R.clone();
            t_best      = u3_.clone();
//            inliers     = bestSetInliers;
            bestNumPts  = numPts;
        }
    }

#if LOG
    writeOnLogFile("Num points in front of the camera: ", std::to_string(bestNumPts));
//    std::cout << "Num points in front of the camera: " << bestNumPts << std::endl;
#endif

    R_est   = R_best.clone();
    t_est  = t_best.clone();

    //Free memory
//    std::vector<bool>().swap(tmpInliers);


}

bool Tracking::pointFrontCamera(cv::Mat &R2, const cv::Mat &t2, const cv::Mat &pt_l, const cv::Mat &pt_r, const cv::Mat &P, cv::Mat &P_l,
                                const cv::Mat &K, const cv::Mat &K_l) {

    R2.copyTo(P_l.rowRange(0,3).colRange(0,3));
    t2.copyTo(P_l.rowRange(0,3).col(3));

    Mat R1 = cv::Mat::eye(3,3, CV_64F);
    Mat t1 = cv::Mat::zeros(3,1, CV_64F);

    // Linear Triangulation Method
    cv::Mat A = Mat::zeros(4,4,CV_64F);

    A.row(0) = pt_l.at<double>(0)*P.row(2)   -   P.row(0);
    A.row(1) = pt_l.at<double>(1)*P.row(2)   -   P.row(1);
    A.row(2) = pt_r.at<double>(0)*P_l.row(2) -   P_l.row(0);
    A.row(3) = pt_r.at<double>(1)*P_l.row(2) -   P_l.row(1);

    cv::Mat D,U,Vt;
    cv::SVD::compute(A,D,U,Vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

    Mat pt3D = Vt.row(3).t();

    if(pt3D.at<double>(3) == 0)
        std::cerr << "pt3D.at<float>(3) == 0 \n";

    //Euclidean coordinates
    pt3D = pt3D.rowRange(0,3)/pt3D.at<double>(3);

    Mat pt3D_t = pt3D.t();

    double Z1c = R2.row(2).dot(pt3D_t) + t2.at<double>(2);

    double Z2c = R1.row(2).dot(pt3D_t) + t1.at<double>(2);

    return Z1c > 0 && Z2c > 0;

}


bool Tracking::triangulation(const cv::Point2f &kp_l, const cv::Point2f &kp_r, cv::Point3f &pt3d, double &error, double &depth) {


    double w0, w1;
    double dist;

    w0 = w1 = 1.0;

    Mat point3d;
    double w;

    for(int j = 0; j < max_iter_3d; j++){

        Mat A   = Mat::zeros(4,4,CV_64F);
        Mat D, U, Vt;

        A.row(0) = w0*(kp_l.x*P1.row(2)-P1.row(0));
        A.row(1) = w0*(kp_l.y*P1.row(2)-P1.row(1));
        A.row(2) = w1*(kp_r.x*P2.row(2)-P2.row(0));
        A.row(3) = w1*(kp_r.y*P2.row(2)-P2.row(1));

        SVD::compute(A,D,U,Vt, SVD::MODIFY_A| SVD::FULL_UV);

        point3d = Vt.row(3).t();

        point3d = point3d.rowRange(0,4)/point3d.at<double>(3);

        Mat p0 = P1*point3d;
        Mat p1 = P2*point3d;


        w0 = 1.0/p0.at<double>(2);
        w1 = 1.0/p1.at<double>(2);

        double dx0 = kp_l.x - p0.at<double>(0)/p0.at<double>(2);
        double dy0 = kp_l.y - p0.at<double>(1)/p0.at<double>(2);
        double dx1 = kp_r.x - p1.at<double>(0)/p1.at<double>(2);
        double dy1 = kp_r.y - p1.at<double>(1)/p1.at<double>(2);

        dist = sqrt(dx0*dx0+dy0*dy0) + sqrt(dx1*dx1+dy1*dy1);

        w = p0.at<double>(2);
    }

//    Mat M   = P1.rowRange(0,3).colRange(0,3);
//    Mat m3  = M.row(2);
//
//    depth = (sign(determinant(M))*w)/cv::norm(m3);

    depth = (baseline * fu)/(kp_l.x - kp_r.x);

//    std::cout << "Depth: " << depth << std::endl;

    pt3d.x           = (float) point3d.at<double>(0);
    pt3d.y           = (float) point3d.at<double>(1);
    pt3d.z           = (float) point3d.at<double>(2);

//    std::cout << "Z coordinate: " << pt3d.z << std::endl;

    error = dist;

    if (dist < 2*th_3d)
        return true;
    else
        return false;

}



void Tracking::featureExtraction(const cv::Mat &im0, const cv::Mat &im1, std::vector<KeyPoint> &kpts0,
                                 std::vector<KeyPoint> &kpts1, std::vector<Point2f> &pts0,
                                 std::vector<Point2f> &pts1) {

    std::thread orbThreadLeft (&Tracking::extractORB, this, 0, std::ref(im0), std::ref (kpts0), std::ref (pts0));
    std::thread orbThreadRight (&Tracking::extractORB, this, 1, std::ref(im1), std::ref (kpts1), std::ref(pts1));

    orbThreadLeft.join();
    orbThreadRight.join();

#if LOG
    logFeatureExtraction(kpts0, kpts1, pts0, im0);
#endif
}

void Tracking::featureTracking(const cv::Mat &imL0, const cv::Mat &imL1, const cv::Mat &imR0, const cv::Mat &imR1, std::vector<Point2f> &ptsL0,
                               std::vector<Point2f> &ptsL1, std::vector<Point2f> &ptsR0, std::vector<Point2f> &ptsR1,
                               std::vector<Point3f> &pts3D, std::vector<bool> &ptsClose ) {

    std::vector <Mat> left0_pyr, left1_pyr, right0_pyr, right1_pyr;
    Size win (winSize,winSize);
    int maxLevel = pyrMaxLevel;
    std::vector<uchar> status0, status1;
    std::vector<float > error0, error1;

    std::thread kltThreadLeft (&Tracking::opticalFlowFeatureTrack, this, std::ref(imL0), std::ref(imL1), win,
                               maxLevel, std::ref(status0), std::ref(error0), std::ref(ptsL0), std::ref(ptsL1),
                               std::ref(left0_pyr), std::ref(left1_pyr), 0, std::ref(pts3D), std::ref(ptsClose));

    std::vector<Point3f> aux (0);
    std::vector<bool> aux1(0);
    std::thread kltThreadRight (&Tracking::opticalFlowFeatureTrack, this, std::ref(imR0), std::ref(imR1), win,
                                maxLevel, std::ref(status1), std::ref(error1), std::ref(ptsR0), std::ref(ptsR1),
                                std::ref(right0_pyr), std::ref(right1_pyr), 1, std::ref(aux), std::ref(aux1));

    kltThreadLeft.join();
    kltThreadRight.join();


}


void Tracking::opticalFlowFeatureTrack(const cv::Mat &imT0, const cv::Mat &imT1, Size win, int maxLevel, std::vector<uchar> &status, std::vector<float> &error,
                                       std::vector<Point2f> &prevPts, std::vector<Point2f> &nextPts, std::vector <Mat> imT0_pyr,
                                       std::vector <Mat> imT1_pyr, int flag, std::vector<Point3f> &pts3D, std::vector<bool> &ptsClose) {


//    std::vector <Mat> imT0_pyr, imT1_pyr;
    std::lock_guard<std::mutex> lock1(mtx1);
    buildOpticalFlowPyramid(imT0, imT0_pyr, win, maxLevel, true);
    std::lock_guard<std::mutex> lock2(mtx2);
    buildOpticalFlowPyramid(imT1, imT1_pyr, win, maxLevel, true);
    std::lock_guard<std::mutex> lock3(mtx3);
    calcOpticalFlowPyrLK(imT0_pyr, imT1_pyr, prevPts, nextPts, status, error, win, maxLevel,
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.03), 1);

//    std::lock_guard<std::mutex> lock4(mtx4);
    checkPointOutBounds(prevPts, nextPts, imT1, status, flag, pts3D, ptsClose);

}

void Tracking::checkPointOutBounds(std::vector<Point2f> &prevPts, std::vector<Point2f> &nextPts,const cv::Mat &imT1,
                                   const  std::vector<uchar> &status, int flag, std::vector<Point3f> &pts3, std::vector<bool> &ptsClose) {

    std::vector<Point2f> tmpPrevPts (prevPts.size());
    std::vector<Point2f> tmpNexPts (nextPts.size());
    std::vector<Point3f> tmpPts3D (pts3.size());
    std::vector<bool>    tmpPtsClose(ptsClose.size());

    if (flag == 0){
//        std::vector<Point3f> tmpPts3D (pts3.size());
        tmpPts3D.swap(pts3);
        pts3.clear();
        tmpPtsClose.swap(ptsClose);
        ptsClose.clear();
    }

    tmpPrevPts.swap(prevPts);
    prevPts.clear();
    tmpNexPts.swap(nextPts);
    nextPts.clear();


    for (int i=0; i< tmpPrevPts.size(); i++){

        const Point2f &pt_r = tmpNexPts[i];
        const Point2f &pt_l = tmpPrevPts[i];

        const Point3f &pt3d     = tmpPts3D[i];
        const bool &ptClose     = tmpPtsClose[i];

        if(status[i] == 1 && pt_r.x >= 0 && pt_r.x <= imT1.cols && pt_r.y >= 0 && pt_r.y <= imT1.rows){

            prevPts.push_back(pt_l);
            nextPts.push_back(pt_r);

            if(flag == 0){
                pts3.push_back(pt3d);
                ptsClose.push_back(ptClose);

            }


        }
    }

    //Free memory
    std::vector<Point2f>().swap(tmpPrevPts);
    std::vector<Point2f>().swap(tmpNexPts);
    std::vector<Point3f>().swap(tmpPts3D);


}
void Tracking::outlierRemovalAndMotionEstimation(const cv::Mat &imL0, const std::vector<Point2f> &ptsL0,
                                                 const cv::Mat &imL1, const std::vector<Point2f> &ptsL1,
                                                 const cv::Mat &imR0, const std::vector<Point2f> &ptsR0,
                                                 const cv::Mat &imR1, const std::vector<Point2f> &ptsR1,
                                                 std::vector<bool> &inliers, std::vector<double> &rvec_est, cv::Mat &t_est) {

    Mat fmat;
    std::vector<DMatch> mll, mrr;
    (*mEightPointLeft) (ptsL0, ptsL1, mll, inliers, true, 0, fmat);

#if LOG_DRAW
    mEightPointLeft->drawEpLines(ptsL0, ptsL1, fmat, inliers, 0, imL0, imL1, mll);
#endif

#if LOG
    writeOnLogFile("RANSAC num iterations:", std::to_string(mEightPointLeft->getRansacNumit()));
    logFeatureTracking(ptsL0, ptsR1, fmat, ptsL1, inliers, imL0, imL1, mll);
    writeOnLogFile("Num of inliers tracking:", std::to_string(mll.size()));
    writeOnLogFile("det(F):", std::to_string(determinant(fmat)));
    ptsTracking.push_back(mll.size());
    ransacIt_8point.push_back(mEightPointLeft->getRansacNumit());
#endif

//    Mat R_est;
//    essentialMatrixDecomposition(fmat, K, K, ptsL0, ptsL1, inliers, R_est, t_est);
//
//    Rodrigues(R_est, rvec_est, noArray());


}

void Tracking:: relativePoseEstimation(const std::vector<cv::Point2f> &pts2DL, const std::vector<cv::Point2f> &pts2DR,
                                      const std::vector<cv::Point3f> &pts3D, const std::vector<double> &rvec_est, 
                                      const cv::Mat &t_est , cv::Mat &Tcw_, cv::Mat &cov_mat) {


    //initialize vector of parameters with rotation and translation from essential matrix
    std::vector<double> p0 (6, 0.0);
//    p0.at(0) = rvec_est.at(0);
//    p0.at(1) = rvec_est.at(1);
//    p0.at(2) = rvec_est.at(2);
//    p0.at(3) = t_est.at<double>(0);
//    p0.at(4) = t_est.at<double>(1);
//    p0.at(5) = t_est.at<double>(2);

    std::vector<bool> inliers2;
    inliers2.reserve(pts3D.size());

    std::vector<double> p (6, 0.0);
    int bestNumInliers = ransacMinSet;
    poseEstimationRansac(pts2DL, pts2DR, pts3D, p0, inliers2, p, reweigh, bestNumInliers, cov_mat);

    //pose refinment with all inliers
    Mat rot_vec = cv::Mat::zeros(3,1, CV_64F);
    Mat tr_vec  = cv::Mat::zeros(3,1, CV_64F);

    poseRefinment(pts2DL, pts2DR, pts3D, inliers2, p, rot_vec, tr_vec, bestNumInliers, cov_mat);

    Mat Rotmat;
    Rodrigues(rot_vec, Rotmat, noArray());

    Rotmat.copyTo(Tcw_.rowRange(0,3).colRange(0,3));
    tr_vec.copyTo(Tcw_.rowRange(0,3).col(3));
#if LOG
    writeOnLogFile("Rotation matrix det(): ", std::to_string(cv::determinant(Rotmat)));
#endif


}

void Tracking::poseRefinment(const std::vector<Point2f> &pts2DL, const std::vector<Point2f> &pts2DR,
                             const std::vector<Point3f> &pts3D, const std::vector<bool> &inliers, std::vector<double> &p, 
                             cv::Mat &rot_vec, cv::Mat &tr_vec, const int &numInliers, cv::Mat &cov_mat) {

//    std::cout << "Num inliers for refinment: " << numInliers << std::endl;
    std::vector<Point2d> inPts_l1, inPts_r1;
    inPts_l1.reserve(numInliers);
    inPts_r1.reserve(numInliers);

    std::vector<Point3d> inPts_3D;
    inPts_3D.reserve(numInliers);

    for (int i=0; i<inliers.size(); i++){
        if(inliers.at(i)){
            Point2f aux1 = pts2DL[i];
            inPts_l1.push_back(aux1);
            Point2f aux2 = pts2DR[i];
            inPts_r1.push_back(aux2);
            Point3f aux3 = pts3D[i];
            inPts_3D.push_back(aux3);
        }
    }



    // pose refinement with all inliers
    int status = 0;
    for (int i = 0; i < finalMaxIteration; i++){
        status = poseEstimation(inPts_l1, inPts_r1, inPts_3D, p, numInliers, reweigh, cov_mat);
        if(status != UPDATE)
            break;
    }



    rot_vec.at<double>(0) = p.at(0);
    rot_vec.at<double>(1) = p.at(1);
    rot_vec.at<double>(2) = p.at(2);

    tr_vec.at<double>(0)  = p.at(3);
    tr_vec.at<double>(1)  = p.at(4);
    tr_vec.at<double>(2)  = p.at(5);


//    rot_vec.at<double>(0) = 0.0;
//    rot_vec.at<double>(1) = p.at(1);
//    rot_vec.at<double>(2) = p.at(2);

//    tr_vec.at<double>(0)  = p.at(3);
//    tr_vec.at<double>(1)  = 0.0;
//    tr_vec.at<double>(2)  = p.at(5);


}

cv::Mat Tracking::getCurrentPose() {
    return Tcw;
}

void Tracking::saveTrajectoryEuroc(const string &filename) {

    ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    cv::Mat Twc = cv::Mat::eye(4,4,CV_32F);

    Mat R0 = Twc.rowRange(0,3).colRange(0,3);
    Mat t0 = Twc.rowRange(0,3).col(3);

//    std::vector<float> q0 =  toQuaternion(R0);
    std::vector<float> q0 =  mRot2Quat(R0);

    f << setprecision(6) << initTimestamp << " " <<  setprecision(9) << t0.at<float>(0) << " " << t0.at<float>(1) << " "
            << t0.at<float>(2) << " " << q0[3] << " " << q0[2] << " " << q0[1] << " " << q0[0] << endl;
//    f << setprecision(6) << initTimestamp << " " <<  setprecision(9) << t0.at<float>(0) << " " << t0.at<float>(1) << " "
//            << t0.at<float>(2) << " " << q0[0] << " " << q0[1] << " " << q0[2] << " " << q0[3] << endl;

    /*
        * The global pose is computed in reference to the first frame by concatanation
        * The current global pose is computed by
        * so Twc * inv(Tcw) where Tcw is current relative pose estimated and Twc is the last global pose
        * Initial Pwc = [I | 0]
    */
    std::list<cv::Mat>::iterator lit;
    std::list<double>::iterator lTime = frameTimeStamp.begin();
    for(lit = relativeFramePoses.begin(); lit != relativeFramePoses.end(); ++lit, ++lTime){


        //Compute the inverse of relative pose estimation inv(Tcw) = [R' | C]
        //where C = -1 * R' * t

        Mat rot_mat = cv::Mat::zeros(3,1, CV_64F);
        Mat tr_vec  = cv::Mat::zeros(3,1, CV_64F);

        rot_mat = (*lit).rowRange(0,3).colRange(0,3);
        tr_vec  = (*lit).col(3);

        cv::Mat Rt  = rot_mat.t();
        cv::Mat C   = -1 * Rt * tr_vec;

        cv::Mat Tcw_inv = cv::Mat::eye(4,4,CV_32F);
        Rt.convertTo(Rt, CV_32F);
        C.convertTo(C, CV_32F);

        Rt.copyTo(Tcw_inv.rowRange(0,3).colRange(0,3));
        C.copyTo(Tcw_inv.rowRange(0,3).col(3));

        Twc = Twc * Tcw_inv;

        Mat Rw = Twc.rowRange(0,3).colRange(0,3);
        Mat tw = Twc.rowRange(0,3).col(3);

//        std::vector<float> q =  toQuaternion(Rw);
//        std::vector<float> q =  mRot2Quat(Rw);

//        std::cout << "Rotation Matrix: " << Rw << std::endl;
        Eigen::Matrix<double,3,3> M;

        M <<    Rw.at<float>(0,0), Rw.at<float>(0,1), Rw.at<float>(0,2),
                Rw.at<float>(1,0), Rw.at<float>(1,1), Rw.at<float>(1,2),
                Rw.at<float>(2,0), Rw.at<float>(2,1), Rw.at<float>(2,2);

        Eigen::Quaterniond q(M);

//        std::cout << "quaternion: " <<  " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

//        f << setprecision(6) << (*lTime) << " " <<  setprecision(9) << tw.at<float>(0) << " " << tw.at<float>(1) << " "
//                << tw.at<float>(2) << " " << q[3] << " " << q[2] << " " << q[1] << " " << q[0] << endl;
        f << setprecision(6) << (*lTime) << " " <<  setprecision(9) << tw.at<float>(0) << " " << tw.at<float>(1) << " "
          << tw.at<float>(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

    }

    f.close();
    std::cout << endl << "trajectory saved on "<< filename << std::endl;

}


void Tracking::saveTrajectoryKitti(const string &filename) {

    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    cv::Mat Twc = cv::Mat::eye(4,4,CV_32F);

    f << setprecision(9) << Twc.at<float>(0,0) << " " << Twc.at<float>(0,1)  << " " << Twc.at<float>(0,2) << " "  << Twc.at<float>(0,3) << " " <<
    Twc.at<float>(1,0) << " " << Twc.at<float>(1,1)  << " " << Twc.at<float>(1,2) << " "  << Twc.at<float>(1,3) << " " <<
    Twc.at<float>(2,0) << " " << Twc.at<float>(2,1)  << " " << Twc.at<float>(2,2) << " "  << Twc.at<float>(2,3) << endl;
    /*
        * The global pose is computed in reference to the first frame by concatanation
        * The current global pose is computed by
        * so Twc * inv(Tcw) where Tcw is current relative pose estimated and Twc is the last global pose
        * Initial Pwc = [I | 0]
    */
    std::list<cv::Mat>::iterator lit;
    for(lit = relativeFramePoses.begin(); lit != relativeFramePoses.end(); ++lit){


        //Compute the inverse of relative pose estimation inv(Tcw) = [R' | C]
        //where C = -1 * R' * t

        Mat rot_mat = cv::Mat::zeros(3,1, CV_64F);
        Mat tr_vec  = cv::Mat::zeros(3,1, CV_64F);

        rot_mat = (*lit).rowRange(0,3).colRange(0,3);
        tr_vec  = (*lit).col(3);

//        std::cout << "det rot_mat: " << determinant(rot_mat) << std::endl;
//        std::cout << "tr_vec: "  << tr_vec << std::endl;

        cv::Mat Rt  = rot_mat.t();
        cv::Mat C   = -1 * Rt * tr_vec;

        cv::Mat Tcw_inv = cv::Mat::eye(4,4,CV_32F);
        Rt.convertTo(Rt, CV_32F);
        C.convertTo(C, CV_32F);

        Rt.copyTo(Tcw_inv.rowRange(0,3).colRange(0,3));
        C.copyTo(Tcw_inv.rowRange(0,3).col(3));

//        std::cout << "Tcw_inv: " << Tcw_inv << std::endl;

        Twc = Twc * Tcw_inv;

//        std::cout << "Twc: " << Twc << std::endl;

        f << setprecision(9) << Twc.at<float>(0,0) << " " << Twc.at<float>(0,1)  << " " << Twc.at<float>(0,2) << " "  << Twc.at<float>(0,3) << " " <<
        Twc.at<float>(1,0) << " " << Twc.at<float>(1,1)  << " " << Twc.at<float>(1,2) << " "  << Twc.at<float>(1,3) << " " <<
        Twc.at<float>(2,0) << " " << Twc.at<float>(2,1)  << " " << Twc.at<float>(2,2) << " "  << Twc.at<float>(2,3) << endl;

    }

    f.close();
    std::cout << endl << "trajectory saved on "<< filename << std::endl;
}

void Tracking::saveTrajectoryKitti8point(const string &filename)
{
    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    cv::Mat Twc = cv::Mat::eye(4,4,CV_32F);

    f << setprecision(9) << Twc.at<float>(0,0) << " " << Twc.at<float>(0,1)  << " " << Twc.at<float>(0,2) << " "  << Twc.at<float>(0,3) << " " <<
      Twc.at<float>(1,0) << " " << Twc.at<float>(1,1)  << " " << Twc.at<float>(1,2) << " "  << Twc.at<float>(1,3) << " " <<
      Twc.at<float>(2,0) << " " << Twc.at<float>(2,1)  << " " << Twc.at<float>(2,2) << " "  << Twc.at<float>(2,3) << endl;
    /*
        * The global pose is computed in reference to the first frame by concatanation
        * The current global pose is computed by
        * so Twc * inv(Tcw) where Tcw is current relative pose estimated and Twc is the last global pose
        * Initial Pwc = [I | 0]
    */
    std::list<cv::Mat>::iterator lit;
    for(lit = relativeFramePoses_.begin(); lit != relativeFramePoses_.end(); ++lit){


        //Compute the inverse of relative pose estimation inv(Tcw) = [R' | C]
        //where C = -1 * R' * t

        Mat rot_mat = cv::Mat::zeros(3,1, CV_64F);
        Mat tr_vec  = cv::Mat::zeros(3,1, CV_64F);

        rot_mat = (*lit).rowRange(0,3).colRange(0,3);
        tr_vec  = (*lit).col(3);

//        std::cout << "det rot_mat: " << determinant(rot_mat) << std::endl;
//        std::cout << "tr_vec: "  << tr_vec << std::endl;

        cv::Mat Rt  = rot_mat.t();
        cv::Mat C   = -1 * Rt * tr_vec;

        cv::Mat Tcw_inv = cv::Mat::eye(4,4,CV_32F);
        Rt.convertTo(Rt, CV_32F);
        C.convertTo(C, CV_32F);

        Rt.copyTo(Tcw_inv.rowRange(0,3).colRange(0,3));
        C.copyTo(Tcw_inv.rowRange(0,3).col(3));

//        std::cout << "Tcw_inv: " << Tcw_inv << std::endl;

        Twc = Twc * Tcw_inv;

//        std::cout << "Twc: " << Twc << std::endl;

        f << setprecision(9) << Twc.at<float>(0,0) << " " << Twc.at<float>(0,1)  << " " << Twc.at<float>(0,2) << " "  << Twc.at<float>(0,3) << " " <<
          Twc.at<float>(1,0) << " " << Twc.at<float>(1,1)  << " " << Twc.at<float>(1,2) << " "  << Twc.at<float>(1,3) << " " <<
          Twc.at<float>(2,0) << " " << Twc.at<float>(2,1)  << " " << Twc.at<float>(2,2) << " "  << Twc.at<float>(2,3) << endl;

    }

    f.close();
    std::cout << endl << "trajectory saved on "<< filename << std::endl;
}

void Tracking::saveStatistics(const string &filename, float &meanTime, bool withTime)
{

#if LOG
    std::ofstream f;
    f.open(filename.c_str());
    if(withTime){
        f<< "frame,time,Pts_detected,Pts_after_NMS,Pts_Stereo_Match,3D_reproj_error_mean,8-point_ransac_it,Pts_Tracking,Pts_quad_match,"
            "GN_it,GN_mean_it,GN_num_inliers,Pose_entropy,mean_time\n";
    }else {
        f<< "frame,Pts_detected,Pts_after_NMS,Pts_Stereo_Match,3D_reproj_error_mean,8-point_ransac_it,Pts_Tracking,Pts_quad_match,"
                   "GN_it,GN_mean_it,GN_num_inliers,Pose_entropy,mean_time,\n";
    }

    std::list<int >::iterator lGNit;
    auto lMeanGNit          = gnMeanIterations.begin();     auto lPtsNMS            = ptsNMS.begin();
    auto lPtsDetec          = leftPtsDetec.begin();         auto lPtsStereoMatch    = ptsStereoMatch.begin();
    auto lPtsTracking       = ptsTracking.begin();          auto lPtsQuadMatch      = ptsQuadMatch.begin();
    auto lNumInliersGN      = numInliersGN.begin();         auto lMeanRepErr3d      = rep_err_3d.begin();
    auto lRansacIt8point    = ransacIt_8point.begin();      auto lTime              = frameTimeStamp.begin();
    auto lPosesEntropy      = poses_entropy_.begin();

    int nFrames = 0;
    bool first = true;
    for (lGNit = gnIterations.begin(); lGNit != gnIterations.end(); ++lGNit, ++lMeanGNit, ++lPtsNMS,
            ++lPtsDetec, ++lPtsStereoMatch, ++lPtsTracking, ++lPtsQuadMatch, ++lNumInliersGN, ++lMeanRepErr3d,
            ++lRansacIt8point, ++lTime, ++lPosesEntropy)
    {
        if(withTime){
            if(first){
                f  << setprecision(18) << (*lTime) <<"," << setprecision(6) << 0.0 <<","  << (*lPtsDetec) << ","<< (*lPtsNMS) << "," << (*lPtsStereoMatch) << "," << (*lMeanRepErr3d) << ","
                  << (*lRansacIt8point) << "," << (*lPtsTracking) << "," << (*lPtsQuadMatch) << ","  << (*lGNit) << ","
                  << (*lMeanGNit) << "," << (*lNumInliersGN) << "," << (*lPosesEntropy) << ","<< meanTime << "\n";
            } else{
                f << setprecision(18) << (*lTime) <<","<< setprecision(6) << (*lTime)-initTimestamp <<"," << (*lPtsDetec) << ","<< (*lPtsNMS) << "," << (*lPtsStereoMatch) << "," << (*lMeanRepErr3d) << ","
                  << (*lRansacIt8point) << "," << (*lPtsTracking) << "," << (*lPtsQuadMatch) << ","  << (*lGNit) << ","
                  << (*lMeanGNit) << "," << (*lNumInliersGN) << "," << (*lPosesEntropy) <<"\n";
            }
            first = false;
            nFrames ++;
        }else{
            if(first){
                f << nFrames <<"," << (*lPtsDetec) << ","<< (*lPtsNMS) << "," << (*lPtsStereoMatch) << "," << (*lMeanRepErr3d) << ","
                  << (*lRansacIt8point) << "," << (*lPtsTracking) << "," << (*lPtsQuadMatch) << ","  << (*lGNit) << ","
                  << (*lMeanGNit) << "," << (*lNumInliersGN) << "," << (*lPosesEntropy) << "," << meanTime << "\n";
            } else{
                f << nFrames <<"," << (*lPtsDetec) << ","<< (*lPtsNMS) << "," << (*lPtsStereoMatch) << "," << (*lMeanRepErr3d) << ","
                  << (*lRansacIt8point) << "," << (*lPtsTracking) << "," << (*lPtsQuadMatch) << ","  << (*lGNit) << ","
                  << (*lMeanGNit) << "," << (*lNumInliersGN) << "," << (*lPosesEntropy) <<"\n";
            }
            first = false;
            nFrames ++;
        }
    }

    f.close();
    std::cout << endl << "Statistics saved on "<< filename << std::endl;
#endif
}


void Tracking::writeOnLogFile(const string &name, const string &value) {
#if ENABLE_PRINT
    std::cout << name << value << std::endl;
#endif
//    logFile << name << " " << value << "\n";
}

void Tracking::drawGridAndPoints(const cv::Mat &im, const std::vector<Point2f> &pts, const string &fileName) {

    Mat dIm = im.clone();

    for (int y = 0; y < im.rows; y += frameGridRows)
    {
        for (int x = 0; x < im.cols; x += frameGridCols)
        {
            cv::Rect rect =  cv::Rect(x,y, frameGridCols, frameGridRows);
            cv::rectangle(dIm, rect, cv::Scalar(0, 255, 0));
        }
    }

    drawPointfImage(dIm, pts, fileName);
//    drawPointfImage(im, pts, fileName);
}

void Tracking::logFeatureExtraction(const std::vector<cv::KeyPoint> &kpts_l, const std::vector<cv::KeyPoint> &kpts_r, const std::vector<Point2f> &pts,
                                    const cv::Mat &im) {

    leftPtsDetec.push_back(kpts_l.size());
    writeOnLogFile("Kpts left detected:", std::to_string(kpts_l.size()));
    writeOnLogFile("Kpts rigth detected:", std::to_string(kpts_r.size()));

    cv::Mat imOut;

#if LOG_DRAW
    drawKeypoints(im,kpts_l,imOut, cv::Scalar(0,255,0));
    imwrite("kptsORBoctree.png", imOut);
    drawGridAndPoints(im, pts, "GridNMS.png");
#endif

    writeOnLogFile("Num keypoints after NMS: ", std::to_string(pts.size()));
    ptsNMS.push_back(pts.size());

}

void Tracking::logStereoMatching(const cv::Mat &im_r, const cv::Mat &im_l, const std::vector<cv::DMatch> &mrl,
                                 const std::vector<Point2f> &pts_r, const std::vector<Point2f> &pts_l) {
#if LOG_DRAW
    std::string prefix = "stereo";
    mEightPointLeft->drawMatches_(im_l, im_r, pts_l, pts_r, mrl, false, prefix);
#endif
    writeOnLogFile("Num of stereo matches:", std::to_string(pts_l.size()));
    ptsStereoMatch.push_back(pts_l.size());

}

void Tracking::logLocalMaping(const std::vector<Point3f> &pts3D, double &meanError) {
    writeOnLogFile("Num of 3D points:", std::to_string(pts3D.size()));
    writeOnLogFile("Mean reprojection error:", std::to_string(meanError));
}

void Tracking::logFeatureTracking(const std::vector<Point2f> &pts_l0, const std::vector<Point2f> &pts_r1,
                                  const cv::Mat &fmat, const std::vector<Point2f> &pts_l1, const std::vector<bool> &inliers,
                                  const cv::Mat &im_l0, const cv::Mat &im_l1, const std::vector<cv::DMatch> &mll) {


    writeOnLogFile("Num of left points tracked:", std::to_string(pts_l1.size()));
    writeOnLogFile("Num of right points tracked:", std::to_string(pts_r1.size()));

}

void Tracking::logQuadMatching(const cv::Mat &im_l1, const cv::Mat &im_r1, const std::vector<Point2f> &pts_l1,
                               const std::vector<Point2f> &pts_r1, const std::vector<cv::DMatch> &mlr1, int numPts) {
#if LOG_DRAW
    std::string prefix = "quad";
    mEightPointLeft->drawMatches_(im_l1, im_r1, pts_l1, pts_r1, mlr1, false, prefix);
#endif
    writeOnLogFile("left points before quadMatching:", std::to_string(numPts));
    ptsQuadMatch.push_back(numPts);

}

int Tracking::sign(double value) {

    if (value > 0)
        return 1;
    else
        return -1;
}


void Tracking::drawFarAndClosePts(const cv::Point2f &pt, const cv::Scalar &color, cv::Mat &im) {

    cv::circle(im, pt, 3, color);
}


std::vector<float > Tracking::toQuaternion(const cv::Mat &R) {

    Eigen::Matrix<double,3,3> M;

    M << R.at<float>(0,0), R.at<float>(0,1), R.at<float>(0,2),
         R.at<float>(1,0), R.at<float>(1,1), R.at<float>(1,2),
         R.at<float>(2,0), R.at<float>(2,1), R.at<float>(2,2);

    Eigen::Matrix<double,3,3> eigMat = M;
    Eigen::Quaterniond q(eigMat);

    std::vector<float > v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;

}

cv::Mat Tracking::computeGlobalPose(const cv::Mat &current_pose)
{   
    // Compute global pose
    // Compute the inverse of relative pose estimation inv(current_pose) = [R' | C]
    // where C = -1 * R' * t
    cv::Mat R = current_pose.rowRange(0,3).colRange(0,3);
    cv::Mat t = current_pose.col(3).rowRange(0,3);
    
    cv::Mat Rt  = R.t();
    cv::Mat C   = -1 * Rt * t; 
    
    cv::Mat inv_pose = cv::Mat::eye(4,4,CV_32F);
    Rt.copyTo(inv_pose.rowRange(0,3).colRange(0,3));
    C.copyTo(inv_pose.rowRange(0,3).col(3));

    cameraCurrentPose_ = cameraCurrentPose_ * inv_pose;
    return cameraCurrentPose_.clone();
}
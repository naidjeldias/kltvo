//
// Created by nigel on 21/01/19.
//


#include "tracking.h"



using namespace cv;

Tracking::Tracking(const string &strSettingPath) {
    srand(time(0));

    cv::FileStorage fsSettings(strSettingPath, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fu = fsSettings["Camera.fx"];
    fv = fsSettings["Camera.fy"];
    uc = fsSettings["Camera.cx"];
    vc = fsSettings["Camera.cy"];

    cv::Mat mK = cv::Mat::eye(3,3,CV_64F);
    mK.at<double>(0,0) = fu;
    mK.at<double>(1,1) = fv;
    mK.at<double>(0,2) = uc;
    mK.at<double>(1,2) = vc;
    mK.copyTo(K);

    double bf = fsSettings["Camera.bf"];
    baseline = bf / fu;

    Mat mP1 = cv::Mat::eye(3,4, CV_64F);
    Mat mP2 = cv::Mat::eye(3,4, CV_64F);

    mK.copyTo(mP1.rowRange(0,3).colRange(0,3));
    mK.copyTo(mP2.rowRange(0,3).colRange(0,3));

    mP2.at<double>(0,3) = -bf;

    mP1.copyTo(P1);
    mP2.copyTo(P2);

    nFeatures       = fsSettings["ORBextractor.nFeatures"];
    fScaleFactor  = fsSettings["ORBextractor.scaleFactor"];
    nLevels         = fsSettings["ORBextractor.nLevels"];
    fIniThFAST      = fsSettings["ORBextractor.iniThFAST"];
    fMinThFAST      = fsSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft  = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    mEightPointLeft     = new EightPoint(0.99, 8, 200, 2.0);

    initPhase = true;

    //----local mapping
    max_iter_3d         = 1;       // max iteration for 3D estimation
    th_3d               = 0.5;      // max reprojection error for 3D estimation

    //----Pose estimation
    ransacProb          = 0.99;
    ransacTh            = 7.0;      // th for RANSAC inlier selection using reprojection error
    ransacMinSet        = 3;
    ransacMaxIt         = 50;       // RANSAC iteration for pose estimation
    minIncTh            = 10E-5;    // min increment for pose optimization
    maxIteration        = 20;       // max iteration for minimization into RANSAC routine
    finalMaxIteration   = 100;      // max iterations for minimization final refinement
    reweigh             = true;     // reweight in optimization

    //----Stereo Matching
    minDisp = (0.0F);
    maxDisp = bf/baseline;

//    std::cout << "Max disparity: " << maxDisp << std::endl;

    debug_              = true;
    if(debug_){
        logFile.open("LOG_FILE.txt");
        logFile << std::fixed;
    }

}


Tracking::~Tracking() {
    if(debug_)
        logFile.close();
}

void Tracking::start(const Mat &imLeft, const Mat &imRight, const double timestamp) {


    if (initPhase){
        imLeft0         = imLeft.clone();
        imRight0        = imRight.clone();

        initPhase       = false;
        numFrame        = 0;
        initTimestamp   = timestamp;
    }else{

        if(debug_){
            numFrame ++;
            writeOnLogFile("Frame:", std::to_string(numFrame));
        }

        //detect features
        std::vector<KeyPoint> kpts_l, kpts_r;
        std::vector<Point2f> pts_l0, pts_r0;

        pts_l0.reserve(nFeatures);
        pts_r0.reserve(nFeatures);
        std::thread orbThreadLeft (&Tracking::extractORB, this, 0, std::ref(imLeft0), std::ref (kpts_l), std::ref (pts_l0));
        std::thread orbThreadRight (&Tracking::extractORB, this, 1, std::ref(imRight0), std::ref (kpts_r), std::ref(pts_r0));

        orbThreadLeft.join();
        orbThreadRight.join();

        if(debug_)
            logFeatureExtraction(kpts_l, kpts_r, pts_l0, imLeft0);


        //convert vector of keypoints to vector of Point2f
        for (auto& kpt:kpts_r)
            pts_r0.push_back(kpt.pt);


        //finding matches in left and right previous frames
        std::vector<Point2f> pts_l1, pts_r1, new_pts_l0, new_pts_r0;
        std::vector<DMatch> mlr0, mlr1, mll, mrr;
        std::vector<cv::Point3f> pointCloud;
        stereoMatching(pts_l0, pts_r0, imLeft0, imRight0, mlr0, new_pts_l0, new_pts_r0, pointCloud);
        if(debug_)
            logStereoMatching(imRight0, imLeft0, mlr0, new_pts_r0, new_pts_l0);


        //triangulate previous keypoints
//        std::vector<Point3f> pts3D;
//        double meanError;
//        localMapping(new_pts_l0, new_pts_r0, pts3D, mlr0, meanError);
//        if(debug_)
//            logLocalMaping(pts3D, meanError);


        //tracking features from previous frames to current frames
        std::vector <Mat> left0_pyr, left1_pyr, right0_pyr, right1_pyr;
        Size win (15,15);
        int maxLevel = 3;
        Mat status0, status1, error0, error1;

        std::thread kltThreadLeft (&Tracking::opticalFlowFeatureTrack, this, std::ref(imLeft0), std::ref(imLeft), win, maxLevel,
                        std::ref(status0), std::ref(error0), std::ref(new_pts_l0), std::ref(pts_l1), std::ref(left0_pyr), std::ref(left1_pyr));

        std::thread kltThreadRight (&Tracking::opticalFlowFeatureTrack, this, std::ref(imRight0), std::ref(imRight), win, maxLevel,
                        std::ref(status1), std::ref(error1), std::ref(new_pts_r0), std::ref(pts_r1), std::ref(right0_pyr), std::ref(right1_pyr));

        kltThreadLeft.join();
        kltThreadRight.join();

//        buildOpticalFlowPyramid(imLeft0, left0_pyr, win, maxLevel, true);
//        buildOpticalFlowPyramid(imLeft, left1_pyr, win, maxLevel, true);
//
//        buildOpticalFlowPyramid(imRight0, right0_pyr, win, maxLevel, true);
//        buildOpticalFlowPyramid(imRight,  right1_pyr, win, maxLevel, true);
//
//        calcOpticalFlowPyrLK(left0_pyr, left1_pyr, new_pts_l0, pts_l1, status0, error0, win, maxLevel,
//                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.03), 1);
//
//        calcOpticalFlowPyrLK(right0_pyr, right1_pyr, new_pts_r0, pts_r1, status1, error1, win, maxLevel,
//                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.03), 1);

        std::vector<bool>       inliers;
        Mat fmat;
        (*mEightPointLeft) (new_pts_l0, pts_l1, mll, inliers, true, 0, fmat);

        Mat R_est, t_est;
        essentialMatrixDecomposition(fmat, K, K, new_pts_l0, pts_l1, inliers, R_est, t_est);

        std::vector<double> rvec_est;
        Rodrigues(R_est, rvec_est, noArray());
        if (debug_)
            logFeatureTracking(new_pts_l0, pts_r1, fmat, pts_l1, inliers, imLeft0, imLeft, mll,R_est);


        std::vector<Point3f> new_pts3D;
        std::vector<Point2f> new_pts_l1, new_pts_r1;
        quadMatching(pts3D, pts_l1, pts_r1, inliers, imLeft, imRight, new_pts3D, new_pts_l1, new_pts_r1, mlr1);
        if(debug_)
            logQuadMatching(imLeft, imRight, new_pts_l1, new_pts_r1, mlr1, new_pts3D.size());

        //initialize vector of parameters with rotation and translation from essential matrix
        std::vector<double> p0 (6, 0.0);
        p0.at(0) = rvec_est.at(0);
        p0.at(1) = rvec_est.at(1);
        p0.at(2) = rvec_est.at(2);
        p0.at(3) = t_est.at<double>(0);
        p0.at(4) = t_est.at<double>(1);
        p0.at(5) = t_est.at<double>(2);


        std::vector<bool> inliers2;
        std::vector<Point2d> inPts_l1, inPts_r1;
        std::vector<Point3d> inPts_3D;

        std::vector<double> p (6, 0.0);
        poseEstimationRansac(new_pts_l1, new_pts_r1, new_pts3D, p0, inliers2, p, reweigh);


        //pose refinment with all inliers
        for (int i=0; i<inliers2.size(); i++){
            if(inliers2.at(i)){
                Point2f aux1 = new_pts_l1[i];
                inPts_l1.push_back(aux1);
                Point2f aux2 = new_pts_r1[i];
                inPts_r1.push_back(aux2);
                Point3f aux3 = new_pts3D[i];
                inPts_3D.push_back(aux3);
            }
        }

        // pose refinement with all inliers
        int status = 0;
        for (int i = 0; i < finalMaxIteration; i++){
            status = poseEstimation(inPts_l1, inPts_r1, inPts_3D, p, inPts_l1.size(), reweigh);
            if(status != UPDATE)
                break;
        }

        Mat rot_vec = cv::Mat::zeros(3,1, CV_64F);
        Mat tr_vec  = cv::Mat::zeros(3,1, CV_64F);

        rot_vec.at<double>(0) = p.at(0);
        rot_vec.at<double>(1) = p.at(1);
        rot_vec.at<double>(2) = p.at(2);


        Mat Rotmat;
        Rodrigues(rot_vec, Rotmat, noArray());
//        std::cout << "Rodrigues Rotation mat: \n" << Rotmat << std::endl;

        tr_vec.at<double>(0)  = p.at(3);
        tr_vec.at<double>(1)  = p.at(4);
        tr_vec.at<double>(2)  = p.at(5);

//        Rotmat.convertTo(Rotmat, CV_32F);
//        tr_vec.convertTo(tr_vec, CV_32F);

        Mat Tcw_ = cv::Mat::eye(3,4,CV_64F);

        Rotmat.copyTo(Tcw_.rowRange(0,3).colRange(0,3));
        tr_vec.copyTo(Tcw_.rowRange(0,3).col(3));

        //saving relative pose estimated
        relativeFramePoses.push_back(Tcw_.clone());
        frameTimeStamp.push_back(timestamp);

        Tcw = Tcw_.clone();

        imLeft0     = imLeft.clone();
        imRight0    = imRight.clone();

        if(debug_){
            writeOnLogFile("----------------------------", " ");
        }


    }

}

void Tracking::extractORB(int flag, cv::Mat &im, std::vector<KeyPoint> &kpt, std::vector<cv::Point2f> &pts) {

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

    int nBucketX = im.cols / FRAME_GRID_COLS;
    int nBucketY = im.rows / FRAME_GRID_ROWS;

    //______________Image grid for non-maximum suppression
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

    drawKeypoints( im, kpts, imOut, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imwrite(filename, imOut);
}


void Tracking::opticalFlowFeatureTrack(cv::Mat &imT0, const cv::Mat &imT1, Size win, int maxLevel, cv::Mat &status, cv::Mat &error,
                                       std::vector<Point2f> &prevPts, std::vector<Point2f> &nextPts, std::vector <Mat> imT0_pyr, std::vector <Mat> imT1_pyr) {


//    std::vector <Mat> imT0_pyr, imT1_pyr;
    std::lock_guard<std::mutex> lock1(mtx1);
    buildOpticalFlowPyramid(imT0, imT0_pyr, win, maxLevel, true);
    std::lock_guard<std::mutex> lock2(mtx2);
    buildOpticalFlowPyramid(imT1, imT1_pyr, win, maxLevel, true);
    std::lock_guard<std::mutex> lock3(mtx3);
    calcOpticalFlowPyrLK(imT0_pyr, imT1_pyr, prevPts, nextPts, status, error, win, maxLevel,
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.03), 1);

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


int Tracking::poseEstimationRansac(const std::vector<cv::Point2f> &pts2dl, const std::vector<cv::Point2f> &pts2dr,
                              const std::vector<cv::Point3f> &pts3d, std::vector<double> &p0, std::vector<bool> &bestInliers, std::vector<double> &p, bool reweigh) {


//    std::vector<bool> bestInliers;
    int bestNumInliers = ransacMinSet;

//    for (int i = 0; i < 6; i++)
//        std::cout << p0.at(i) << "\n";

    long double minSumErr = FLT_MAX;

    p = p0;

    for (int n = 0; n < ransacMaxIt; n++){

        //compute rand index
        std::vector<int> randIndex;    //vector of rand index
        randIndex      = generateRandomIndices(pts3d.size(), ransacMinSet);

        std::vector<Point3d> aux_pt3d;
        std::vector<Point2d> aux_pt2dl, aux_pt2dr;

        //selecting the random points
        for(auto &index:randIndex){
            aux_pt3d.push_back(pts3d.at(index));
            aux_pt2dl.push_back(pts2dl.at(index));
            aux_pt2dr.push_back(pts2dr.at(index));
        }

        // initialize p0_ for pose otimization iteration
        std::vector<double> p0_ = p0;

        int status = 0;
        for (int i = 0; i < maxIteration; i++){
            status = poseEstimation(aux_pt2dl, aux_pt2dr, aux_pt3d, p0_, randIndex.size(), reweigh);
            if(status != UPDATE)
                break;
        }

//        std::cout << "Status: " << status << std::endl;
        if(status == FAILED)
            continue;

        std::vector<bool> inliers;

        long double sumErr = 0;
        //validate model againts the init set
        int numInliers = checkInliers(pts3d, pts2dl, pts2dr, randIndex, p0_, inliers, sumErr, true);


        if(numInliers > bestNumInliers){
//            std::cout << "Num inliers: " << numInliers << std::endl;
            bestInliers     = inliers;
            p = p0_;
            bestNumInliers  = numInliers;
            minSumErr = sumErr;
        }
    }

//    std::cout << "Best Num inliers: " << bestNumInliers << std::endl;
//    std::cout << "Num iterations: " << n << std::endl;

}

int Tracking::poseEstimation(const std::vector<cv::Point2d> &pts2dl, const std::vector<cv::Point2d> &pts2dr,
                             const std::vector<cv::Point3d> &pts3d, std::vector<double> &p0, const int numPts, bool reweigh) {

    // 6 parameters rx, ry, rz, tx, ty, tz
    // 2 equation for each point, 2 parameters (u, v) for each 2Dpoint (left and right)
    Mat J   = cv::Mat::zeros(4*numPts, 6, CV_64F);// Jacobian matrix
    //residual matrix,
    Mat res = cv::Mat::zeros(4*numPts, 1, CV_64F);

    computeJacobian(numPts, pts3d, pts2dl, pts2dr, p0, J, res, reweigh);

    cv::Mat A = cv::Mat(6,6,CV_64F);
    cv::Mat B = cv::Mat(6,1,CV_64F);
    cv::Mat S = cv::Mat(6,1,CV_64F);
//    cv::Mat I = cv::Mat::eye(6,6, CV_64F);

    //computing augmented normal equations
    A = J.t() * J;
    B = J.t() * res;

    bool status = cv::solve(A, B, S, DECOMP_NORMAL);

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
                           const std::vector<double> &p0, std::vector<bool> &inliers, long double &sumErr, bool reweigh) {

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

//    std::cout << "Rotation mat: " << std::endl;
//    cout << r00 << " " << r01 << " " << r02 << endl;
//    cout << r10 << " " << r11 << " " << r12 << endl;
//    cout << r20 << " " << r21 << " " << r22 << endl;

    //aux variables
    //3D point computed in previous camera coordinate
    double X1p,Y1p,Z1p;
    //3D point in each next camera coordinate
    double X1c,Y1c,Z1c,X2c, Y2c, Z2c;

    int numInliers = 0;



    for (int i = 0; i < pts3d.size(); i++){


        // weighting
        double weight = 1.0;

        //give more significance to features located closer to the image center in horizontal direction
        //the value 0.05 depends on the stereo camera and lens setup, was empirically set
        if (reweigh)
            weight = 1.0/(fabs(pts2dl.at(i).x - uc)/fabs(uc) + 0.05);

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

        if( rx0*rx0+ry0*ry0+rx1*rx1+ry1*ry1 < ransacTh*ransacTh){
            inliers.push_back(true);
            numInliers++;
        }

    }

    return numInliers;


}

std::vector<int> Tracking::generateRandomIndices(const unsigned long &maxIndice, const int &vecSize){
    std::vector<int> randValues;
    int index;

    do{
        index = rand() % maxIndice;
        if(!(std::find(randValues.begin(), randValues.end(), index) != randValues.end()))
            randValues.push_back(index);
    }while(randValues.size() < vecSize);

//    std::cout << "----------------------------- \n";
//     std::cout << "Rand vector: \n";
//     for(int i = 0; i < randValues.size(); i++)
//         std::cout << randValues.at(i) << std::endl;
//    std::cout << "----------------------------- \n";

    return randValues;
}

double Tracking::euclideanDist(const cv::Point2d &p, const cv::Point2d &q) {
    Point2d diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}


void Tracking::stereoMatching(const std::vector<cv::Point2f> &pts_l, const std::vector<cv::Point2f> &pts_r,
                              const cv::Mat &imLeft, const cv::Mat &imRight, std::vector<cv::DMatch> &matches,
                              std::vector<cv::Point2f> &new_pts_l, std::vector<cv::Point2f> &new_pts_r, std::vector<cv::Point3f> &pointCloud) {

    std::vector<Point2f> aux_pts_r(pts_r);

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
    int index_l = 0;
    for (auto &pt_l:pts_l) {

        Point2f ptr;
        Point3f pt3D;
        int index;
        //find point correspondece in the right image using epipolar constraints
        bool found = findMatchingSAD(pt_l, imLeft, imRight, aux_pts_r, ptr, index, vecRowIndices);
        if(found){
            //check if the point have a good triangulation
            if(triangulation(pt_l, ptr, pt3D)){
                new_pts_l.push_back(pt_l);
                new_pts_r.push_back(ptr);

                pointCloud.push_back(pt3D);

                double dst = euclideanDist(pt_l, ptr);
                DMatch match(pos, pos, dst);
                matches.push_back(match);
                pos++;

            }
        }

        index_l ++;
    }


//    std::cout << "remaining left points: " << new_pts_l.size() << std::endl;
//    std::cout << "remaining right points: "<< new_pts_r.size() << std::endl;
//    std::cout << "Number of matches: "<< matches.size()   << std::endl;

}


bool Tracking::findMatchingSAD(const cv::Point2f &pt_l, const cv::Mat &imLeft, const cv::Mat &imRight,
                               std::vector<cv::Point2f> &pts_r, cv::Point2f &ptr_m, int &index, const std::vector<std::vector<std::size_t>> &vecRowIndices) {

    int halfBlockSize = 2;
    int blockSize = 2 * halfBlockSize + 1;

    int width = imRight.size().width;
    int height = imRight.size().height;

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
            } else {
                template_.at<float>(j, i) = 0;
            }
        }
    }

    const float &vL = pt_l.y;
    const float &uL = pt_l.x;

    const int yi = round(vL);

    const std::vector<std::size_t> &vecCandidates = vecRowIndices[yi];
    if(vecCandidates.empty())
        return false;

    int minSAD = 100000;
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

        //epipolar constraints, the correspondent keypoint must be in the same row and disparity should be positive
        if (deltax >= minDisp && deltax <= maxDisp) {

            //compute SAD
            int sum = 0;
            for (int i = 0; i < blockSize; i++) {
                for (int j = 0; j < blockSize; j++) {
                    int x = (int) pt_r.x - (halfBlockSize - i);
                    int y = (int) pt_r.y - (halfBlockSize - j);
                    //check frame limits
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        Scalar intensity = imRight.at<uchar>(y, x);
                        sum += abs(template_.at<float>(j, i) - intensity[0]);
                    } else {
                        sum += abs(template_.at<float>(j, i) - 0);
                    }
                }
            }

            if (sum < minSAD) {
                noMatching = false;
                minSAD = sum;
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
                            const cv::Mat &imRight, std::vector<cv::Point3f> &new_pts3D,
                            std::vector<cv::Point2f> &new_pts2D_l, std::vector<cv::Point2f> &new_pts2D_r, std::vector<cv::DMatch> &matches) {

    std::vector<Point2f> aux_pts_r(pts2D_r);

    //Assign keypoints on right image to a row table
    std::vector<std::vector<std::size_t>> vecRowIndices (imRight.rows, std::vector<std::size_t>());

    for (int i=0; i<imRight.rows; i++)
        vecRowIndices[i].reserve(pts2D_l.size());

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

        if(inliers.at(i)){

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

//    std::cout << "remaining left points: " << new_pts2D_l.size() << std::endl;
//    std::cout << "remaining right points: "<< new_pts2D_r.size() << std::endl;
//    std::cout << "remaining 3D points: "<< new_pts3D.size()   << std::endl;

}

void Tracking::essentialMatrixDecomposition(const cv::Mat &F, const cv::Mat &K, const cv::Mat &K_l,
                                            const std::vector<cv::Point2f> &pts_l,
                                            const std::vector<cv::Point2f> &pts_r, const std::vector<bool> &inliers, cv::Mat &R_est, cv::Mat &t_est) {


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

    Mat newE = U*Diag*Vt;
    SVD::compute(newE, D, U, Vt, SVD::MODIFY_A| SVD::FULL_UV);

    Mat R1 =  U * W * Vt;
    if(determinant(R1) < 0)
        R1 = -R1;

    Mat R2 = U * W.t() * Vt;
    if(determinant(R2) < 0)
        R2 = -R2;

    Point2f pt_l, pt_r;
    //get first inlier par to check solution
    for(int i = 0; i < inliers.size(); i++){
        if(inliers.at(i)){
            pt_l = pts_l.at(i);
            pt_r = pts_r.at(i);
            break;
        }
    }

    checkSolution(R1,R2, U.col(2), K, K, pt_l, pt_r, R_est, t_est);

}

void Tracking::checkSolution(const cv::Mat &R1, const cv::Mat &R2, const cv::Mat &u3, const cv::Mat &K, const cv::Mat &K_l, const cv::Point2f &pt_l
        , const cv::Point2f &pt_r, cv::Mat &R_est, cv::Mat &t_est) {


    cv::Mat P    = cv::Mat::eye(3,4,CV_64F);
    cv::Mat P_l  = cv::Mat::eye(3,4,CV_64F);


    cv::Mat R;
    cv::Mat u3_;

    // pick a random point to check the solution
//    vector<int> index = generateRandomIndices(pts_r.size() - 1, 1);
//    std::cout << "index: \n" << index.at(0) << std::endl;

    Mat x_r = cv::Mat::zeros(3,1, CV_64F);
    x_r.at<double>(0) = pt_r.x;
    x_r.at<double>(1) = pt_r.y;
    x_r.at<double>(2) = 1.0;

    Mat x_l = cv::Mat::zeros(3,1, CV_64F);
    x_l.at<double>(0) = pt_l.x;
    x_l.at<double>(1) = pt_l.y;
    x_l.at<double>(2) = 1.0;

//    std::cout << "x_r: \n" << x_r << std::endl;

    //point in normalized coordinates
//    Mat xn_r = K_l.inv() * x_r;

//    std::cout << "xn_r: \n" << xn_r << std::endl;

    //compute the 4 possible solutions

    for (int i = 0; i < 4; i++){

        switch (i){

            //------- solution 1
            // [UWVt | +u3]
            case 0:
                R   = R1;
                u3_ = u3.clone();
//                std::cout << "u3: \n" << u3 << std::endl;
                break;
            //------- solution 2
            // [UWVt | -u3]
            case 1:
                R   = R1;
                u3_  = -1 * u3.clone();
//                std::cout << "u3: \n" << u3 << std::endl;
                break;
            //------ solution 3
            // [UWtVt | +u3]
            case 2:
                R = R2;
                u3_ = 1 * u3.clone();
//                std::cout << "u3: \n" << u3 << std::endl;
                break;
            //------ solution 4
            // [UWtVt | -u3]
            case 3:
                R = R2;
                u3_  = -1 * u3.clone();
//                std::cout << "u3: \n" << u3 << std::endl;
                break;
        }

        if(pointFrontCamera(R,u3_,x_l, x_r, P, P_l, K, K_l)){
            R_est = R;
            t_est = u3_;
            break;
        }
    }

}

bool Tracking::pointFrontCamera(cv::Mat &R2, const cv::Mat &t2, const cv::Mat &pt_l, const cv::Mat &pt_r, const cv::Mat &P, cv::Mat &P_l,
                                const cv::Mat &K, const cv::Mat &K_l) {

    R2.copyTo(P_l.rowRange(0,3).colRange(0,3));
    t2.copyTo(P_l.rowRange(0,3).col(3));

//    std::cout << "Determinant: \n" << determinant(R2)  << "\n";
//    if(determinant(R2) == -1)
//        R2 = -1*R2;
//    std::cout << "Determinant: \n" << determinant(R2)  << "\n";

    Mat R1 = cv::Mat::eye(3,3, CV_64F);
    Mat t1 = cv::Mat::zeros(3,1, CV_64F);

//    std::cout << "t2: \n" << t2 << std::endl;

//    std::cout << "P: \n" << P << std::endl;
//    std::cout << "P_l: \n" << P_l << std::endl;

    // Linear Triangulation Method
    cv::Mat A = Mat::zeros(4,4,CV_64F);

    A.row(0) = pt_l.at<double>(0)*P.row(2)   -   P.row(0);
    A.row(1) = pt_l.at<double>(1)*P.row(2)   -   P.row(1);
    A.row(2) = pt_r.at<double>(0)*P_l.row(2) -   P_l.row(0);
    A.row(3) = pt_r.at<double>(1)*P_l.row(2) -   P_l.row(1);

    cv::Mat D,U,Vt;
    cv::SVD::compute(A,D,U,Vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

    Mat pt3D = Vt.row(3).t();

//    std::cout << "pt3D \n" << pt3D << "\n";

    if(pt3D.at<double>(3) == 0)
        std::cerr << "pt3D.at<float>(3) == 0 \n";

    //Euclidean coordinates
    pt3D = pt3D.rowRange(0,3)/pt3D.at<double>(3);

//    std::cout << "pt3D \n" << pt3D << "\n";


    Mat pt3D_t = pt3D.t();

//    std::cout << "R.row(2) \n" << R.row(2) << "\n";
//    std::cout << "pt3D_t \n" << pt3D_t << "\n";
//    std::cout << "u3.at<double>(2) \n" << u3.at<double>(2) << "\n";

//    std::cout << "R.row(2).dot(pt3D_t) \n" << R.row(2).dot(pt3D_t) << "\n";

    double Z1c = R2.row(2).dot(pt3D_t) + t2.at<double>(2);
//    if(Z1c <= 0)
//        std::cout << "Behind camera \n";
//    else
//        std::cout << "Front of camera \n";

    double Z2c = R1.row(2).dot(pt3D_t) + t1.at<double>(2);
//    if(Z2c <= 0)
//        std::cout << "Behind camera \n";
//    else
//        std::cout << "Front of camera \n";

    return Z1c > 0 && Z2c > 0;



}

cv::Mat Tracking::getCurrentPose() {
    return Tcw;
}

void Tracking::saveTrajectoryTUM(const string &filename) {

    ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    cv::Mat Twc = cv::Mat::eye(4,4,CV_32F);

    Mat R0 = Twc.rowRange(0,3).colRange(0,3);
    Mat t0 = Twc.rowRange(0,3).col(3);

//    std::vector<double> q0 =  getQuaternion(R0);

    std::vector<float> q0 =  mRot2Quat(R0);

    f << setprecision(6) << initTimestamp << " " <<  setprecision(9) << t0.at<float>(0) << " " << t0.at<float>(1) << " " << t0.at<float>(2) << " " << q0[3] << " " << q0[2] << " " << q0[1] << " " << q0[0] << endl;
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

        Mat Rw = Twc.rowRange(0,3).colRange(0,3);
        Mat tw = Twc.rowRange(0,3).col(3);

//        std::vector<double> q =  getQuaternion(Rw);

        std::vector<float> q =  mRot2Quat(Rw);

//        std::cout << "Twc: " << Twc << std::endl;

        f << setprecision(6) << initTimestamp << " " <<  setprecision(9) << tw.at<float>(0) << " " << tw.at<float>(1) << " " << tw.at<float>(2) << " " << q[3] << " " << q[2] << " " << q[1] << " " << q[0] << endl;


    }

    f.close();
    std::cout << endl << "trajectory saved!" << std::endl;

}


void Tracking::saveTrajectoryKitti(const string &filename) {

    ofstream f;
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
    std::cout << endl << "trajectory saved!" << std::endl;
}


void Tracking::writeOnLogFile(const string &name, const string &value) {
//    std::cout << name << value << std::endl;
    logFile << name << " " << value << "\n";
}

void Tracking::drawGridAndPoints(const cv::Mat &im, const std::vector<Point2f> &pts, const string &fileName) {

    Mat dIm = im.clone();

    for (int y = 0; y < im.rows; y += FRAME_GRID_ROWS)
    {
        for (int x = 0; x < im.cols; x += FRAME_GRID_COLS)
        {
            cv::Rect rect =  cv::Rect(x,y, FRAME_GRID_COLS, FRAME_GRID_ROWS);
            cv::rectangle(dIm, rect, cv::Scalar(0, 255, 0));
        }
    }

    drawPointfImage(dIm, pts, fileName);

}

void Tracking::logFeatureExtraction(const std::vector<cv::KeyPoint> &kpts_l, const std::vector<cv::KeyPoint> &kpts_r, const std::vector<Point2f> &pts,
                                    const cv::Mat &im) {

    writeOnLogFile("Kpts left detected:", std::to_string(kpts_l.size()));
    writeOnLogFile("Kpts rigth detected:", std::to_string(kpts_r.size()));

    cv::Mat imOut;
    drawKeypoints(im,kpts_l,imOut);
    imwrite("kptsORBoctree.png", imOut);

    drawGridAndPoints(im, pts, "GridNMS.png");
    writeOnLogFile("Num keypoints after NMS: ", std::to_string(pts.size()));


}

void Tracking::logStereoMatching(const cv::Mat &im_r, const cv::Mat &im_l, const std::vector<cv::DMatch> &mrl,
                                 const std::vector<Point2f> &pts_r, const std::vector<Point2f> &pts_l) {

    mEightPointLeft->drawMatches_(im_l, im_r, pts_l, pts_r, mrl, false);
    writeOnLogFile("Number of stereo matches:", std::to_string(pts_l.size()));


}

void Tracking::logLocalMaping(const std::vector<Point3f> &pts3D, double &meanError) {
    writeOnLogFile("Number of 3D points:", std::to_string(pts3D.size()));
    writeOnLogFile("Mean reprojection error:", std::to_string(meanError));
}

void Tracking::logFeatureTracking(const std::vector<Point2f> &pts_l0, const std::vector<Point2f> &pts_r1,
                                  const cv::Mat &fmat, const std::vector<Point2f> &pts_l1, const std::vector<bool> &inliers,
                                  const cv::Mat &im_l0, const cv::Mat &im_l1, const std::vector<cv::DMatch> &mll, const cv::Mat &R) {
    mEightPointLeft->drawEpLines(pts_l0, pts_l1, fmat, inliers, 0, im_l0, im_l1, mll);

    writeOnLogFile("Number of left points tracked:", std::to_string(pts_l1.size()));
    writeOnLogFile("Number of right points tracked:", std::to_string(pts_r1.size()));

    writeOnLogFile("det(F):", std::to_string(determinant(fmat)));
    writeOnLogFile("Number of inliers:", std::to_string(mll.size()));


    writeOnLogFile("det(R) of E:", std::to_string(determinant(R)));
}

void Tracking::logQuadMatching(const cv::Mat &im_l1, const cv::Mat &im_r1, const std::vector<Point2f> &pts_l1,
                               const std::vector<Point2f> &pts_r1, const std::vector<cv::DMatch> &mlr1, int numPts) {

    mEightPointLeft->drawMatches_(im_l1, im_r1, pts_l1, pts_r1, mlr1, false);
    writeOnLogFile("left points before quadMatching:", std::to_string(numPts));

}


bool Tracking::triangulation(const cv::Point2f &kp_l, const cv::Point2f &kp_r, cv::Point3f &pt3d) {


    double w0, w1;
    double dist;

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

    }

    pt3d.x           = (float) point3d.at<double>(0);
    pt3d.y           = (float) point3d.at<double>(1);
    pt3d.z           = (float) point3d.at<double>(2);

    if (dist < 2*th_3d)
        return true;
    else
        return false;

}

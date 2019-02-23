//
// Created by nigel on 21/01/19.
//

#include "tracking.h"
#include "eightpoint.hpp"


using namespace cv;

Tracking::Tracking() {
    srand(time(0));

    initPhase = true;

    //----local mapping
    max_iter_3d         = 10;
    th_3d               = 0.5;

    //----Pose estimation
    ransacProb          = 0.99;
    ransacTh            = 7;
    ransacMinSet        = 3;
    ransacMaxIt         = 100;
    minIncTh            = 10E-6;
    maxIteration    = 20;


    //init pose
    PcwT0 = cv::Mat::eye(4,4,CV_32F);
}

void Tracking::start(const Mat &imLeft, const Mat &imRight) {


    if (initPhase){
        imLeft0     = imLeft;
        imRight0    = imRight;

        initPhase = false;
    }else{
//        std::cout << "Entrou " << std::endl;

        std::vector<KeyPoint> kpts_l, kpts_r;

        Ptr<FeatureDetector> detector = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::FAST_SCORE, 31);

        detector -> detect(imLeft0, kpts_l);
        detector -> detect(imRight0, kpts_r);

        //convert vector of keypoints to vector of Point2f
        std::vector<Point2d> pts_l0, pts_l1, pts_r0, pts_r1, new_pts_l0, new_pts_r0, new_pts_l1, new_pts_r1;
        std::vector<DMatch> mlr0, mlr1, mll;

        //convert vector of keypoints to vector of Point2f
        for (auto& kpt:kpts_l)
            pts_l0.push_back(kpt.pt);

        for (auto& kpt:kpts_r)
            pts_r0.push_back(kpt.pt);

        //finding matches in left and right previous frames
        stereoMatching(pts_l0, pts_r0, imLeft0, imRight0, mlr0, new_pts_l0, new_pts_r0);

        //triangulate previous keypoints
        std::vector<Point3d> pts3D;
        localMapping(new_pts_l0, new_pts_r0, pts3D, mlr0);

        std::vector <Mat> left0_pyr, left1_pyr;

        Size win (21,21);
        int maxLevel = 4;

        //tracking features on left frame to next left frame
        Mat status0, status1, error0, error1;
        buildOpticalFlowPyramid(imLeft0, left0_pyr, win, maxLevel, true);
        buildOpticalFlowPyramid(imLeft, left1_pyr, win, maxLevel, true);

        calcOpticalFlowPyrLK(left0_pyr, left1_pyr, new_pts_l0, pts_l1, status1, error0, win, maxLevel,
                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), 1);

        EightPoint eightPoint;

        std::vector<bool>       inliers;

        eightPoint.setRansacParameters(0.99, 8, 10, 2.0);
        Mat fmat = eightPoint.ransacEightPointAlgorithm(new_pts_l0, pts_l1, mll, inliers, true, 0);



        imLeft0     = imLeft;
        imRight0    = imRight;

    }

}

void Tracking::stereoMatching(const std::vector<cv::Point2d> &pts_l, const std::vector<cv::Point2d> &pts_r,
                              const cv::Mat &imLeft, const cv::Mat &imRight, std::vector<cv::DMatch> &matches,
                              std::vector<cv::Point2d> &new_pts_l, std::vector<cv::Point2d> &new_pts_r) {

    //Define the size of the blocks for block matching.
    int halfBlockSize = 3;
    int blockSize = 2 * halfBlockSize + 1;

    int width = imRight.size().width;
    int height = imRight.size().height;


    int index_l = 0;
    for (auto &pt_l:pts_l) {
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
                    template_.at<double>(j, i) = (int) intensity[0];
                } else {
                    template_.at<double>(j, i) = 0;
                }
            }
        }

        int minSAD = INT_MAX;
        Point2f bestPt;

        //flag to know when the point has no matching
        bool noMatching = true;

        int index_r = 0, bestIndex_r = 0;
        for (auto &pt_r:pts_r) {
            auto deltay = (int) abs(pt_l.y - pt_r.y);
            auto deltax = (int) pt_l.x - (int) pt_r.x;

            //epipolar constraints, the correspondent keypoint must be in the same row and disparity should be positive
            if (deltax > 0 && deltay <= MAX_DELTAY && abs (deltax) <= MAX_DELTAX) {
                noMatching = false;

                //compute SAD
                int sum = 0;
                for (int i = 0; i < blockSize; i++) {
                    for (int j = 0; j < blockSize; j++) {
                        int x = (int) pt_r.x - (halfBlockSize - i);
                        int y = (int) pt_r.y - (halfBlockSize - j);
                        //check frame limits
                        if (x >= 0 && x < width && y >= 0 && y < height) {
                            Scalar intensity = imRight.at<uchar>(y, x);
                            sum += abs(template_.at<double>(j, i) - intensity[0]);
                        } else {
                            sum += abs(template_.at<double>(j, i) - 0);
                        }
                    }
                }

                if (sum < minSAD) {
                    minSAD = sum;
                    bestPt = pt_r;
                    bestIndex_r = index_r;
                }
            }

            index_r ++;
        }

        if (!noMatching) {
            new_pts_l.push_back(pt_l);
            new_pts_r.push_back(bestPt);

            float dst = (float) norm(Mat(pt_l), Mat(bestPt));
            DMatch match(index_l, bestIndex_r, dst);
            matches.push_back(match);

        }

        index_l ++;
    }


//    std::cout << "Size left points: " << new_pts_l.size() << std::endl;
//    std::cout << "Size right points: "<< new_pts_r.size() << std::endl;
//    std::cout << "Number of matches: "<< matches.size()   << std::endl;
}

void Tracking::localMapping(const std::vector<cv::Point2d> &pts_l, const std::vector<cv::Point2d> &pts_r,
                            std::vector<cv::Point3d> &pts3D, const std::vector<DMatch> &matches) {

    Point2f kp_l, kp_r;
    // std::vector<Landmark> points3d;
    double w0, w1;

    for( int i = 0; i < pts_l.size() ; i++ ){

        kp_l = pts_l.at(i);
        kp_r = pts_r.at(i);

//        std::cout << "kp_l: " << kp_l << std::endl;
//        std::cout << "kp_r: " << kp_r << std::endl;

        w0 = w1 = 1.0;

        Mat point3d;

        for(int j = 0; j < max_iter_3d; j++){

            Mat A   = Mat::zeros(4,4,CV_64F);
            Mat D, U, Vt;


            // std::cout << "w0: " << w0 << std::endl;
            // std::cout << "w1: " << w1 << std::endl;

            A.row(0) = w0*(kp_l.x*P1.row(2)-P1.row(0));
            A.row(1) = w0*(kp_l.y*P1.row(2)-P1.row(1));
            A.row(2) = w1*(kp_r.x*P2.row(2)-P2.row(0));
            A.row(3) = w1*(kp_r.y*P2.row(2)-P2.row(1));

//             std::cout << "Mat A: " << A << std::endl;

            SVD::compute(A,D,U,Vt);

//             std::cout << "Vt: " << Vt << std::endl;

            point3d = Vt.row(3).t();

//             std::cout << "point 3D: " << point3d << std::endl;

            point3d = point3d.rowRange(0,4)/point3d.at<double>(3);

//             std::cout << "point 3D cc: " << point3d << std::endl;

            Mat p0 = P1*point3d;
            Mat p1 = P2*point3d;

//             std::cout << "p0 : " << p0 << std::endl;
//             std::cout << "p1 : " << p1 << std::endl;


            // std::cout << "p_0: [ " << kp_l.pt.x << "," << kp_l.pt.y << "]" << std::endl;
            // std::cout << "p_1: [ " << kp_r.pt.x << "," << kp_r.pt.y << "]" << std::endl;
            // std::cout << "p0: " << p0.at<double>(2) << std::endl;

            w0 = 1.0/p0.at<double>(2);
            w1 = 1.0/p1.at<double>(2);

            double dx0 = kp_l.x - p0.at<double>(0)/p0.at<double>(2);
            double dy0 = kp_l.y - p0.at<double>(1)/p0.at<double>(2);
            double dx1 = kp_r.x - p1.at<double>(0)/p1.at<double>(2);
            double dy1 = kp_r.y - p1.at<double>(1)/p1.at<double>(2);

            double dist = sqrt(dx0*dx0+dy0*dy0) + sqrt(dx1*dx1+dy1*dy1);

//            std::cout << "dist: " << dist << std::endl;
//            std::cout << "it: " << j << std::endl;

            if(dist < 2*th_3d){
                break;
            }
        }
        // p0 = p0.rowRange(0,3)/p0.at<double>(2);

        // std::cout << "p0 : " << p0 << std::endl;

        // std::cout << "p_0: [ " << kp_l.pt.x << "," << kp_l.pt.y << "]" << std::endl;


        // std::cout << "dist2: " << dist << std::endl;
        Point3d pt3d;

        pt3d.z           = point3d.at<double>(0);
//        std::cout << "pt3d z: " << pt3d.z << std::endl;
        pt3d.y           = point3d.at<double>(1);
        pt3d.x           = point3d.at<double>(2);

        pts3D.push_back(pt3d);
//        std::cout << "Num points 3D: " << pts3D.size() << std::endl;
    }

    //  std::cout << "Num points 3D: " << points3d.size() << std::endl;

}

void Tracking::bucketFeatureExtraction(Mat &image, Size block, std::vector<KeyPoint> &kpts) {

    //Ptr<FeatureDetector> detector = ORB::create();
    Ptr<FeatureDetector> detector = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31);

    int maxFeature =50;

    int W = image.size().width;
    int H = image.size().height;

    //std::cout << "Image width : " << W << std::endl;
    //std::cout << "Image height : " << H << std::endl;

    int w = block.width;
    int h = block.height;

    //std::cout << "Patch width : " << w << std::endl;
    //std::cout << "Patch height : " << h << std::endl;

    int cont = 0;

    for (int y = 0; y <= H - h; y += h){
        for(int x = 0; x <= W - w; x += w){
            //cont ++;
            //std::cout << "Patch: " << cont << std::endl;
            Mat imPatch = image(Rect (x, y, w, h)).clone();

            std::vector<KeyPoint> aux;
            detector -> detect(imPatch, aux);

            //std::cout << "Num features: " << aux.size() << std::endl;

            //sort keypoints by response
            std::sort(aux.begin(), aux.end(), []( const KeyPoint &p1, const KeyPoint &p2){
                return p1.response > p2.response;
            });

            if(aux.size() >= maxFeature ){
                for (int i = 0; i < maxFeature; i++){
                    KeyPoint kpt_ = aux.at(i);

                    //std::cout << "Harris score : " << kpt_.response <<std::endl;
                    KeyPoint kpt;
                    kpt.pt.x = kpt_.pt.x + x;
                    kpt.pt.y = kpt_.pt.y + y;

                    kpts.push_back(kpt);
                }
            }else if (aux.size() > 0 && aux.size() < maxFeature ){
                for (int i = 0; i < aux.size(); i++){
                    KeyPoint kpt_ = aux.at(i);

                    //std::cout << "Harris score : " << kpt_.response <<std::endl;
                    KeyPoint kpt;
                    kpt.pt.x = kpt_.pt.x + x;
                    kpt.pt.y = kpt_.pt.y + y;

                    kpts.push_back(kpt);
                }
            }
            //rectangle(image,Rect (x, y, w, h),(0,0,255),1);
        }
    }
    //Mat rgb;
    //cvtColor(image, rgb, COLOR_GRAY2RGB);

    //drawKeypoints(rgb, kpts, rgb, (0,0,255), 1);

    //std::cout << "Num features: " << kpts.size() << std::endl;

    //imshow("Image", rgb); // visualization
    //waitKey(25); // visualization

}

int Tracking::poseEstimationRansac(const std::vector<cv::Point2d> &pts2dl, const std::vector<cv::Point2d> &pts2dr,
                              const std::vector<cv::Point3d> &pts3d, std::vector<double> p0) {

    int n = 0;
    long int r          = 1000;//adjusted dinamically

    std::vector<bool> bestInliers;
    int bestNumInliers = ransacMinSet;

    while (n < r && n < ransacMaxIt){

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

        poseEstimation(aux_pt2dl, aux_pt2dr, aux_pt3d, p0, randIndex.size());

        std::vector<bool> inliers;

        //validate model againts the init set
        int numInliers = checkInliers(pts3d, pts2dl, pts2dr, randIndex, p0, inliers);


        if(numInliers > bestNumInliers){
//            std::cout << "Num inliers: " << numInliers << std::endl;
            bestInliers     = inliers;

            bestNumInliers  = numInliers;

            //fraction of inliers in the set of points
            double w    = (double) bestNumInliers / (double) pts3d.size();
//            std::cout << "fraction of inliers: " << w << std::endl;
            //probability of not all N points are inliers
            //in each iteration we pick N points that are all inliers with probability w^N
            double p1   = 1 - pow(w, ransacMinSet);
            p1 = MAX(LDBL_MIN, p1);     // Avoid division by -Inf
            p1 = MIN(1-LDBL_MIN, p1);   // Avoid division by 0.
//            std::cout << "probability : " << p1 << std::endl;
            //probability of not all N points are inliers in r iterations is (1 - w^N)^r
            //the probability that in r iteration, at least once, all N points are inliers: p = 1-(1 - W^N)^r
            r = log(1 - ransacProb)/log(p1);
//            std::cout << "r estimated: " << r << std::endl;
        }
        n++;
    }





}

int Tracking::poseEstimation(const std::vector<cv::Point2d> &pts2dl, const std::vector<cv::Point2d> &pts2dr,
                             const std::vector<cv::Point3d> &pts3d, std::vector<double> &p0, const int numPts) {

    bool converged;

    for (int i = 0; i < maxIteration; i++){

        // 6 parameters rx, ry, rz, tx, ty, tz
        // 2 equation for each point, 2 parameters (u, v) for each 2Dpoint (left and right)
        Mat J   = cv::Mat::zeros(4*numPts, 6, CV_64F);// Jacobian matrix
        //residual matrix,
        Mat res = cv::Mat::zeros(4*numPts, 1, CV_64F);

        computeJacobian(numPts, pts3d, pts2dl, pts2dr, p0, J, res, true);

        cv::Mat A = cv::Mat(6,6,CV_64F);
        cv::Mat B = cv::Mat(6,1,CV_64F);
        cv::Mat S = cv::Mat(6,1,CV_64F);
        cv::Mat I = cv::Mat::eye(6,6, CV_64F);

        //computing augmented normal equations
        A = J.t() * J;
        B = -J.t() * res;

        bool status = cv::solve(A, B, S, DECOMP_NORMAL);

        converged = false;

        if(status){
            converged = true;
            //compute increments
            for(int j = 0; j < 6; j++){
                p0.(j) += S.at<double>(j);
                if(fabs(S.at<double>(j)) > minIncTh)
                    converged = false;
            }
            if(converged)
                break;
        }
    }

    return converged;


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
            J.at<double>(i,j)   = weight*fu*(X1cd*Z1c-X1c*Z1cd)/(Z1c*Z1c); // left u'
            J.at<double>(i+1,j) = weight*fu*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c); // left v'

            J.at<double>(i+2,j) = weight*fv*(X1cd*Z2c-X2c*Z1cd)/(Z2c*Z2c); // right u'
            J.at<double>(i+3,j) = weight*fv*(Y1cd*Z2c-Y2c*Z1cd)/(Z2c*Z2c); // right v'

        }

        // set prediction (project via K)
        double pred_u1 = fu*X1c/Z1c+uc; //  left u;
        double pred_v1 = fv*Y1c/Z1c+vc; //  left v

        double pred_u2 = fu*X2c/Z2c+uc; // right u
        double pred_v2 = fv*Y2c/Z2c+vc; // right v

        // set residuals
        res.at<double>(i)   = weight*(pts2d_l.at(i).x - pred_u1);
        res.at<double>(i+1) = weight*(pts2d_l.at(i).y - pred_v1);

        res.at<double>(i+2) = weight*(pts2d_r.at(i).x - pred_u2);
        res.at<double>(i+3) = weight*(pts2d_r.at(i).y - pred_v2);

    }

}


int Tracking::checkInliers(const std::vector<cv::Point3d> &pts3d, const std::vector<cv::Point2d> &pts2dl,
                           const std::vector<cv::Point2d> &pts2dr, const std::vector<int> &index,
                           const std::vector<double> &p0, std::vector<bool> &inliers) {

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

    for (int i = 0; i < pts3d.size(); i++){

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
        double rx0   = (pts2dl.at(i).x - pred_u1);
        double ry0 = (pts2dl.at(i).y - pred_v1);

        double rx1 = (pts2dr.at(i).x - pred_u2);
        double ry1 = (pts2dr.at(i).y - pred_v2);

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
        randValues.push_back(index);
    }while(randValues.size() < vecSize);

//     std::cout << "Rand vector: ";
//     for(int i = 0; i < randValues.size(); i++)
//         std::cout << randValues.at(i) << std::endl;

    return randValues;
}
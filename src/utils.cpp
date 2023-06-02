//
// Created by nigel on 21/01/19.
//

#include "utils.h"

namespace kltvo
{
namespace utils
{
void load_camCalib_yaml(string path, cv::Mat &K, double &baseline) {

    cv::FileStorage fsSettings(path, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["Camera.fx"] >> K.at<double>(0,0);
    fsSettings["Camera.fy"] >> K.at<double>(1,1);
    fsSettings["Camera.cx"] >> K.at<double>(0,2);
    fsSettings["Camera.cy"] >> K.at<double>(1,2);
    fsSettings["Camera.bf"] >> baseline;

    baseline = baseline / K.at<double>(0,0);

    //std::cout << K << std::endl;
//    std::cout << baseline << std::endl;

}

inline float SIGN(float x) {
    return (x >= 0.0f) ? +1.0f : -1.0f;
}

inline float NORM(float a, float b, float c, float d) {
    return sqrt(a * a + b * b + c * c + d * d);
}

// quaternion = [w, x, y, z]'
std::vector<float> mRot2Quat(const cv::Mat& m) {

    std::vector<float> Q;
    Q.reserve(4);

    float r11 = m.at<float>(0, 0);
    float r12 = m.at<float>(0, 1);
    float r13 = m.at<float>(0, 2);
    float r21 = m.at<float>(1, 0);
    float r22 = m.at<float>(1, 1);
    float r23 = m.at<float>(1, 2);
    float r31 = m.at<float>(2, 0);
    float r32 = m.at<float>(2, 1);
    float r33 = m.at<float>(2, 2);
    float q0 = (r11 + r22 + r33 + 1.0f) / 4.0f;
    float q1 = (r11 - r22 - r33 + 1.0f) / 4.0f;
    float q2 = (-r11 + r22 - r33 + 1.0f) / 4.0f;
    float q3 = (-r11 - r22 + r33 + 1.0f) / 4.0f;
    if (q0 < 0.0f) {
        q0 = 0.0f;
    }
    if (q1 < 0.0f) {
        q1 = 0.0f;
    }
    if (q2 < 0.0f) {
        q2 = 0.0f;
    }
    if (q3 < 0.0f) {
        q3 = 0.0f;
    }
    q0 = sqrt(q0);
    q1 = sqrt(q1);
    q2 = sqrt(q2);
    q3 = sqrt(q3);
    if (q0 >= q1 && q0 >= q2 && q0 >= q3) {
        q0 *= +1.0f;
        q1 *= SIGN(r32 - r23);
        q2 *= SIGN(r13 - r31);
        q3 *= SIGN(r21 - r12);
    }
    else if (q1 >= q0 && q1 >= q2 && q1 >= q3) {
        q0 *= SIGN(r32 - r23);
        q1 *= +1.0f;
        q2 *= SIGN(r21 + r12);
        q3 *= SIGN(r13 + r31);
    }
    else if (q2 >= q0 && q2 >= q1 && q2 >= q3) {
        q0 *= SIGN(r13 - r31);
        q1 *= SIGN(r21 + r12);
        q2 *= +1.0f;
        q3 *= SIGN(r32 + r23);
    }
    else if (q3 >= q0 && q3 >= q1 && q3 >= q2) {
        q0 *= SIGN(r21 - r12);
        q1 *= SIGN(r31 + r13);
        q2 *= SIGN(r32 + r23);
        q3 *= +1.0f;
    }
    else {
        printf("coding error\n");
    }
    float r = NORM(q0, q1, q2, q3);
    q0 /= r;
    q1 /= r;
    q2 /= r;
    q3 /= r;

    Q.push_back(q0); Q.push_back(q1); Q.push_back(q2); Q.push_back(q3);

//    cv::Mat res = (std::Mat_<float>(4, 1) << q0, q1, q2, q3);
    return Q;
}


std::vector<double> getQuaternion(cv::Mat &R){
    double trace = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
    std::vector<double> Q;
    if (trace > 0.0)
    {
        double s = sqrt(trace + 1.0);
        Q[3] = (s * 0.5);
        s = 0.5 / s;
        Q[0] = ((R.at<double>(2,1) - R.at<double>(1,2)) * s);
        Q[1] = ((R.at<double>(0,2) - R.at<double>(2,0)) * s);
        Q[2] = ((R.at<double>(1,0) - R.at<double>(0,1)) * s);
    }

    else
    {
        int i = R.at<double>(0,0) < R.at<double>(1,1) ? (R.at<double>(1,1) < R.at<double>(2,2) ? 2 : 1) : (R.at<double>(0,0) < R.at<double>(2,2) ? 2 : 0);
        int j = (i + 1) % 3;
        int k = (i + 2) % 3;

        double s = sqrt(R.at<double>(i, i) - R.at<double>(j,j) - R.at<double>(k,k) + 1.0);
        Q[i] = s * 0.5;
        s = 0.5 / s;

        Q[3] = (R.at<double>(k,j) - R.at<double>(j,k)) * s;
        Q[j] = (R.at<double>(j,i) + R.at<double>(i,j)) * s;
        Q[k] = (R.at<double>(k,i) + R.at<double>(i,k)) * s;
    }
    return Q;
}

void saveTrajectoryKitti(const string &filename, std::list<cv::Mat> &relativeFramePoses) {

    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    cv::Mat Twc = cv::Mat::eye(4,4,CV_32F);

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

        cv::Mat rot_mat = cv::Mat::zeros(3,1, CV_64F);
        cv::Mat tr_vec  = cv::Mat::zeros(3,1, CV_64F);

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

void saveTrajectoryEuroc(const string &filename, std::list<cv::Mat> &relativeFramePoses, std::vector<double>  frameTimeStamps) {

    ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    cv::Mat Twc = cv::Mat::eye(4,4,CV_32F);

    cv::Mat R0 = Twc.rowRange(0,3).colRange(0,3);
    cv::Mat t0 = Twc.rowRange(0,3).col(3);

//    std::vector<float> q0 =  toQuaternion(R0);
    std::vector<float> q0 =  utils::mRot2Quat(R0);

    /*
        * The global pose is computed in reference to the first frame by concatanation
        * The current global pose is computed by
        * so Twc * inv(Tcw) where Tcw is current relative pose estimated and Twc is the last global pose
        * Initial Pwc = [I | 0]
    */
    std::list<cv::Mat>::iterator lit;
    int cont=0;
    for(lit = relativeFramePoses.begin(); lit != relativeFramePoses.end(); ++lit, ++cont){

        double timestamp = frameTimeStamps[cont];
        //Compute the inverse of relative pose estimation inv(Tcw) = [R' | C]
        //where C = -1 * R' * t

        cv::Mat rot_mat = cv::Mat::zeros(3,1, CV_64F);
        cv::Mat tr_vec  = cv::Mat::zeros(3,1, CV_64F);

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

        cv::Mat Rw = Twc.rowRange(0,3).colRange(0,3);
        cv::Mat tw = Twc.rowRange(0,3).col(3);

//        std::vector<float> q =  toQuaternion(Rw);
//        std::vector<float> q =  mRot2Quat(Rw);

//        std::cout << "Rotation Matrix: " << Rw << std::endl;
        Eigen::Matrix<double,3,3> M;

        M <<    Rw.at<float>(0,0), Rw.at<float>(0,1), Rw.at<float>(0,2),
                Rw.at<float>(1,0), Rw.at<float>(1,1), Rw.at<float>(1,2),
                Rw.at<float>(2,0), Rw.at<float>(2,1), Rw.at<float>(2,2);

        Eigen::Quaterniond q(M);

    //    std::cout << "quaternion: " <<  " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

//        f << setprecision(6) << (*lTime) << " " <<  setprecision(9) << tw.at<float>(0) << " " << tw.at<float>(1) << " "
//                << tw.at<float>(2) << " " << q[3] << " " << q[2] << " " << q[1] << " " << q[0] << endl;
        f << setprecision(6) << timestamp << " " <<  setprecision(9) << tw.at<float>(0) << " " << tw.at<float>(1) << " "
          << tw.at<float>(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

    }

    f.close();
    std::cout << endl << "trajectory saved on "<< filename << std::endl;

}

void drawPointfImage(const cv::Mat &im, const std::vector<cv::Point2f> pts, const string &filename) {
    std::vector<cv::KeyPoint> kpts;
    cv::Mat imOut;
    for (unsigned int i = 0; i < pts.size(); i++){
        cv::KeyPoint kpt;
        kpt.pt = pts.at(i);

        kpts.push_back(kpt);
    }

    drawKeypoints( im, kpts, imOut, cv::Scalar(0,225,0), cv::DrawMatchesFlags::DEFAULT );
    imwrite(filename, imOut);
}

void writeOnLogFile(const string &name, const string &value) {
#if ENABLE_PRINT
    std::cout << name << value << std::endl;
#endif
//    logFile << name << " " << value << "\n";
}

void drawGridAndPoints(const cv::Mat &im, const std::vector<cv::Point2f> &pts, const string &fileName, 
                    int frameGridRows, int frameGridCols) {

    cv::Mat dIm = im.clone();

    for (int y = 0; y < im.rows; y += frameGridRows)
    {
        for (int x = 0; x < im.cols; x += frameGridCols)
        {
            cv::Rect rect =  cv::Rect(x,y, frameGridCols, frameGridRows);
            cv::rectangle(dIm, rect, cv::Scalar(0, 255, 0));
        }
    }

    utils::drawPointfImage(dIm, pts, fileName);
}

void logFeatureExtraction(const std::vector<cv::KeyPoint> &kpts_l, const std::vector<cv::KeyPoint> &kpts_r, 
                        const std::vector<cv::Point2f> &pts, const cv::Mat &im, int frameGridRows, int frameGridCols) {
#if LOG
    utils::writeOnLogFile("Kpts left detected:", std::to_string(kpts_l.size()));
    utils::writeOnLogFile("Kpts rigth detected:", std::to_string(kpts_r.size()));
    utils::writeOnLogFile("Num keypoints after NMS: ", std::to_string(pts.size()));
#endif

#if LOG_DRAW
    cv::Mat imOut;
    drawKeypoints(im,kpts_l,imOut, cv::Scalar(0,255,0));
    imwrite("kptsORBoctree.png", imOut);
    utils::drawGridAndPoints(im, pts, "GridNMS.png", frameGridRows, frameGridCols);
#endif

}

void logStereoMatching(const cv::Mat &im_r, const cv::Mat &im_l, const std::vector<cv::DMatch> &mrl,
                                 const std::vector<cv::Point2f> &pts_r, const std::vector<cv::Point2f> &pts_l) {
#if LOG_DRAW
    std::string prefix = "stereo";
    drawMatches(im_l, im_r, pts_l, pts_r, mrl, false, prefix);
#endif

#if LOG
    utils::writeOnLogFile("Num of stereo matches:", std::to_string(pts_l.size()));
#endif
}

void drawMatches(const cv::Mat &left_image, const cv::Mat &right_image,
                                const std::vector<cv::Point2f> &kpts_l, const std::vector<cv::Point2f> &kpts_r,
                                const std::vector<cv::DMatch> &matches, bool hold, const std::string &prefix) {

    cv::Mat imageMatches, imageKptsLeft, imageKptsRight;
    //convert vector of Point2f to vector of Keypoint
    std::vector<cv::KeyPoint> prevPoints, nextPoints;
    for (unsigned i = 0; i < kpts_l.size(); i++){
        cv::KeyPoint kpt_l, kpt_r;
        kpt_l.pt = kpts_l.at(i);
        kpt_r.pt = kpts_r.at(i);
        prevPoints.push_back(kpt_l);
        nextPoints.push_back(kpt_r);
    }

    drawMatches(left_image, prevPoints, right_image, nextPoints, matches, imageMatches, cv::Scalar::all(-1), 
            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    drawKeypoints( left_image, prevPoints, imageKptsLeft, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    drawKeypoints( right_image, nextPoints, imageKptsRight, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );


    std::string kptsLeft     = std::string(prefix+"KptsLeft.png");
    std::string kptsRight    = std::string(prefix+"kptsRight.png");
    std::string matchStr     = std::string(prefix+"matches.png");

    imwrite(kptsLeft, imageKptsLeft);
    imwrite(kptsRight, imageKptsRight);
    imwrite(matchStr, imageMatches);

//    if(hold)
//        waitKey(0);

}

void logLocalMaping(const std::vector<cv::Point3f> &pts3D, double &meanError) {
    utils::writeOnLogFile("Num of 3D points:", std::to_string(pts3D.size()));
    utils::writeOnLogFile("Mean reprojection error:", std::to_string(meanError));
}

void logFeatureTracking(const std::vector<cv::Point2f> &pts_l0, const std::vector<cv::Point2f> &pts_r1,
                                  const cv::Mat &fmat, const std::vector<cv::Point2f> &pts_l1, const std::vector<bool> &inliers,
                                  const cv::Mat &im_l0, const cv::Mat &im_l1, const std::vector<cv::DMatch> &mll) {

    utils::writeOnLogFile("Num of left points tracked:", std::to_string(pts_l1.size()));
    utils::writeOnLogFile("Num of right points tracked:", std::to_string(pts_r1.size()));

}

void logQuadMatching(const cv::Mat &im_l1, const cv::Mat &im_r1, const std::vector<cv::Point2f> &pts_l1,
                               const std::vector<cv::Point2f> &pts_r1, const std::vector<cv::DMatch> &mlr1, int numPts) {
#if LOG_DRAW
    std::string prefix = "quad";
    utils::drawMatches(im_l1, im_r1, pts_l1, pts_r1, mlr1, false, prefix);
#endif

#if LOG
    utils::writeOnLogFile("left points before quadMatching:", std::to_string(numPts));
#endif
}

void drawFarAndClosePts (const std::vector<cv::Point2f> &pts, const cv::Mat &im, const std::vector<bool> &isClose)
{
    cv::Mat imOut = im.clone();
    for (int i = 0 ; i < isClose.size() ; i++){
        if(isClose[i])
            cv::circle(imOut, pts[i], 3, cv::Scalar(0, 255, 0));
        else
            cv::circle(imOut, pts[i], 3, cv::Scalar(0, 0, 255));
    }

    imwrite("dstPts.png", imOut);
}

void drawEpLines(const std::vector<cv::Point2f> &pts_l, const std::vector<cv::Point2f> &pts_r, const cv::Mat &F,
            const std::vector<bool> &inliers, int rightFlag, const cv::Mat &image, const cv::Mat &image1,
            const std::vector<cv::DMatch> &matches){

    cv::Mat border  = cv::Mat::zeros(4,2,CV_64F);
    cv::Mat X_l     = cv::Mat::zeros(3,1,CV_64F);
    cv::Mat X_r     = cv::Mat::zeros(3,1,CV_64F);
    cv::Mat eplines;

    std::vector<cv::Point2f> ptsl_, ptsr_;
    std::vector<cv::DMatch> matches_;

    int w = image.size().width;
    int h = image.size().height;

//    Mat rgb = image.clone();
    cv::Mat rgb, rgb1;
    cvtColor(image, rgb, cv::COLOR_GRAY2BGR);
    cvtColor(image1, rgb1, cv::COLOR_GRAY2BGR);
    int count = 0;
    // std::vector<Point2f> points;
    for(unsigned i = 0; i < inliers.size(); i++ ){
        if(inliers.at(i)){
            count ++;
            //point on left frame
            X_l.at<double>(0)     = pts_l.at(i).x;
            X_l.at<double>(1)     = pts_l.at(i).y;
            X_l.at<double>(2)     = 1.0;
            //point on right frame
            X_r.at<double>(0)   = pts_r.at(i).x;
            X_r.at<double>(1)   = pts_r.at(i).y;
            X_r.at<double>(2)   = 1.0;

            ptsl_.push_back(pts_l.at(i));
            ptsr_.push_back(pts_r.at(i));
//            matches_.push_back(matches.at(i));

            cv::Mat ep_line, ep_line1;

            //if zero draw in left image else draw in right image
//            if(rightFlag == 0){
//                ep_line = F.t() * X_r;
//            }else
//                ep_line = F * X_l;

            ep_line  = F.t() * X_r;

            ep_line1 = F * X_l;

            std::vector<double> linePts, linePts1;

            //computing ep lines Left
            double a    =   ep_line.at<double>(0);
            double b    =   ep_line.at<double>(1);
            double c    =   ep_line.at<double>(2);

            //borders and epipolar line intersection points
            border.at<double>(0,0) = 0.0;           border.at<double>(0,1) = -c/b;          //left
            border.at<double>(1,0) = w;             border.at<double>(1,1) = (-c-a*w)/b;    //right
            border.at<double>(2,0) = -c/a;          border.at<double>(2,1) = 0.0;           //up
            border.at<double>(3,0) = (-c-b*h)/a;    border.at<double>(3,1) = h;             //down
            //points of epipolar lines


            for(int i = 0; i < 4; i++){
                double x = border.at<double>(i,0);
                double y = border.at<double>(i,1);
                if( x>=0 && x<=w && y>=0 && y<=h){
                    linePts.push_back(x);
                    linePts.push_back(y);
                }
            }

            //computing ep lines Right
            a    =   ep_line1.at<double>(0);
            b    =   ep_line1.at<double>(1);
            c    =   ep_line1.at<double>(2);

            //borders and epipolar line intersection points
            border.at<double>(0,0) = 0.0;           border.at<double>(0,1) = -c/b;          //left
            border.at<double>(1,0) = w;             border.at<double>(1,1) = (-c-a*w)/b;    //right
            border.at<double>(2,0) = -c/a;          border.at<double>(2,1) = 0.0;           //up
            border.at<double>(3,0) = (-c-b*h)/a;    border.at<double>(3,1) = h;             //down
            //points of epipolar lines


            for(int i = 0; i < 4; i++){
                double x = border.at<double>(i,0);
                double y = border.at<double>(i,1);
                if( x>=0 && x<=w && y>=0 && y<=h){
                    linePts1.push_back(x);
                    linePts1.push_back(y);
                }
            }

            cv::Scalar color (rand() % 255,rand() % 255,rand() % 255);

            if(linePts.size()>=4){
//                cv::Scalar color (rand() % 255,rand() % 255,rand() % 255);
                cv::Point2d x0(linePts.at(0), linePts.at(1));
                cv::Point2d x1(linePts.at(2), linePts.at(3));
                line(rgb, x0, x1, color, 1);

                cv::Point2d x(X_l.at<double>(0), X_l.at<double>(1));
                circle(rgb,x, 5, color, -1);

            }

            if(linePts1.size()>=4){
//                cv::Scalar color (rand() % 255,rand() % 255,rand() % 255);
                cv::Point2d x0(linePts1.at(0), linePts1.at(1));
                cv::Point2d x1(linePts1.at(2), linePts1.at(3));
                line(rgb1, x0, x1, color, 1);

                cv::Point2d x(X_r.at<double>(0), X_r.at<double>(1));
                circle(rgb1,x, 5, color, -1);

            }
        }
    }

    std::string prefix = "track";
    utils::drawMatches(image, image1, ptsl_, ptsr_, matches, false, prefix);

    imwrite("lefteplines.png",rgb);
    imwrite("righteplines.png",rgb1);

}

} // namespace utils
} // namespace kltvo
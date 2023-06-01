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

void saveTrajectoryEuroc(const string &filename, std::list<cv::Mat> &relativeFramePoses, std::list<double>  frameTimeStamps) {

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
    std::list<double>::iterator lTime = frameTimeStamps.begin();
    for(lit = relativeFramePoses.begin(); lit != relativeFramePoses.end(); ++lit, ++lTime){


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
        f << setprecision(6) << (*lTime) << " " <<  setprecision(9) << tw.at<float>(0) << " " << tw.at<float>(1) << " "
          << tw.at<float>(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

    }

    f.close();
    std::cout << endl << "trajectory saved on "<< filename << std::endl;

}

} // namespace utils
} // namespace kltvo
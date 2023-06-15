//
// Created by nigel on 21/01/19.
//

#ifndef KLTVO_UTILS_H
#define KLTVO_UTILS_H

#define LOG             true
#define ENABLE_PRINT    false
#define LOG_DRAW        false

#include <opencv2/opencv.hpp>
#include<Eigen/Dense>


using namespace std;
namespace kltvo
{
namespace utils
{
void load_camCalib_yaml(string path, cv::Mat &K, double &baseline);

std::vector<double> getQuaternion(cv::Mat &R);
std::vector<float> mRot2Quat(const cv::Mat& m);

void saveTrajectoryKitti(const string &filename, std::list<cv::Mat> &relativeFramePoses);
void saveTrajectoryEuroc(const string &filename, std::list<cv::Mat> &relativeFramePoses, std::vector<double> frameTimeStamps);

void drawPointfImage(const cv::Mat &im, const std::vector<cv::Point2f> pts, const string &filename);
void writeOnLogFile(const string &name, const string &value);
void drawGridAndPoints(const cv::Mat &im, const std::vector<cv::Point2f> &pts, const string &fileName, int frameGridRows, int frameGridCols);
void logFeatureExtraction(const std::vector<cv::KeyPoint> &kpts_l, const std::vector<cv::KeyPoint> &kpts_r,
                        const std::vector<cv::Point2f> &pts, const cv::Mat &im, int frameGridRows, int frameGridCols);
void logStereoMatching(const cv::Mat &im_r, const cv::Mat &im_l, const std::vector<cv::DMatch> &mrl,
                        const std::vector<cv::Point2f> &pts_r, const std::vector<cv::Point2f> &pts_l);
void drawMatches(const cv::Mat &left_image, const cv::Mat &right_image, const std::vector<cv::Point2f> &kpts_l,
                    const std::vector<cv::Point2f> &kpts_r, const std::vector<cv::DMatch> &matches, bool hold, const std::string &prefix);
void logLocalMaping(const std::vector<cv::Point3f> &pts3D, double &meanError);
void logFeatureTracking(const std::vector<cv::Point2f> &pts_l0, const std::vector<cv::Point2f> &pts_r1, const cv::Mat &fmat,
                        const std::vector<cv::Point2f> &pts_l1, const std::vector<bool> &inliers, const cv::Mat &im_l0,
                        const cv::Mat &im_l1, const std::vector<cv::DMatch> &mll);
void logQuadMatching(const cv::Mat &im_l1, const cv::Mat &im_r1, const std::vector<cv::Point2f> &pts_l1,
                    const std::vector<cv::Point2f> &pts_r1, const std::vector<cv::DMatch> &mlr1, int numPts);
void drawFarAndClosePts (const std::vector<cv::Point2f> &pts, const cv::Mat &im, const std::vector<bool> &isClose);
void drawEpLines(const std::vector<cv::Point2f> &pts_l, const std::vector<cv::Point2f> &pts_r, const cv::Mat &F,
                    const std::vector<bool> &inliers, int rightFlag, const cv::Mat &image, const cv::Mat &image1,
                    const std::vector<cv::DMatch> &matches);
} // namespace utils
} // namespace kltvo

#endif //KLTVO_UTILS_H

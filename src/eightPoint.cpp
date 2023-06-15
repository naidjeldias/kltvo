//
// Created by nigel on 06/12/18.
//

#include "eightpoint.hpp"

using namespace cv;

namespace kltvo
{
EightPoint::EightPoint(double probability, int minSet, int maxIteration, double maxError){
    //initialize random seed for ransac
    //srand((unsigned)time(NULL));
    ransacMaxIt     = maxIteration;
    ransacMinSet    = minSet;
    ransacProb      = probability;
    ransacTh        = maxError;
}

void EightPoint::setRansacParameters(double probability, int minSet, int maxIteration, double maxError) {
    ransacMaxIt     = maxIteration;
    ransacMinSet    = minSet;
    ransacProb      = probability;
    ransacTh        = maxError;
}

std::vector<int> EightPoint::generateRandomIndices(const unsigned long &maxIndice, const unsigned int &vecSize){
    std::vector<int> randValues;
    int index;

    do{
        index = rand() % maxIndice;
        if(!(std::find(randValues.begin(), randValues.end(), index) != randValues.end()))
            randValues.push_back(index);
    }while(randValues.size() < vecSize);
//    std::cout << "----------------------------- \n";
//    std::cout << "Rand vector: \n";
//     for(int i = 0; i < randValues.size(); i++)
//         std::cout << randValues.at(i) << std::endl;
//    std::cout << "----------------------------- \n";

    return randValues;
}

void EightPoint::operator()(const std::vector<Point2f> &kpt_l, const std::vector<Point2f> &kpt_r,
                                              std::vector<DMatch>& finalMatches, std::vector<bool> &bestInliers, bool normalize,
                                              int method, cv::Mat &bestFmat) {

    finalMatches.clear();

    std::vector<Point2f> pts_left, pts_right;
    //matrix for normalizing transformations
    Mat leftScaling     =  Mat::zeros(3,3,CV_64F);
    Mat rightScaling    =  Mat::zeros(3,3,CV_64F);
    //Fundamental matrix
    Mat fmat;

    double* errorVect = new double[kpt_l.size()];

    if(normalize && method == 0) computeMatNormTransform(kpt_l, kpt_r, kpt_l.size(), leftScaling, rightScaling);

    unsigned int bestNumInliers = ransacMinSet;
    // int bestNumInliers = 0;

    int n = 0;
    long int r              = 1000;//adjusted dinamically
    long double bestStdDev  = LDBL_MAX;
    while (n < r && n < ransacMaxIt){

        std::vector<int> randValues;    //vector of rand index
        std::vector<DMatch> matches_;   //inliers matches in each iteration
        //-----------find inliers
        std::vector<bool> inliers;
        unsigned int numInliers = 0;

        //compute indices to pick 8 random points
        randValues      = generateRandomIndices(kpt_l.size(), ransacMinSet);

        //compute fundamental matrix with the subset
        if(method == 0)
            fmat            = computeFundamentalMatrix(kpt_l, kpt_r, randValues, leftScaling, rightScaling, normalize);
        else
            fmat            = findFundamentalMat(kpt_l, kpt_r, FM_8POINT);

        if(fmat.empty()) continue;

        //validate model againts the init set
        int pos = 0;
        long double meanError = 0;
        for(unsigned int i = 0; i < kpt_l.size() ; i++){

            //validadte against other elements
            if(!(std::find(randValues.begin(), randValues.end(), i) != randValues.end())){

                Mat X_l   = Mat::zeros(3,1,CV_64F);
                Mat X_r   = Mat::zeros(3,1,CV_64F);

                //point on left frame
                X_l.at<double>(0)     = kpt_l.at(i).x;
                X_l.at<double>(1)     = kpt_l.at(i).y;
                X_l.at<double>(2)     = 1.0;
                //point on right frame
                X_r.at<double>(0)   = kpt_r.at(i).x;
                X_r.at<double>(1)   = kpt_r.at(i).y;
                X_r.at<double>(2)   = 1.0;

                double d   =  sampsonError(fmat, X_l, X_r);

                if( d < ransacTh){
                    inliers.push_back(true);
                    errorVect[numInliers] = sqrt(d);
                    meanError += sqrt(d);
                    numInliers++;
                    double dst = euclideanDist(kpt_l.at(i), kpt_r.at(i));

                    DMatch match (pos,pos, dst);
                    matches_.push_back(match);
                    pos++;

                }else{
                    inliers.push_back(false);
                }


            }else{
                inliers.push_back(true);//the points from subset are considered as inliers
                numInliers++;
                double dst = euclideanDist(kpt_l.at(i), kpt_r.at(i));
                DMatch match (pos,pos, dst);
                matches_.push_back(match);
                pos++;
            }

        }

        meanError /= (long double) numInliers;

        long double stdDev = 0;
        {
            for (unsigned int p=0; p<numInliers; p++)
            {
                long double delta = errorVect[p]-meanError;
                stdDev += delta*delta;
            }
        }
        stdDev /= (double)(numInliers);

        if((numInliers > bestNumInliers) || (numInliers == bestNumInliers && stdDev<bestStdDev)){

            bestInliers     = inliers;
            finalMatches    = matches_;

            bestFmat        = fmat.clone();
            bestStdDev      = stdDev;
            bestNumInliers  = numInliers;

            //fraction of inliers in the set of points
            double w    = (double) bestNumInliers / (double) kpt_l.size();

            //probability of not all N points are inliers
            //in each iteration we pick N points that are all inliers with probability w^N
            double p1   = 1 - pow(w, ransacMinSet);
            p1 = MAX(LDBL_MIN, p1);     // Avoid division by -Inf
            p1 = MIN(1-LDBL_MIN, p1);   // Avoid division by 0.
            
            //probability of not all N points are inliers in r iterations is (1 - w^N)^r
            //the probability that in r iteration, at least once, all N points are inliers: p = 1-(1 - W^N)^r
            r = std::ceil(log(1 - ransacProb)/log(p1));

        }
        n ++;
    }

    ransacNumit = n;

//    std::cout << "8 point algorithm RANSAC it: " << ransacNumit << std::endl;

    delete [] errorVect;


}

cv::Mat EightPoint::computeFundamentalMatrix(const std::vector<Point2f> &kpt_l, const std::vector<Point2f> &kpt_r, const std::vector<int> &indices,
                                             const cv::Mat &T, const cv::Mat &T_l, bool normalize){
    //square matrix 9x9
    Mat A   = Mat::zeros(9,9,CV_64F);

    Mat  D, U, Vt;

    //fill mat A
    for (int row = 0; row < 8; row++){

        int i = indices.at(row);
        double xLeft, yLeft, xRight, yRight;

        if(normalize){
            xLeft = kpt_l.at(i).x * T.at<double> (0,0) + T.at<double>(0,2);
            yLeft = kpt_l.at(i).y * T.at<double> (1,1) + T.at<double>(1,2);

            xRight = kpt_r.at(i).x * T_l.at<double> (0,0) + T_l.at<double>(0,2);
            yRight = kpt_r.at(i).y * T_l.at<double> (1,1) + T_l.at<double>(1,2);
        }else{
            xLeft = kpt_l.at(i).x;
            yLeft = kpt_l.at(i).y;

            xRight = kpt_r.at(i).x;
            yRight = kpt_r.at(i).y;
        }


        A.at<double>(row,0) = xLeft*xRight;
        A.at<double>(row,1) = xRight*yLeft;
        A.at<double>(row,2) = xRight;
        A.at<double>(row,3) = yRight*xLeft;
        A.at<double>(row,4) = yRight*yLeft;
        A.at<double>(row,5) = yRight;
        A.at<double>(row,6) = xLeft;
        A.at<double>(row,7) = yLeft;
        A.at<double>(row,8) = 1.0;

    }

    // std::cout << "Mat A: "<< A << std::endl;
    // -------------------------retrieve fundamental matrix from SVD of A
    // F = UDV_t
    SVD::compute(A,D,U,Vt);

    std::vector<double> fvec = Vt.row(8);

    Mat F = Mat::zeros(3, 3, CV_64F);

    for(int i=0; i < F.rows; i++){
        F.at<double>(i,0) = fvec.at(3*i);
        F.at<double>(i,1) = fvec.at(3*i + 1);
        F.at<double>(i,2) = fvec.at(3*i + 2);
    }

    //--------------------------force rank 2 of F
    //D_l = diag(r,s,t) so F_l = U_ldiag(r,s,0)Vt_l
    Mat F_l, D_l, U_l, Vt_l;
    SVD::compute(F,D_l,U_l,Vt_l);
    Mat diag    = Mat::zeros(3,3, CV_64F);
    diag.at<double>(0,0) = D_l.at<double>(0);
    diag.at<double>(1,1) = D_l.at<double>(1);
    // std::cout << "Diag: " << diag << std::endl;
    F_l = U_l * diag * Vt_l;

    //-------------------------denormalize F_l
    if(normalize) F_l =  T_l.t() * F_l * T;

    //std::cout << "Fmatrix: " << Fmatrix << std::endl;
    F_l = F_l/F_l.at<double>(2,2);

    return F_l;

}

void EightPoint::computeMatNormTransform(const std::vector<Point2f> &kpt_l, const std::vector<Point2f> &kpt_r, unsigned long nPts,
                                         cv::Mat &leftScalingMat, cv::Mat &rightScalingMat){

    Point2f meanLeft;
    Point2f meanRight;

    meanLeft.x  = 0.0; meanLeft.y   = 0.0;
    meanRight.x = 0.0; meanRight.y  = 0.0;

    for(unsigned i = 0; i < nPts; i++){

        meanLeft.x      += kpt_l.at(i).x;
        meanLeft.y      += kpt_l.at(i).y;

        meanRight.x     += kpt_r.at(i).x;
        meanRight.y     += kpt_r.at(i).y;

    }

    //means
    meanLeft.x      /= (double) nPts;
    meanLeft.y      /= (double) nPts;
    meanRight.x     /= (double) nPts;
    meanRight.y     /= (double) nPts;

    double leftDst      = 0.0;
    double rightDst     = 0.0;

    //normalization of the points: distance from centroide
    for(unsigned i = 0; i < nPts; i++){

        double deltaX   = meanLeft.x - kpt_l.at(i).x;
        double deltaY   = meanLeft.y - kpt_l.at(i).y;

        leftDst         += sqrt(deltaX*deltaX + deltaY*deltaY);

        deltaX           = meanRight.x - kpt_r.at(i).x;
        deltaY           = meanRight.y - kpt_r.at(i).y;

        rightDst        += sqrt(deltaX*deltaX + deltaY*deltaY);
    }

    //mean of distance
    leftDst     /= (double) nPts;
    rightDst    /= (double) nPts;

    //scale factor
    double scaleLeft    = sqrt(2.0)/leftDst;
    double scaleRight   = sqrt(2.0)/rightDst;

    leftScalingMat.at<double>(0,0)  = scaleLeft;
    leftScalingMat.at<double>(1,1)  = scaleLeft;
    leftScalingMat.at<double>(0,2)  = -scaleLeft*meanLeft.x;
    leftScalingMat.at<double>(1,2)  = -scaleLeft*meanLeft.y;
    leftScalingMat.at<double>(2,2)  = 1.0;

    rightScalingMat.at<double>(0,0)  = scaleRight;
    rightScalingMat.at<double>(1,1)  = scaleRight;
    rightScalingMat.at<double>(0,2)  = -scaleRight*meanRight.x;
    rightScalingMat.at<double>(1,2)  = -scaleRight*meanRight.y;
    rightScalingMat.at<double>(2,2)  = 1.0;


}


double EightPoint::sampsonError(cv::Mat fmat, cv::Mat left_pt, cv::Mat right_pt){

    Mat matTmp         = right_pt.t() * fmat;
    matTmp             = matTmp  * left_pt;

    Mat letfTmpMat     = fmat * left_pt;
    Mat rightTmpMat    = fmat.t() * right_pt;

    double num = (double) matTmp.at<double>(0,0);

    double den = letfTmpMat.at<double>(0)*letfTmpMat.at<double>(0) + letfTmpMat.at<double>(1)*letfTmpMat.at<double>(1) +
                 rightTmpMat.at<double>(0)*rightTmpMat.at<double>(0) + rightTmpMat.at<double>(1)*rightTmpMat.at<double>(1);

    return ((num*num)/den);
}

double EightPoint::euclideanDist(const cv::Point2d &p, const cv::Point2d &q) {
    Point2d diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

int EightPoint::getRansacNumit()
{
    return ransacNumit;
}
} // namespace kltvo
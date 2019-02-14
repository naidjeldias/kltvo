//
// Created by nigel on 21/01/19.
//

#include "tracking.h"

using namespace cv;

Tracking::Tracking() {

    initPhase = true;
}

void Tracking::start(const Mat &imLeft, const Mat &imRight) {


    if (initPhase){
        imLeft0     = imLeft;
        imRight0    = imRight;

        initPhase = false;
    }else{
        std::vector<KeyPoint> kpts;

        Ptr<FeatureDetector> detector = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::FAST_SCORE, 31);

        detector -> detect(imLeft0, kpts);

        //convert vector of keypoints to vector of Point2f
        std::vector<Point2f> prevPoints, nextPoints, rightPoints;
        for (auto& kpt:kpts){
            prevPoints.push_back(kpt.pt);
        }

        TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

        cornerSubPix(imLeft0, prevPoints, Size(7,7), Size(-1,-1), criteria);

        std::vector <Mat> left0_pyr, right0_pyr, left1_pyr;

        Size win (21,21);
        int maxLevel = 4;

        Mat status0, status1;

        //TO DO - use the pyramid in keyframe selection
        //buildOpticalFlowPyramid(imLeft0, left0_pyr, win, maxLevel, true);
        //buildOpticalFlowPyramid(imLeft, left1_pyr, win, maxLevel, true);

        calcOpticalFlowPyrLK(imLeft0, imLeft, prevPoints, nextPoints, status1, noArray(), win, maxLevel,
                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), 1);

        //TO DO - method for setereo matching
        //buildOpticalFlowPyramid(imRight0, right0_pyr, win, maxLevel, true);
        calcOpticalFlowPyrLK(imLeft0, imRight0, prevPoints, rightPoints, status0, noArray(), win, maxLevel,
                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), 1);


//        localMapping(prevPoints, rightPoints, )

        imLeft0     = imLeft;
        imRight0    = imRight;

    }

}

void Tracking::stereoMatching(std::vector<cv::Point2f> &pts_l, std::vector<cv::Point2f> &pts_r, const cv::Mat &imLeft,
                              const cv::Mat &imRight, const std::vector<bool> &inliers,
                              std::vector<cv::DMatch> &matches) {

    TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

    cornerSubPix(imLeft, pts_l, Size(7,7), Size(-1,-1), criteria);

    // The disparity range defines how many pixels away from the block's location
    // in the first image to search for a matching block in the other image.
    int disparityRange = 200;

//    imshow("Image Right", imRight);
//
//    waitKey(0);

    //Define the size of the blocks for block matching.
    int halfBlockSize = 3;
    int blockSize = 2 * halfBlockSize + 1;

    int width = imRight.size().width;
    int height = imRight.size().height;

    int pos = 0;
    for (auto &pt:pts_l){
        Mat template_ (blockSize, blockSize, CV_64F);
        //get pixel neighbors
        //        Mat template_ = imLeft(Rect ((int)pt.x - halfBlockSize, (int)pt.y - halfBlockSize, halfBlockSize, halfBlockSize)).clone();
        for (int i=0; i < blockSize; i++){
            for(int j=0; j < blockSize; j++){
                int x = (int) pt.x - (halfBlockSize - i);
                int y = (int) pt.y - (halfBlockSize - j);
                //check frame limits
                if(x >= 0 && x < width && y >= 0 && y < height){
                    Scalar intensity = imLeft.at<uchar>(y,x);
                    template_.at<float>(j,i) = (int) intensity[0];
                }else{
                    template_.at<float>(j,i) = 0;
                }
            }
        }

        // Set the min column bounds for the template search.
        int minc = MAX(blockSize, (int) pt.x - disparityRange);

        int minSAD = INT_MAX;
        Point2f bestPt;


        for (int k = pt.x; k >= minc; k--){

            Point point (k, pt.y);

//            Mat block = imRight(Rect (i - halfBlockSize, i - halfBlockSize, halfBlockSize, halfBlockSize)).clone();

            //compute SAD
            int sum =0;
            for (int i=0; i < blockSize; i++){
                for(int j=0; j < blockSize; j++){
                    int x = (int) point.x - (halfBlockSize - i);
                    int y = (int) point.y - (halfBlockSize - j);
                    //check frame limits
                    if(x >= 0 && x < width && y >= 0 && y < height){
                        Scalar intensity = imRight.at<uchar>(y,x);
                        sum += abs(template_.at<float>(j,i) - intensity[0]);
                    }else{
                        sum += abs(template_.at<float>(j,i) - 0);
                    }
                }
            }

            if(sum < minSAD){
                minSAD = sum;
                bestPt = point;
            }

        }
        pts_r.push_back(bestPt);
        double dst = norm(Mat(pts_l.at(pos)), Mat(pts_r.at(pos)));
        DMatch match (pos,pos, dst);
        matches.push_back(match);

        pos ++;
    }

    cornerSubPix(imRight, pts_r, Size(7,7), Size(-1,-1), criteria);
}

void Tracking::localMapping(const std::vector<cv::Point2d> &pts_l, const std::vector<cv::Point2d> &pts_r,
                            std::vector<cv::Point3d> &pts3D, const std::vector<DMatch> &matches) {

    Point2f kp_l, kp_r;
    // std::vector<Landmark> points3d;
    double w0, w1;

    for( int i = 0; i < matches.size() ; i++ ){

        kp_l = pts_l.at(matches[i].queryIdx);
        kp_r = pts_r.at(matches[i].trainIdx);

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

            // std::cout << "Mat A: " << A << std::endl;

            SVD::compute(A,D,U,Vt);

            // std::cout << "Vt: " << Vt << std::endl;

            point3d = Vt.row(3).t();

            // std::cout << "point 3D: " << point3d << std::endl;

            point3d = point3d.rowRange(0,4)/point3d.at<double>(3);

            // std::cout << "point 3D cc: " << point3d << std::endl;

            Mat p0 = P1*point3d;
            Mat p1 = P2*point3d;

            // std::cout << "p0 : " << p0 << std::endl;
            // std::cout << "p1 : " << p1 << std::endl;


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

            // std::cout << "dist: " << dist << std::endl;
            //  std::cout << "it: " << j << std::endl;

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
        pt3d.y           = point3d.at<double>(1);
        pt3d.x           = point3d.at<double>(2);

        pts3D.push_back(pt3d);
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
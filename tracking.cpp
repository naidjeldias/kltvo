//
// Created by nigel on 21/01/19.
//

#include "tracking.h"

using namespace cv;

Tracking::Tracking() {

    once = true;
}

void Tracking::start(Mat &imLeft, Mat &imRight) {

    std::vector<KeyPoint> kpt;
    
    if(once){
        bucketFeatureExtraction(imLeft, Size (207,74), kpt);
        //once = false;
    }


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
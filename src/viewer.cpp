#include "viewer.hpp"

Viewer::Viewer(Tracking* trackerPtr):trackerPtr_(trackerPtr)
{}

void Viewer::computeOpenGLCameraMatrix(const cv::Mat& cameraPose, pangolin::OpenGlMatrix& Twc)
{
    cv::Mat Rwc(3,3,CV_32F);
    cv::Mat twc(3,1,CV_32F);

    Rwc = cameraPose.rowRange(0,3).colRange(0,3);
    twc = cameraPose.rowRange(0,3).col(3);
    

    Twc.m[0] = Rwc.at<float>(0,0);
    Twc.m[1] = Rwc.at<float>(1,0);
    Twc.m[2] = Rwc.at<float>(2,0);
    Twc.m[3]  = 0.0;

    Twc.m[4] = Rwc.at<float>(0,1);
    Twc.m[5] = Rwc.at<float>(1,1);
    Twc.m[6] = Rwc.at<float>(2,1);
    Twc.m[7]  = 0.0;

    Twc.m[8] = Rwc.at<float>(0,2);
    Twc.m[9] = Rwc.at<float>(1,2);
    Twc.m[10] = Rwc.at<float>(2,2);
    Twc.m[11]  = 0.0;

    Twc.m[12] = twc.at<float>(0);
    Twc.m[13] = twc.at<float>(1);
    Twc.m[14] = twc.at<float>(2);
    Twc.m[15]  = 1.0;
}
cv::Mat Viewer::computeGlobalPose()
{
    // std::lock_guard<std::mutex> lock(tracker_);
    cv::Mat current_pose = trackerPtr_->getCurrentPose();
    
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

    global_pose_ = global_pose_ * inv_pose;

    std::cout << "Pose: " << global_pose_.at<float>(0,3) << ", " << global_pose_.at<float>(1,3)<< ", " << global_pose_.at<float>(2,3) << std::endl;
    return global_pose_;
}
void Viewer::run()
{
    pangolin::CreateWindowAndBind("Trajectory viewer",1024,768);
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);
    
    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
    pangolin::ModelViewLookAt(-1,1,-1, 0,0,0, pangolin::AxisY)
  );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::Display("cam")
      .SetBounds(0,1.0f,0,1.0f,-640/480.0)
      .SetHandler(new pangolin::Handler3D(s_cam));
    
    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    
    while(!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // d_cam.Activate(s_cam);

        // s_cam.Follow(Twc);

        // cv::Mat globalPose = computeGlobalPose();
        // computeOpenGLCameraMatrix(globalPose, Twc);

        // glLineWidth(1);
        // glColor4f(0.0f,1.0f,0.0f,0.6f);
        // glBegin(GL_LINES);

        // cv::Mat cameraCenter = globalPose.rowRange(0,3).col(3);
        // glVertex3f(cameraCenter.at<float>(0),cameraCenter.at<float>(1),cameraCenter.at<float>(2));
        // glColor3f(1.0,1.0,1.0);
        // pangolin::glDrawColouredCube();

        // glEnd();
        // Swap frames and Process Events
        pangolin::FinishFrame();

    }


}
#include "viewer.hpp"

Viewer::Viewer(const string &strSettingPath):finishRequested_(false), trackingState_(Tracking::NOT_INITIALIZED), rotZ_(cv::Mat::eye(4,4,CV_32F))
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Viewer.Camera.fps"];
    if(fps<1)
        fps=30;
    updateRate_ = 1e3/fps;
    
    imageWidth_ = fSettings["Viewer.Camera.width"];
    imageHeight_ = fSettings["Viewer.Camera.height"];
    if(imageWidth_<1 || imageHeight_<1)
    {
        imageWidth_ = 640;
        imageHeight_ = 480;
    }

    viewpointX_ = fSettings["Viewer.ViewpointX"];
    viewpointY_ = fSettings["Viewer.ViewpointY"];
    viewpointZ_ = fSettings["Viewer.ViewpointZ"];
    viewpointF_ = fSettings["Viewer.ViewpointF"];

    rotZ_.at<float>(0, 0) = -1.0;
    rotZ_.at<float>(0, 1) = 0.0;
    rotZ_.at<float>(1, 0) = 0.0;
    rotZ_.at<float>(1, 1) = -1.0;
    rotZ_.at<float>(2, 2) = 1.0;
    
}

Viewer::~Viewer()
{
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
    // for more info see: https://www.songho.ca/opengl/gl_transform.html
    pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(imageWidth_, imageHeight_, viewpointF_, viewpointF_, 320, 240, 0.1, 1000),
      pangolin::ModelViewLookAt(viewpointX_, viewpointY_, viewpointZ_, 0, 0, 0, pangolin::AxisY));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::Display("cam")
      .SetBounds(0,1.0f,pangolin::Attach::Pix(175),1.0f,-imageWidth_/(float)imageHeight_)
      .SetHandler(new pangolin::Handler3D(s_cam));
    
    // Menu.
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));

    pangolin::Var<bool> menu_follow_cam("menu.Follow Camera", true, true);
    pangolin::Var<bool> show_cam("menu.Show Camera", true, true);
    pangolin::Var<bool> show_traj("menu.Show Traj", true, true);
    pangolin::Var<bool> show_features("menu.Features", true, true);
    pangolin::Var<bool> show_keypoints("menu.Keypoints", true, true);


    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    
    while(!pangolin::ShouldQuit())
    {

        if(trackingState_ == Tracking::OK)
        {

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            d_cam.Activate(s_cam);
            glClearColor(0.0f, 0.0f, 0.0f,1.0f);
                  
            if(!cameraPoses_.back().empty())
            {
              computeOpenGLCameraMatrix(convertToOpenGLFrame(cameraPoses_.back()), Twc);
              if(menu_follow_cam)
                s_cam.Follow(Twc);
              
            }

            if (show_cam.Get()) 
            {
              glColor3f(0.0f, 1.0f, 0.0f);
              renderCamera(Twc);
            }

            if (show_traj.Get()) 
            {
              glColor3f(1.0f, 0.0f, 0.0f);
              drawTrajectory();
            }
            // Swap frames and Process Events
            pangolin::FinishFrame();

            // Draw features
            if(show_features.Get())
              drawPointsImage(imLeft0_, features_, cv::Scalar(0,0,255));

            // Draw keypoints
            if(show_keypoints.Get())
              drawPointsImage(imLeft0_, keypoints_, cv::Scalar(0,255,0));
            
            cv::imshow("Current Frame",imLeft0_);
            
        }

        if(finishRequested_)
          break;
        cv::waitKey(updateRate_);

    }


}

void Viewer::shutdown()
{
  finishRequested_ = true;
}


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

void Viewer::drawTrajectory()
{
    glLineWidth(3.0);
    glBegin(GL_LINE_STRIP);
    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    for (const cv::Mat& cameraPose : cameraPoses_) 
    {
      cv::Mat global_pose = convertToOpenGLFrame(cameraPose);
      glVertex3f(global_pose.at<float>(0,3), global_pose.at<float>(1,3), global_pose.at<float>(2,3));
    }

    glEnd();
}

void Viewer::renderCamera(const pangolin::OpenGlMatrix& camtMat)
{
    const float w = 0.5;
    const float h = w * 0.75;
    const float z = w * 0.6;

    glPushMatrix();

    glMultMatrixd(camtMat.m);

    glLineWidth(3.0);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();

    pangolin::glDrawAxis(camtMat, 0.2);
}

cv::Mat Viewer::convertToOpenGLFrame(const cv::Mat& camMat)
{
  cv::Mat camMatOpenGL = rotZ_ * camMat;
  return camMatOpenGL;
}

void Viewer::drawPointsImage(cv::Mat &im, const std::vector<cv::Point2f> &pts, cv::Scalar color)
{
  std::lock_guard<std::mutex> lg(data_buffer_mutex_);
  if(pts.empty())
    return;
  // Draw the points
  for (const cv::Point2f& pt : pts) 
  {
    cv::circle(im, pt, 2, color, 2);
  }
}

void Viewer::update(Tracking* trackerPtr)
{
  std::lock_guard<std::mutex> lg(data_buffer_mutex_);
  trackingState_ = trackerPtr->trackingState_;
  if(trackingState_ == Tracking::OK)
  {
    trackerPtr->currentKeyframe_.imLeft0.copyTo(imLeft0_);
    cv::cvtColor(imLeft0_, imLeft0_, cv::COLOR_GRAY2RGB);
    cameraPoses_ = trackerPtr->cameraPoses_;
    features_ = trackerPtr->currentKeyframe_.features;
    keypoints_ = trackerPtr->currentKeyframe_.keypoints;
  }
}
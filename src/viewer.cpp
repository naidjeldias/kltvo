#include "viewer.hpp"

Viewer::Viewer()
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

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f,1.0f);

        {
          std::lock_guard<std::mutex> lg(data_buffer_mutex_);
          if(!cameraPoses_.empty())
          {
            computeOpenGLCameraMatrix(cameraPoses_.back(), Twc);
            s_cam.Follow(Twc);
            
          }
        }

        glColor3f(1.0f, 0.0f, 0.0f);
        drawTrajectory();

        // Swap frames and Process Events
        pangolin::FinishFrame();

    }


}

void Viewer::setCameraPoses(const std::vector<cv::Mat>& cameraPoses)
{
  std::lock_guard<std::mutex> lg(data_buffer_mutex_);
  cameraPoses_ = cameraPoses;
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
    for (const cv::Mat& global_pose : cameraPoses_) 
    {
        glVertex3f(global_pose.at<float>(0,3), global_pose.at<float>(1,3), global_pose.at<float>(2,3));
    }

    glEnd();
}
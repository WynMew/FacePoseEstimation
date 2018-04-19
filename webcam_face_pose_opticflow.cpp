#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

using namespace dlib;
using namespace std;
#define UNKNOWN_FLOW_THRESH 1e9

void makecolorwheel(std::vector<cv::Scalar> &colorwheel)
{
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;

    int i;

    for (i = 0; i < RY; i++) colorwheel.push_back(cv::Scalar(255,       255*i/RY,     0));
    for (i = 0; i < YG; i++) colorwheel.push_back(cv::Scalar(255-255*i/YG, 255,       0));
    for (i = 0; i < GC; i++) colorwheel.push_back(cv::Scalar(0,         255,      255*i/GC));
    for (i = 0; i < CB; i++) colorwheel.push_back(cv::Scalar(0,         255-255*i/CB, 255));
    for (i = 0; i < BM; i++) colorwheel.push_back(cv::Scalar(255*i/BM,      0,        255));
    for (i = 0; i < MR; i++) colorwheel.push_back(cv::Scalar(255,       0,        255-255*i/MR));
}

void motionToColor(cv::Mat flow, cv::Mat &color)
{
    if (color.empty())
        color.create(flow.rows, flow.cols, CV_8UC3);

    static std::vector<cv::Scalar> colorwheel; //Scalar r,g,b
    if (colorwheel.empty())
        makecolorwheel(colorwheel);

    // determine motion range:
    float maxrad = -1;

    // Find max flow to normalize fx and fy
    for (int i= 0; i < flow.rows; ++i)
    {
        for (int j = 0; j < flow.cols; ++j)
        {
            cv::Vec2f flow_at_point = flow.at<cv::Vec2f>(i, j);
            float fx = flow_at_point[0];
            float fy = flow_at_point[1];
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
                continue;
            float rad = sqrt(fx * fx + fy * fy);
            maxrad = maxrad > rad ? maxrad : rad;
        }
    }

    std::cout<<maxrad<<std::endl;

    for (int i= 0; i < flow.rows; ++i)
    {
        for (int j = 0; j < flow.cols; ++j)
        {
            uchar *data = color.data + color.step[0] * i + color.step[1] * j;
            cv::Vec2f flow_at_point = flow.at<cv::Vec2f>(i, j);

            float fx=0;
            float fy=0;
            if((flow_at_point[0]>5) && (flow_at_point[0] > maxrad/3))
            {
            	fx = flow_at_point[0] / maxrad;
            }

            /*
            if((flow_at_point[1]>10) && (flow_at_point[1] > maxrad/3))
            {
            	fy = flow_at_point[1] / maxrad;
            }
            */

            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
            {
                data[0] = data[1] = data[2] = 0;
                continue;
            }
            float rad = sqrt(fx * fx + fy * fy);

            float angle = atan2(-fy, -fx) / CV_PI;
            float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);
            int k0 = (int)fk;
            int k1 = (k0 + 1) % colorwheel.size();
            float f = fk - k0;
            //f = 0; // uncomment to see original color wheel

            for (int b = 0; b < 3; b++)
            {
                float col0 = colorwheel[k0][b] / 255.0;
                float col1 = colorwheel[k1][b] / 255.0;
                float col = (1 - f) * col0 + f * col1;
                if (rad <= 1)
                    col = 1 - rad * (1 - col); // increase saturation with radius
                else
                    col *= .75; // out of range
                data[2 - b] = (int)(255.0 * col);
            }
        }
    }
}

int main()
{

    // 3D model points.
    std::vector<cv::Point3d> model_points;
    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner

    try
    {
        cv::VideoCapture cap(0);
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;


        cv::Mat prevgray, gray, flow, cflow, frame;
        cv::namedWindow("flow", 1);
        cv::Mat motion2color;
        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp))
            {
                break;
            }
            frame=temp;
            double focal_length = temp.cols; // Approximate focal length.
            int iIC = temp.cols;
            int iIR = temp.rows;
            //std::cout<<iIC<< " " <<iIR << std::endl;
            cv::Point2d center = cv::Point2d(iIC/2,iIR/2);
            cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
            cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion

            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
            {
            	full_object_detection shape = pose_model(cimg, faces[i]);
            	//cout << "number of parts: "<< shape.num_parts() << endl;
            	//cout << "pixel position of first part:  " << shape.part(0)(1) << endl;
            	//cout << "pixel position of second part: " << shape.part(1) << endl;
                shapes.push_back(shape);

                std::vector<cv::Point2d> image_points;
                image_points.push_back( cv::Point2d(shape.part(30)(0), shape.part(30)(1)) );    // Nose tip
                image_points.push_back( cv::Point2d(shape.part(8)(0), shape.part(8)(1)) );    // Chin
                image_points.push_back( cv::Point2d(shape.part(36)(0), shape.part(36)(1)) );     // Left eye left corner
                image_points.push_back( cv::Point2d(shape.part(45)(0), shape.part(45)(1)) );    // Right eye right corner
                image_points.push_back( cv::Point2d(shape.part(48)(0), shape.part(48)(1)) );    // Left Mouth corner
                image_points.push_back( cv::Point2d(shape.part(54)(0), shape.part(54)(1)) );    // Right mouth corner
                //cout << "++++++++" <<endl;
                //cout << image_points[0] << endl;

                //cout << "Camera Matrix " << endl << camera_matrix << endl ;
                // Output rotation and translation
                cv::Mat rotation_vector; // Rotation in axis-angle form
                cv::Mat translation_vector;

                // Solve for pose
                cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);


                // Project a 3D point (0, 0, 1000.0) onto the image plane.
                // We use this to draw a line sticking out of the nose

                cv::vector<cv::Point3d> nose_end_point3D;
                cv::vector<cv::Point2d> nose_end_point2D;
                nose_end_point3D.push_back(cv::Point3d(0,0,1000.0));

                projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);

                for(int i=0; i < image_points.size(); i++)
                    {
                	 circle(temp, image_points[i], 3, cv::Scalar(0,0,255), -1);
                    }

                cv::line(temp,image_points[0], nose_end_point2D[0], cv::Scalar(255,0,0), 2);

                //cout << "Rotation Vector " << endl << rotation_vector << endl;
                //cout << "Translation Vector" << endl << translation_vector << endl;
                //cout <<  nose_end_point2D << endl;

            }
            cv_image<bgr_pixel> ncimg(temp);
            // Display it all on the screen
            win.clear_overlay();
            win.set_image(ncimg);
            win.add_overlay(render_face_detections(shapes));

            double t = (double)cvGetTickCount();
            cvtColor(frame, gray, CV_BGR2GRAY);
            if( prevgray.data )
                {
                    calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
                    motionToColor(flow, motion2color);
                    imshow("flow", motion2color);
                }
            if(cv::waitKey(10)>=0)
                    break;
            std::swap(prevgray, gray);
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}


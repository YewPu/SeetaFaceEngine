#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "face_detection.h"
#include "face_alignment.h"
#include <time.h>
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

std::string DATA_DIR = "D:/git/SeetaFaceEngine/FaceAlignment/data/";
std::string MODEL_DIR = "D:/git/SeetaFaceEngine/FaceAlignment/model/";

seeta::FaceDetection detector("D:/git/SeetaFaceEngine/FaceDetection/model/seeta_fd_frontal_v1.0.bin");
// Initialize face alignment model 
seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());

string outputVideoPath = "D:\\git\\test.avi";
VideoWriter outputVideo;
void alignment(VideoCapture &cap) {
    Mat src;
    static int frameStep = 0;
    if (!cap.read(src)) {
        cerr << "ERROR! blank frame grabbed\n";
        return;
    }
    //imshow("Live", src);
    //return;
    //if (10 > frameStep)
    //{   
    //    frameStep++;
    //    imshow("Live", src);
    //    return;
    //}
    //frameStep = 0;
    //seeta::FaceDetection detector("D:/git/SeetaFaceEngine/FaceDetection/model/seeta_fd_frontal_v1.0.bin");
    //detector.SetMinFaceSize(40);
    //detector.SetScoreThresh(2.f);
    //detector.SetImagePyramidScaleFactor(0.8f);
    //detector.SetWindowStep(4, 4);
    //// Initialize face alignment model 
    //seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());

    //load image
    cv::Mat img_grayscale;
    cv::cvtColor(src, img_grayscale, COLOR_BGR2GRAY);
    cv::Mat img_color;
    cv::cvtColor(src, img_color, COLOR_BGR2BGRA);

    int pts_num = 5;
    int im_width = img_grayscale.cols;
    int im_height = img_grayscale.rows;
    unsigned char* data = new unsigned char[im_width * im_height];
    unsigned char* data_ptr = data;
    unsigned char* image_data_ptr = (unsigned char*)img_grayscale.data;
    int h = 0;
    for (h = 0; h < im_height; h++) {
        memcpy(data_ptr, image_data_ptr, im_width);
        data_ptr += im_width;
        image_data_ptr += img_grayscale.step;
    }

    seeta::ImageData image_data;
    image_data.data = data;
    image_data.width = im_width;
    image_data.height = im_height;
    image_data.num_channels = 1;

    // Detect faces
    std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);
    int32_t face_num = static_cast<int32_t>(faces.size());

    if (face_num == 0)
    {
        delete[]data;
        img_grayscale.release();
        img_color.release();
        return ;
    }

    for (int faceIndex = 0; faceIndex <face_num; faceIndex++)
    {
        // Detect 5 facial landmarks
        seeta::FacialLandmark points[5];
        point_detector.PointDetectLandmarks(image_data, faces[faceIndex], points);
        faces.size();
        // Visualize the results
        cv::rectangle(img_color, cvPoint(faces[faceIndex].bbox.x, faces[faceIndex].bbox.y), cvPoint(faces[faceIndex].bbox.x + faces[faceIndex].bbox.width - 1, faces[faceIndex].bbox.y + faces[faceIndex].bbox.height - 1), CV_RGB(255, 0, 0));
        for (int i = 0; i < pts_num; i++)
        {
           cv::circle(img_color, cvPoint((int)points[i].x, (int)points[i].y), 2, CV_RGB(0, 255, 0), CV_FILLED);
        }
    }
    imshow("Live", img_color);
    delete[]data;

}

int main(int, char**)
{
    detector.SetMinFaceSize(40);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);
    Mat src;
    // use default camera as video source
    VideoCapture cap(0);

    //cv::Size sWH = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
    //    (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    //outputVideo.open(outputVideoPath, CV_FOURCC('M', 'P', '4', '2'), 25.0, sWH);
    //if (!outputVideo.isOpened()) {
    //    cerr << "Could not open the output video file for write\n";
    //    return -1;
    //}
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    //cap.set(CAP_PROP_FRAME_WIDTH, 1440);
    //cap.set(CAP_PROP_FRAME_HEIGHT, 1024);

    for (;;)
    {
        clock_t start, finish;
        double totaltime;
        start = clock();
        alignment(cap);
        finish = clock();
        totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
        cout << "\n��Ƶ֡����ʱ��Ϊ" << totaltime << "�룡" << endl;
        if (waitKey(10) >= 0)
            break;
    }
    //outputVideo.release();
    return 0;
}
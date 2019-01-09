#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "face_detection.h"
#include <time.h>
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

std::string DATA_DIR = "D:/git/SeetaFaceEngine/FaceAlignment/data/";
std::string MODEL_DIR = "D:/git/SeetaFaceEngine/FaceAlignment/model/";

seeta::FaceDetection detector("D:/git/SeetaFaceEngine/FaceDetection/model/seeta_fd_frontal_v1.0.bin");

string outputVideoPath = "D:\\git\\test.avi";
VideoWriter outputVideo;
void detection(VideoCapture &cap) {
    Mat src;
    static int frameStep = 0;
    if (!cap.read(src)) {
        cerr << "ERROR! blank frame grabbed\n";
        return;
    }

    //detector.SetMinFaceSize(40);
    //detector.SetScoreThresh(2.f);
    //detector.SetImagePyramidScaleFactor(0.8f);
    //detector.SetWindowStep(4, 4);
    cv::Mat img;
    cv::cvtColor(src, img, COLOR_BGR2BGRA);
    cv::Mat img_gray;
    if (img.channels() != 1)
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    else
        img_gray = img;

    seeta::ImageData img_data;
    img_data.data = img_gray.data;
    img_data.width = img_gray.cols;
    img_data.height = img_gray.rows;
    img_data.num_channels = 1;

    long t0 = cv::getTickCount();
    std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
    long t1 = cv::getTickCount();
    double secs = (t1 - t0) / cv::getTickFrequency();
    cout << "Detections takes " << secs << " seconds " << endl;
    cv::Rect face_rect;
    int32_t num_face = static_cast<int32_t>(faces.size());

    for (int32_t i = 0; i < num_face; i++) {
        face_rect.x = faces[i].bbox.x;
        face_rect.y = faces[i].bbox.y;
        face_rect.width = faces[i].bbox.width;
        face_rect.height = faces[i].bbox.height;

        cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
    }
    imshow("Live", img);
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
        detection(cap);
        finish = clock();
        totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
        cout << "\n视频帧处理时间为" << totaltime << "秒！" << endl;
        if (waitKey(10) >= 0)
            break;
    }
    //outputVideo.release();
    return 0;
}
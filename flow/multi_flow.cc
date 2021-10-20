
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#define FMT_HEADER_ONLY
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

vector<KeyPoint> keypoint_extraction(Mat img);
void optical_flow(Mat img_1, Mat img_2, vector<KeyPoint> &keypoints,
                  vector<KeyPoint> &keypoints_2);

void multi_optical_flow(Mat img_1, Mat img_2, vector<KeyPoint> &keypoints,
                        vector<KeyPoint> &keypoints_2);

int main(int argc, char **argv) {
  Mat img_1 = imread(argv[1], 0);
  Mat img_2 = imread(argv[2], 0);
  vector<KeyPoint> keypoints_2;
  vector<KeyPoint> keypoints = keypoint_extraction(img_1);
  multi_optical_flow(img_1, img_2, keypoints, keypoints_2);
  cout << "finish" << endl;
  Mat outImg;
  Mat outImg2;
  drawKeypoints(img_1, keypoints, outImg);
  drawKeypoints(img_2, keypoints_2, outImg2);
  imshow("ORB keypoints", outImg);
  imshow("ORB keypoints2", outImg2);
  waitKey(0);
}

vector<KeyPoint> keypoint_extraction(Mat img) {
  vector<KeyPoint> keypoints;
  Mat descriptors;
  Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
  detector->detect(img, keypoints);
  return keypoints;
}

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
  // boundary check
  if (x < 0)
    x = 0;
  if (y < 0)
    y = 0;
  if (x >= img.cols)
    x = img.cols - 1;
  if (y >= img.rows)
    y = img.rows - 1;
  uchar *data = &img.data[int(y) * img.step + int(x)];
  float xx = x - floor(x);
  float yy = y - floor(y);
  return float((1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] +
               (1 - xx) * yy * data[img.step] + xx * yy * data[img.step + 1]);
}

void optical_flow(Mat img_1, Mat img_2, vector<KeyPoint> &keypoints,
                  vector<KeyPoint> &keypoints_2) {
  int patch_size = 4;
  int iterations = 10;
  keypoints_2.resize(keypoints.size());

  for (int i = 0; i < keypoints.size(); i++) {
    KeyPoint kp = keypoints[i];
    KeyPoint kp2 = keypoints_2[i];
    double dx = kp2.pt.x - kp.pt.x;
    double dy = kp2.pt.y - kp.pt.y;
    double error;
    for (int iter = 0; iter < iterations; iter++) {
      Eigen::Vector<double, 2> J = Eigen::Vector<double, 2>::Zero();
      Eigen::Matrix<double, 2, 2> H = Eigen::Matrix<double, 2, 2>::Zero();
      Eigen::Vector<double, 2> g = Eigen::Vector<double, 2>::Zero();
      for (int x = -patch_size; x < patch_size; x++)
        for (int y = -patch_size; y < patch_size; y++) {
          error = GetPixelValue(img_1, kp.pt.x + x, kp.pt.y + y) -
                  GetPixelValue(img_2, kp.pt.x + dx + x, kp.pt.y + dy + y);
          J = -0.5 *
              Eigen::Vector<double, 2>(
                  GetPixelValue(img_2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                      GetPixelValue(img_2, kp.pt.x + dx + x - 1,
                                    kp.pt.y + dy + y),
                  GetPixelValue(img_2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                      GetPixelValue(img_2, kp.pt.x + dx + x,
                                    kp.pt.y + dy + y - 1));
          H += J * J.transpose();
          g += -J * error;
        }
      Eigen::Vector2d update = H.ldlt().solve(g);
      dx += update[0];
      dy += update[1];
    }
    keypoints_2[i].pt = kp.pt + Point2f(dx, dy);
  }
}

void multi_optical_flow(Mat img_1, Mat img_2, vector<KeyPoint> &keypoints,
                        vector<KeyPoint> &keypoints_2) {
  int layers = 4;
  double scale = 0.5;
  double scales[] = {1.0, 0.5, 0.25, 0.125};
  vector<Mat> imgs1;
  vector<Mat> imgs2;
  Mat img1 = img_1;
  Mat img2 = img_2;
  for (int i = 0; i < layers; i++) {
    imgs1.push_back(img1);
    imgs2.push_back(img2);
    resize(img1, img1, Size(img1.cols * scale, img1.rows * scale));
    resize(img2, img2, Size(img2.cols * scale, img2.rows * scale));
  }

  vector<KeyPoint> kp1, kp2;
  for (auto &kp : keypoints) {
    auto kp_top = kp;
    kp_top.pt *= scales[layers - 1];
    kp1.push_back(kp_top);
    kp2.push_back(kp_top);
  }

  for (int i = 0; i < layers; i++) {

    optical_flow(imgs1[layers - i - 1], imgs2[layers - i - 1], kp1, kp2);

    Mat outImg, outImg2;
    // drawKeypoints(imgs1[layers - i - 1], kp1, outImg);
    // drawKeypoints(imgs2[layers - i - 1], kp2, outImg2);
    // imshow("ORB keypoints", outImg);
    // imshow("ORB keypoints2", outImg2);
    // waitKey(0);
    if (i != layers - 1) {
      for (auto &kp : kp1)
        kp.pt /= scale;
      for (auto &kp : kp2)
        kp.pt /= scale;
    }
  }
  for (auto &kp : kp2)
    keypoints_2.push_back(kp);
}

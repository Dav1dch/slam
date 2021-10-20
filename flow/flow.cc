
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#define FMT_HEADER_ONLY
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <math.h>
#include <vector>

using namespace std;
using namespace cv;

vector<KeyPoint> keypoint_extraction(Mat img);
void optical_flow(Mat img_1, Mat img_2, vector<KeyPoint> &keypoints,
                  vector<KeyPoint> &keypoints_2);

int main(int argc, char **argv) {
  Mat img_1 = imread(argv[1], 0);
  Mat img_2 = imread(argv[2], 0);
  vector<KeyPoint> keypoints_2;
  vector<KeyPoint> keypoints = keypoint_extraction(img_1);
  optical_flow(img_1, img_2, keypoints, keypoints_2);
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
  int iterations = 100;
  keypoints_2.resize(keypoints.size());

  for (int i = 0; i < keypoints.size(); i++) {
    KeyPoint kp = keypoints[i];
    double dx = 0;
    double dy = 0;
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

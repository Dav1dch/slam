#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// paths
string left_file = "./left.png";
string disparity_file = "./disparity.png";

// bilinear interpolation
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

int main(int argc, char **argv) {

  cv::Mat left_img = cv::imread(left_file, 0);
  cv::Mat disparity_img = cv::imread(disparity_file, 0);

  // let's randomly pick pixels in the first image and generate some 3d points
  // in the first image's frame
  cv::RNG rng;
  int nPoints = 2000;
  int boarder = 20;
  vector<Eigen::Vector2d> pixels_ref;
  vector<double> depth_ref;

  // generate pixels in ref and load depth data
  for (int i = 0; i < nPoints; i++) {
    int x = rng.uniform(
        boarder, left_img.cols - boarder); // don't pick pixels close to boarder
    int y = rng.uniform(
        boarder, left_img.rows - boarder); // don't pick pixels close to boarder
    int disparity = disparity_img.at<uchar>(y, x);
    double depth =
        fx * baseline / disparity; // you know this is disparity to depth
    depth_ref.push_back(depth);
    pixels_ref.push_back(Eigen::Vector2d(x, y));
  }
}

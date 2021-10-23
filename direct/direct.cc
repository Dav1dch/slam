#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

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

void optimal_direct(Mat img_1, Mat img_2, vector<Eigen::Vector2d> &points,
                    vector<Eigen::Vector2d> &target_points, Sophus::SE3d &pose);

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

void optimal_direct(Mat img_1, Mat img_2,
                    vector<Eigen::Vector<double, 2>> &points,
                    vector<double> &depths,
                    vector<Eigen::Vector2d> &target_points,
                    Sophus::SE3d &pose) {
  int iterations = 10;
  int patch_size = 4;
  double error;
  for (int i = 0; i < points.size(); i++) {
    for (int iter = 0; iter < iterations; iter++) {
      Eigen::Matrix<double, 6, 6> H;
      Eigen::Vector<double, 6> g;
      Eigen::Vector3d tp =
          pose * Eigen::Vector3d(points[i][0], points[i][1], depths[i]);
      double X = tp[0];
      double Y = tp[1];
      double Z = tp[2];
      double ZZ = Z * Z;
      for (int x = -patch_size; x < patch_size; x++)
        for (int y = -patch_size; y < patch_size; y++) {
          Eigen::Vector<double, 6> J;
          double kpx = fx * X / Z + cx;
          double kpy = fy * Y / Z + cy;
          error = GetPixelValue(img_1, points[i][0] + x, points[i][1] + y) -
                  GetPixelValue(img_2, kpx + x, kpy + y);
          Eigen::Vector2d j_pixel;
          Eigen::Matrix<double, 2, 6> j_se;
          j_pixel = -0.5 * Eigen::Vector<double, 2>(
                               GetPixelValue(img_2, kpx + x + 1, kpy + y) -
                                   GetPixelValue(img_2, kpx + x - 1, kpy + y),
                               GetPixelValue(img_2, kpx + x, kpy + y + 1) -
                                   GetPixelValue(img_2, kpx + x, kpy + y - 1));
          j_se << fx / Z, 0, -fx * X / ZZ, fx * X * Y / ZZ,
              fx + fx * X * X / ZZ, -fx * Y / Z, 0, fy / Z, -fy * Y / ZZ,
              -fy - fy * Y * Y, -fy * X * Y / ZZ, fy * X / Z;
          H += J * J.transpose();
          g += -J * error;
        }
      Eigen::Vector<double, 6> delta;
      delta = H.ldlt().solve(g);
      pose = Sophus::SE3d::exp(delta) * pose;
    }
    cout << pose.matrix() << endl;
  }
}

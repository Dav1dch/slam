#include <opencv2/calib3d.hpp>
#define FMT_HEADER_ONLY
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <math.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sophus/se3.hpp>
#include <vector>

using namespace std;
using namespace cv;

vector<DMatch> feature_extraction(Mat img_1, Mat img_2,
                                  vector<KeyPoint> *keypoints_1,
                                  vector<KeyPoint> *keypoints_2);

Point2d pixel2cam(Point2f pt, Mat k);

void bundleAdjustmentGaussNewton(const vector<Eigen::Vector3d> pts_3d,
                                 const vector<Eigen::Vector2d> pts_2d, Mat k,
                                 Sophus::SE3d &pose);

int main(int argc, char **argv) {
  // Eigen::Matrix<int, 3, 3> mat;
  // mat << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  // cout << mat.transpose() * mat << endl;

  if (argc != 4) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], IMREAD_COLOR);
  Mat img_2 = imread(argv[2], IMREAD_COLOR);
  Mat depth_img = imread(argv[3], IMREAD_COLOR);
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  assert(img_1.data != nullptr && img_2.data != nullptr);

  vector<DMatch> matches =
      feature_extraction(img_1, img_2, &keypoints_1, &keypoints_2);

  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Eigen::Vector3d> pts_3d;
  vector<Eigen::Vector2d> pts_2d;

  vector<Point3f> pts_3d1;
  vector<Point2f> pts_2d1;
  for (DMatch m : matches) {
    ushort d = depth_img.ptr<unsigned short>(
        int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0) // bad depth
      continue;
    float dd = d / 5000.0;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    pts_3d.push_back(Eigen::Vector3d(p1.x * dd, p1.y * dd, dd));
    pts_2d.push_back(Eigen::Vector2d(keypoints_2[m.trainIdx].pt.x,
                                     keypoints_2[m.trainIdx].pt.y));

    pts_3d1.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
    pts_2d1.push_back(keypoints_2[m.trainIdx].pt);
  }
  Sophus::SE3d pose;

  Mat r, t;
  solvePnP(pts_3d1, pts_2d1, K, Mat(), r, t, false);
  Mat R;
  cv::Rodrigues(r, R);
  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;

  bundleAdjustmentGaussNewton(pts_3d, pts_2d, K, pose);

  return 0;
}

vector<DMatch> feature_extraction(Mat img_1, Mat img_2,
                                  vector<KeyPoint> *keypoints_1,
                                  vector<KeyPoint> *keypoints_2) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create("BruteForce-Hamming");

  //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1, *keypoints_1);
  detector->detect(img_2, *keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, *keypoints_1, descriptors_1);
  descriptor->compute(img_2, *keypoints_2, descriptors_2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  Mat outimg1;
  drawKeypoints(img_1, *keypoints_1, outimg1, Scalar::all(-1),
                DrawMatchesFlags::DEFAULT);
  imshow("ORB features", outimg1);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> matches;
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  //-- 第四步:匹配点对筛选
  // 计算最小距离和最大距离
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) {
                                  return m1.distance < m2.distance;
                                });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  //-- 第五步:绘制匹配结果
  // Mat img_match;
  // Mat img_goodmatch;
  // drawMatches(img_1, *keypoints_1, img_2, *keypoints_2, matches, img_match);
  // drawMatches(img_1, *keypoints_1, img_2, *keypoints_2, good_matches,
  // img_goodmatch);
  // imshow("all matches", img_match);
  // imshow("good matches", img_goodmatch);
  // waitKey(0);
  return good_matches;
}

Point2d pixel2cam(Point2f pt, Mat k) {
  Point2d ptr;
  ptr.x = (pt.x - k.at<double>(0, 2)) / k.at<double>(0, 0);
  ptr.y = (pt.y - k.at<double>(1, 2)) / k.at<double>(1, 1);
  return ptr;
}

void bundleAdjustmentGaussNewton(const vector<Eigen::Vector3d> pts_3d,
                                 const vector<Eigen::Vector2d> pts_2d, Mat k,
                                 Sophus::SE3d &pose) {

  int iteration = 10;
  double fx = k.at<double>(0, 0);
  double fy = k.at<double>(1, 1);
  double cx = k.at<double>(0, 2);
  double cy = k.at<double>(1, 2);
  Eigen::Vector2d error;

  for (int i = 0; i < iteration; i++) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Vector3d tp;
    for (int j = 0; j < pts_2d.size(); j++) {
      Eigen::Matrix<double, 2, 6> Jt = Eigen::Matrix<double, 2, 6>::Zero();
      Eigen::Vector<double, 2> u = Eigen::Vector<double, 2>::Zero();
      tp = pose * pts_3d.at(j);
      u(0, 0) = fx * tp(0, 0) / tp(2, 0) + cx;
      u(1, 0) = fy * tp(1, 0) / tp(2, 0) + cy;
      error = pts_2d.at(j) - u;

      Jt << fx / tp(2, 0), 0, -fx * tp(0, 0) / pow(tp(2, 0), 2),
          fx * tp(0, 0) * tp(1, 0) / pow(tp(2, 0), 2),
          fx + fx * pow(tp(0, 0), 2) / pow(tp(2, 0), 2),
          -fx * tp(1, 0) / tp(2, 0), 0, fy / tp(2, 0),
          -fy * tp(1, 0) / pow(tp(2, 0), 2),
          -fy - fy * pow(tp(1, 0), 2) / pow(tp(2, 0), 2),
          -fy * tp(0, 0) * tp(1, 0) / pow(tp(2, 0), 2),
          fy * tp(0, 0) / tp(2, 0);

      Jt = -Jt;
      H += Jt.transpose() * Jt;
      g += -Jt.transpose() * error;
    }

    Eigen::Vector<double, 6> deltaX;
    deltaX = H.ldlt().solve(g);
    pose = Sophus::SE3d::exp(deltaX) * pose;
    cout << "iteration" << i << "cost=" << error << endl;
  }
  cout << pose.matrix() << endl;
}

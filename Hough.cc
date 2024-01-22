#include "Hough.h"
#include <iostream>
#include <cmath>

YoloResult::YoloResult(const char *filename) {
  // Loads an image
  src_ = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
  // Check if image is loaded fine
  if(src_.empty()){
    printf(" Error opening image\n");
    printf(" Program Arguments: [image_name -- default %s] \n", filename);
    exit(EXIT_FAILURE);
  }
}

double YoloResult::Dis(const cv::Point2d& x, const cv::Point2d& y) {
  return sqrt((x.x - y.x) * (x.x - y.x) + (x.y - y.y) * (x.y - y.y));
}

double YoloResult::Dis(const cv::Point2d& x, const cv::Vec4i& y) {
  double A = y[3] - y[1], B = y[0] - y[2], C = y[2] * y[1] - y[0] * y[3];
  return abs(A * x.x + B * x.y + C) / sqrt(A * A + B * B);
}

void YoloResult::SetCircle(const cv::Mat& from) {
  cv::cvtColor(from, gray_, cv::COLOR_BGR2GRAY);
  cv::medianBlur(gray_, gray_, 3);

  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(gray_, circles, cv::HOUGH_GRADIENT, 1,
      gray_.rows,  // change this value to detect circles with different distances to each other
      100, 30, gray_.rows / 2.5, gray_.rows / 1.8 // change the last two parameters
      // (min_radius & max_radius) to detect larger circles
      );
  printf(" Circle found: %zu\n", circles.size());
  if (!circles.size()) {
    printf(" No circle found!\n");
    exit(EXIT_FAILURE);
  }

  std::sort(circles.begin(), circles.end(), [&](cv::Vec3f x, cv::Vec3f y) {
      return Dis(cv::Point2d(x[0], x[1]), cv::Point2d(gray_.rows / 2, gray_.cols / 2)) 
      < Dis(cv::Point2d(y[0], y[1]), cv::Point2d(gray_.rows / 2, gray_.cols / 2));
      } );

  center_ = cv::Point2i(circles[0][0], circles[0][1]);
  radi_ = circles[0][2];

  cv::Mat mask = cv::Mat::zeros(from.size(), CV_8UC1);
  cv::circle(mask, center_, radi_, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
  from.copyTo(gray_, mask);
}

void YoloResult::SetPointer(const cv::Mat& from) {
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(from, lines, 1, CV_PI / 180, 80, from.rows / 5, from.rows / 60);
  printf(" Line found: %zu\n", lines.size());

  for (size_t i = 0; i < lines.size(); i++)
  {
    line(from, cv::Point(lines[i][0], lines[i][1]),
        cv::Point(lines[i][2], lines[i][3]), cv::Scalar(100, 100, 100), 2, 8);
  }
  imshow("lines", from);

  sort(lines.begin(), lines.end(), [&](cv::Vec4i x, cv::Vec4i y) {
      return Dis(center_, x) < Dis(center_, y);
//      return std::min(Dis(cv::Point2d(x[0], x[1]), center_), Dis(cv::Point2d(x[2], x[3]), center_))
//      < std::min(Dis(cv::Point2d(y[0], y[1]), center_), Dis(cv::Point2d(y[2], y[3]), center_));
      } );

  if (!lines.size()) {
    printf(" No line found!\n");
    exit(EXIT_FAILURE);
  }
  pointer_ = lines[0];
}

void YoloResult::SetBinary(const cv::Mat& from) {
  Canny(from, dst_, 50, 200, 3);
//  threshold(gray, dst, 110, 255, THRESH_BINARY);
//  bitwise_not(dst, dst);

}

void YoloResult::DilAndEro(cv::Mat& from) {
  const int kerN = 60;
  cv::Mat ker = getStructuringElement(cv::MORPH_RECT, cv::Size(from.rows / kerN, from.rows / kerN));
  cv::dilate(from, from, ker, cv::Point(-1,-1));
  cv::erode(from, from, ker, cv::Point(-1,-1));
}

void YoloResult::GetResult(double& readNum) {
  SetCircle(src_);
  SetBinary(gray_);
  DilAndEro(dst_);
  SetPointer(dst_);
  // circle center
  cv::circle(src_, center_, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
  // circle outline
  cv::circle(src_, center_, radi_, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
  // pointer
  cv::line(src_, cv::Point(pointer_[0], pointer_[1]), cv::Point(pointer_[2], pointer_[3]), cv::Scalar(0, 0, 255), 2, 8);

  cv::imshow("detected pointer", src_);
}

#pragma once

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

class YoloResult {
private:
  cv::Mat src_, gray_, dst_;
  cv::Point2i center_;
  int radi_;
  cv::Vec4i pointer_;
  double Dis(const cv::Point2d&, const cv::Point2d&);
  void SetBinary(const cv::Mat&);
  void SetCircle(const cv::Mat&);
  void SetPointer(const cv::Mat&);
  void DilAndEro(cv::Mat&);
public:
  YoloResult(const char*);
  void GetResult(double&);
};

#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

double dis(double x, double xx, double y, double yy) {
  return sqrt((x - y) * (x - y) + (xx - yy) * (xx - yy));
}

void getCircle(Mat& gray, Vec3i& res) {
  const int NUM_CIRCLE = 1;
  vector<Vec3f> circles;
  HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
      gray.rows / NUM_CIRCLE,  // change this value to detect circles with different distances to each other
      100, 30, gray.rows / 2.5, gray.rows / 1.8 // change the last two parameters
      // (min_radius & max_radius) to detect larger circles
      );
  cout << "circle found: " << circles.size() << endl;

  for (auto i = circles.begin(); i != circles.end(); i++) {
    Vec3i now = *i;
    Point center(now[0], now[1]);
    int borderR = now[2];
    // circle center
    circle(gray, center, 1, Scalar(0, 0, 255), 1, LINE_AA);
    // circle outline
    circle(gray, center, borderR, Scalar(255, 0, 0), 1, LINE_AA);
  }

  sort(circles.begin(), circles.end(), [&](Vec3f x, Vec3f y) {
      return dis(x[0], x[1], gray.rows / 2, gray.cols / 2) < dis(y[0], y[1], gray.rows / 2, gray.cols / 2);
      } );
  res = circles[0];
}

void findPointer(Mat& dst, Vec4i& res) {
  vector<Vec4i> lines;
  HoughLinesP(dst, lines, 1, CV_PI / 180, 80, dst.rows / 5, dst.rows / 60);
  cout << "pointer found: " << lines.size() << endl;
  for (size_t i = 0; i < lines.size(); i++)
  {
    line(dst, Point(lines[i][0], lines[i][1]),
        Point(lines[i][2], lines[i][3]), Scalar(255, 255, 255), 2, 8);
  }

  sort(lines.begin(), lines.end(), [&](Vec4i x, Vec4i y) {
      return min(dis(x[0], x[1], dst.rows / 2, dst.cols / 2), dis(x[2], x[3], dst.rows / 2, dst.cols / 2))
      < min(dis(y[0], y[1], dst.rows / 2, dst.cols / 2), dis(y[2], y[3], dst.rows / 2, dst.cols / 2));
      } );
  if (lines.size())
    res = lines[0];
}

int main(int argc, char** argv)
{
  const char* filename = argc >= 2 ? argv[1] : "img.png";
  // Loads an image
  Mat src = imread(samples::findFile(filename), IMREAD_COLOR);
  // Check if image is loaded fine
  if(src.empty()){
    printf(" Error opening image\n");
    printf(" Program Arguments: [image_name -- default %s] \n", filename);
    return EXIT_FAILURE;
  }

// ----------------------------------------------------

  Mat dst, gray;
  cvtColor(src, gray, COLOR_BGR2GRAY);
  medianBlur(gray, gray, 3);
  Canny(gray, dst, 50, 200, 3);
//  threshold(gray, dst, 110, 255, THRESH_BINARY);
//  bitwise_not(dst, dst);
  imshow("2", dst);

// ----------------------------------------------------

  Vec3i circleBorder;
  getCircle(gray, circleBorder);

  Point center(circleBorder[0], circleBorder[1]);
  int borderR = circleBorder[2];

  // circle center
  circle(src, center, 1, Scalar(0, 0, 255), 1, LINE_AA);
  // circle outline
  circle(src, center, borderR, Scalar(255, 0, 0), 1, LINE_AA);

// ----------------------------------------------------

  Mat mask = Mat::zeros(src.size(), CV_8UC1);
  circle(mask, center, borderR, Scalar(255, 255, 255), FILLED, LINE_AA);
  //imshow("mask", mask);
  Mat cutDst;
  dst.copyTo(cutDst, mask);

// ----------------------------------------------------

  Mat opDst(cutDst);
  dilate(cutDst, opDst, Mat(), Point(-1,-1), 3);
  erode(opDst, opDst, Mat(), Point(-1,-1), 2);
  imshow("opDst", opDst);

  Vec4i pointer;
  findPointer(opDst, pointer);
  line(src, Point(pointer[0], pointer[1]), Point(pointer[2], pointer[3]), Scalar(0, 0, 255), 2, 8);

// ----------------------------------------------------

  imshow("lines", opDst);
  imshow("detected circles", src);
  //imshow("dst", dst);
  waitKey();
  return EXIT_SUCCESS;
}

#include <cstdio>
#include "Hough.h"

int main(const int argc, const char **argv) {
  const char *dir = argc > 1 ? argv[1] : "img.png";
  double res;
  YoloResult img(dir);
  printf("%.3lf", img.GetResult());
  cv::waitKey();
  return 0;
}

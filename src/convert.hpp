#ifndef CONVERT_PDF
#define CONVERT_PDF

#include <iostream>
#include <Magick++.h>
#include <opencv2/opencv.hpp>

using namespace cv;

std::vector<Mat> convert(std::string filePath);
std::vector<Rect> lineBlock(Mat src, std::string destPath);
#endif //CONVERT_PDF

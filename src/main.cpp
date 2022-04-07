#include <iostream>
#include <fstream>
#include <tesseract/baseapi.h>
#include <algorithm>
#include "convert.hpp"

std::vector<std::vector<Rect>> lineBlock(Mat src){
    std::vector<std::vector<Rect>> lineBounds;

    if (src.empty()){
        std::cerr << "[error]: image is empty" << std::endl;
        return lineBounds;
    }

    Mat dest;

    cvtColor(src, src, COLOR_BGR2GRAY);
    src.copyTo(dest);
    src = src < 200;
    GaussianBlur(src, src, Size(9, 9), 0, 0);
    threshold(src, src, 70, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(1, 5));
    dilate(src, src, kernel);

    Mat1f horProj;
    reduce(src, horProj, 1, REDUCE_AVG);
    Mat1b hist = horProj <= 0;

    kernel = getStructuringElement(MORPH_RECT, Size(10, 1));
    dilate(src, src, kernel);
    std::vector<std::vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    std::vector<Rect> rects;
    for (std::vector<Point> contour: contours){
        Rect rect = boundingRect(contour);
        rects.push_back(rect);
    }

    int prevR = -1;
    bool isSpace = true;
    size_t lineNumber = 0;
    for (size_t r = 0; r < src.rows; r++){
        if (isSpace){
            if (!hist(r)){
                isSpace = false;
                prevR = r;
            }
        } else {
            if (hist(r)){
                isSpace = true;
                std::vector<Rect> line;
                do {
                  Rect rect = rects.back();
                  double mid = rect.y + rect.height/2;
                  if (!(mid < r && mid > prevR)) break;
                  line.push_back(rect);
                  rects.pop_back();
                } while (!rects.empty());
                lineBounds.push_back(line);
            }
        }
    }
    return lineBounds;
}

bool rectComparator(Rect r1, Rect r2){
  //comparing in ascending order
  return r1.x < r2.x;
}


int main(int n_args, char ** args){
  if (n_args < 2) {
    std::cout << "Usage: main filename" << std::endl;
    return EXIT_FAILURE;
  }
  std::string filePath = args[1];
  
  size_t start = filePath.rfind('/');
  if (start == std::string::npos){
    start = 0;
  } else {
    start++;
  }
  size_t end = filePath.rfind('.');
  size_t bracket = filePath.rfind('[');
  std::string extension;
  if (bracket == std::string::npos){
    extension = filePath.substr(end+1);
  } else {
    extension = filePath.substr(end+1, bracket-end-1);
  }

  if (extension != "pdf"){
    std::cout << "wrong file extenstion \"." << extension << "\"" << std::endl;
    return EXIT_SUCCESS;
  }

  std::string filename = filePath.substr(start, end-start);
  std::vector<Mat> images =  convert(filePath);

  std::ofstream fileOut;
  fileOut.open(filename + ".txt");

  tesseract::TessBaseAPI *tessOcr = new tesseract::TessBaseAPI();
  tessOcr->Init(NULL, "tha", tesseract::OEM_LSTM_ONLY);
  tessOcr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
  for (size_t ii = 0; ii < images.size(); ii++){
    std::vector<std::vector<Rect>> lines = lineBlock(images[ii]);
    for (std::vector<Rect> rects: lines){
      //sort the list in ascending order
      std::sort(rects.begin(), rects.end(), rectComparator);

      for (Rect rect: rects){
        Mat croped = images[ii](rect);
        tessOcr->SetImage(croped.data, croped.cols, croped.rows, 3, croped.step);
        std::string text = std::string(tessOcr->GetUTF8Text());
        //get rid of new line
        while(!text.empty() && text.back() == '\n') text.pop_back();

        fileOut << text << " ";
      }
      //enter new line
      fileOut << std::endl;
    }
  }
  if (fileOut.is_open()) fileOut.close();
  tessOcr->End();
  return EXIT_SUCCESS;
}

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <tesseract/baseapi.h>

#include "convert.hpp"

std::vector<std::vector<Rect>> lineBlock(Mat src) {
  std::vector<std::vector<Rect>> lineBounds;

  if (src.empty()) {
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

  kernel = getStructuringElement(MORPH_RECT, Size(15, 1));
  dilate(src, src, kernel);
  std::vector<std::vector<Point>> contours;
  findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);


  std::vector<Rect> rects;
  for (std::vector<Point> contour : contours) {
    Rect rect = boundingRect(contour);
    rects.push_back(rect);
  }

  int prevR = -1;
  bool isSpace = true;
  size_t lineNumber = 0;
  for (size_t r = 0; r < src.rows; r++) {
    if (isSpace) {
      if (!hist(r)) {
        isSpace = false;
        prevR = r;
      }
    } else {
      if (hist(r)) {
        isSpace = true;
        std::vector<Rect> line;
        do {
          Rect rect = rects.back();
          double mid = rect.y + rect.height / 2;

          if (!(mid < r && mid > prevR)) {
            break;
          }

          line.push_back(rect);
          rects.pop_back();
        } while (!rects.empty());

        lineBounds.push_back(line);
      }
    }
  }
  return lineBounds;
}

std::vector<int> mergeImages(std::vector<Mat> images) {
  Mat gray;
  Mat1f temp;
  Mat1f hist;
  Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 5));

  for (size_t i = 0; i < images.size(); i++) {
    cvtColor(images[i], gray, COLOR_RGB2GRAY);
    dilate(gray < 200, gray, kernel);
    reduce(gray, temp, 0, REDUCE_AVG);
    if (hist.empty()) {
      temp.copyTo(hist);
    } else {
      hist = hist + temp;
    }
  }

  hist = hist / images.size();

  Mat1b histB = hist < 5;
  Mat dest;
  images[2].copyTo(dest);

  std::vector<int> column;
  for (size_t i = 0; i < images[0].cols - 1; i++) {
    if ((!histB(i) && histB(i + 1)))
      column.push_back(i);
  }

  return column;
}

bool rectComparator(Rect r1, Rect r2) {
  //comparing in ascending order
  return r1.x < r2.x;
}

std::string textBoxeRow(Rect rect, std::string text) {
  std::string row;
  row.append(text);
  row.push_back('\t');
  row.append(std::to_string(rect.tl().x));
  row.push_back('\t');
  row.append(std::to_string(rect.tl().y));
  row.push_back('\t');
  row.append(std::to_string(rect.br().x));
  row.push_back('\t');
  row.append(std::to_string(rect.br().y));
  row.push_back('\n');
  return row;
}

int main(int n_args, char** args) {
  if (n_args < 2) {
    std::cout << "Usage: main filename" << std::endl;
    return EXIT_FAILURE;
  }
  std::string filePath = args[1];

  size_t start = filePath.rfind('/');
  if (start == std::string::npos) {
    start = 0;
  } else {
    start++;
  }
  size_t end = filePath.rfind('.');
  size_t bracket = filePath.rfind('[');
  std::string extension;
  if (bracket == std::string::npos) {
    extension = filePath.substr(end + 1);
  } else {
    extension = filePath.substr(end + 1, bracket - end - 1);
  }

  if (extension != "pdf") {
    std::cout << "wrong file extenstion \"." << extension << "\"" << std::endl;
    return EXIT_FAILURE;
  }

  std::string filename = filePath.substr(start, end - start);
  std::vector<Mat> images = convert(filePath);
  std::vector<int> columnPositions = mergeImages(images);

  std::ofstream fileOut, textBoxes;
  fileOut.open(filename + ".csv");
  textBoxes.open(filename + "_text_boxes" + ".csv");

  tesseract::TessBaseAPI* tessOcr = new tesseract::TessBaseAPI();
  tessOcr->Init("./tessdata/", "tha", tesseract::OEM_LSTM_ONLY);
  tessOcr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);

  for (size_t i = 0; i < images.size(); i++) {
    std::vector<std::vector<Rect>> lines = lineBlock(images[i]);
    bool table = true;
    std::vector<std::string> row;
    for (size_t l = 0; l < lines.size(); l++) {
      std::vector<Rect> rects = lines[l];
      //sort the list in ascending order
      std::sort(rects.begin(), rects.end(), rectComparator);

      //malloc column
      row.assign(columnPositions.size()+1, std::string());

      for (Rect rect : rects) {
        Mat croped = images[i](rect);
        tessOcr->SetImage(croped.data, croped.cols, croped.rows, 3, croped.step);
        std::string text = std::string(tessOcr->GetUTF8Text());
        //get rid of new line
        while (!text.empty() && text.back() == '\n') text.pop_back();
        textBoxes << std::to_string(i) << "\t"; //page number
        textBoxes << std::to_string(l) << "\t"; //line number
        textBoxes << textBoxeRow(rect, text);

        //determine column
        int col = 0;
        for (size_t j = 0;
          j < columnPositions.size() &&
          rect.x > columnPositions[j]; j++) {
          col = j+1;
        }

        //feed a whitespace to non-empty string
        if (!row[col].empty()) {
          row[col].push_back(' ');
        }
        row[col].append(text);
      }

      if (row.size()) {
        for (size_t j = 0; j < row.size()-1; j++) {
          fileOut << row[j];

          //put a comma
          if (j + 1 < row.size()-1) {
            fileOut.put(',');
          }
        }
        //feed a new line
        fileOut.put('\n');
      }
    }
  }

  //close output file
  if (fileOut.is_open()) {
    fileOut.close();
  }
  if (textBoxes.is_open()) {
    textBoxes.close();
  }

  tessOcr->End();

  return EXIT_SUCCESS;
}

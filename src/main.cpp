#include <iostream>
#include <fstream>
#include <tesseract/baseapi.h>
#include "convert.hpp"

int main(){
  std::vector<Mat> images =  convert("./vote-log/20190823113634A11.pdf[1]");
  std::ofstream fileOut;
  fileOut.open("20190823113634A11[1].txt");

  tesseract::TessBaseAPI *tessOcr = new tesseract::TessBaseAPI();
  tessOcr->Init(NULL, "tha", tesseract::OEM_LSTM_ONLY);
  tessOcr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
  for (size_t ii = 0; ii < images.size(); ii++){
    std::vector<Rect> rects = lineBlock(images[ii], "20190823113634A11[1]_" + std::to_string(ii));
    for (Rect rect: rects){
      Mat croped = images[ii](rect);
      tessOcr->SetImage(croped.data, croped.cols, croped.rows, 3, croped.step);
      fileOut << std::string(tessOcr->GetUTF8Text());
    }
  }
  if (fileOut.is_open()) fileOut.close();
  tessOcr->End();
  return EXIT_SUCCESS;
}

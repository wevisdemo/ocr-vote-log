#include "convert.hpp"

std::vector<Mat> convert(std::string filePath){
  std::vector<Magick::Image> imageList;
  Magick::ReadOptions options;
  options.density(Magick::Geometry(190, 0));
  readImages(&imageList, filePath, options);

  std::vector<Mat> images;
  for (int i = 0; i < imageList.size(); i++) {
    Magick::Image magickImage = imageList.at(i);

    //https://stackoverflow.com/a/41847887/15538794
    const int w = magickImage.columns() - 200;
    const int h = magickImage.rows() - 230;
    Mat image(h, w, CV_8UC3);
    //write Magick Image to opencv Mat
    magickImage.write(120, 100, w, h, "BGR", Magick::CharPixel, image.data);
    images.push_back(image);
  }
  return images;
}

std::vector<Rect> lineBlock(Mat src, std::string destPath){
    std::vector<Rect> lineBounds;

    if (src.empty()){
        std::cerr << "[error]: image " << destPath << " is empty" << std::endl;
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
                line(dest, Point(0, r), Point(dest.cols, r), Scalar(0, 0, 255));
                line(dest, Point(0, prevR), Point(dest.cols, prevR), Scalar(255, 0, 0));
                do {
                  Rect rect = rects.back();
                  double mid = rect.y + rect.height/2;
                  if (!(mid < r && mid > prevR)) break;
                  lineBounds.push_back(rect);
                  rects.pop_back();
                  rectangle(dest, rect, Scalar(0, 0, 255));
                  putText(dest, std::to_string(lineNumber), rect.tl(), FONT_HERSHEY_COMPLEX,  .5, Scalar(0, 0, 255));
                } while (!rects.empty());
                lineNumber++;
            }
        }
    }

    imwrite(destPath + ".jpg", dest);
    return lineBounds;
}
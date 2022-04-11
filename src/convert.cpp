#include "convert.hpp"

std::vector<Mat> convert(std::string filePath) {
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
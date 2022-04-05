#include "convert.hpp"

void convert(std::string filePath){
  std::vector<Magick::Image> imageList;
  Magick::ReadOptions options;
  options.density(Magick::Geometry(170, 0));
  readImages(&imageList, filePath, options);

  for (int i = 0; i < imageList.size(); i++) {
    std::string fileName = "20190823113634A11_" + std::to_string(i) + ".png";
    Magick::Image magickImage = imageList.at(i);
    
    //https://stackoverflow.com/a/41847887/15538794
    const int w = magickImage.columns();
    const int h = magickImage.rows();
    Mat image(h, w, CV_8UC3);
    //write Magick Image to opencv Mat
    magickImage.write(0, 0, w, h, "BGR", Magick::CharPixel, image.data);
  }
}
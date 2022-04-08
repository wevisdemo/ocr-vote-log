PDF_FILE = ./vote-log/20190823113634A11.pdf[0]
main_built := ./built/main.o

main: convert
	g++ -O3 -std=c++11 ./src/main.cpp ./built/convert.o -o $(main_built) `pkg-config --cflags --libs tesseract opencv4 Magick++` 
	$(main_built) $(PDF_FILE)

convert:
	g++ -O3 -std=c++11 -c ./src/convert.cpp `pkg-config --cflags --libs Magick++` -o ./built/convert.o
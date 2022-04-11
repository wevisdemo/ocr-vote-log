PDF_FILE = ./vote-log/20190823113634A11.pdf
main_built := ./main.o

main: convert
	g++ -O3 -std=c++11 ./src/main.cpp ./convert.o -o $(main_built) `pkg-config --cflags --libs tesseract opencv4 Magick++` 
	$(main_built) $(PDF_FILE)

convert:
	g++ -O3 -std=c++11 -c ./src/convert.cpp `pkg-config --cflags --libs Magick++` -o ./convert.o
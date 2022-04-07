main_built := ./built/main.o

main: convert
	g++ -O3 -std=c++11 ./src/main.cpp ./built/convert.o -o $(main_built) `pkg-config --cflags --libs tesseract opencv4 Magick++` 
	$(main_built) ./vote-log/20190823113634A11.pdf[1]

convert:
	g++ -O3 -std=c++11 -c ./src/convert.cpp -o ./built/convert.o `pkg-config --cflags --libs Magick++`
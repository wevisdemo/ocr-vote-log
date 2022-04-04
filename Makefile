main_built := ./built/main

main:
	g++ -O3 -std=c++11 ./src/main.cpp `pkg-config --cflags --libs tesseract opencv4` -o $(main_built)
	$(main_built)
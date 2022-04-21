PDF_FILE = ./vote-log/20190823113634A11.pdf[0-2]
main_built := ./main.o

all: main.o
	$(main_built) $(PDF_FILE)

main.o: convert.o csv.o
	g++ -O3 -std=c++11 ./src/main.cpp ./convert.o -o $(main_built) `pkg-config --cflags --libs tesseract opencv4 Magick++` 

convert.o:
	g++ -O3 -std=c++11 -c ./src/convert.cpp `pkg-config --cflags --libs opencv4 Magick++` -o ./convert.o

csv.o:
	g++ -std=c++11 -c ./src/csv.h -o ./csv.o

cross:
	/usr/local/bin/python3 cross.py PeopleVote.csv 20220209190321A17.csv votelog.__82
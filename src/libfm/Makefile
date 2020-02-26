BIN_DIR := ../../bin/

FLAGS := -O3 -Wall -std=c++17
INCLUDES := `python3 -m pybind11 --includes` -I ./eigen/

OBJECTS := \
	libfm.o \
	tools/transpose.o \
	tools/convert.o \

all: libFM transpose convert

libFM: libfm.o
	mkdir -p $(BIN_DIR)
	g++ $(FLAGS) libfm.o -o $(BIN_DIR)libFM

%.o: %.cpp
	g++ $(FLAGS) $(INCLUDES) -c $< -o $@

clean:	clean_lib
	mkdir -p $(BIN_DIR)
	rm -f $(BIN_DIR)libFM $(BIN_DIR)convert $(BIN_DIR)transpose

clean_lib:
	rm -f $(OBJECTS)


transpose: tools/transpose.o
	mkdir -p $(BIN_DIR)
	g++ $(FLAGS) tools/transpose.o -o $(BIN_DIR)transpose

convert: tools/convert.o
	mkdir -p $(BIN_DIR)
	g++ $(FLAGS) tools/convert.o -o $(BIN_DIR)convert


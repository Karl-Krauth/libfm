BIN_DIR := ../../bin/
PY_DIR := ../../pyfm/

FLAGS := -O3 -Wall -std=c++17 -Wl,-undefined,dynamic_lookup
INCLUDES := `python3 -m pybind11 --includes` -I ./eigen/

OBJECTS := \
	libfm.o \
	tools/transpose.o \
	tools/convert.o \

all: libFM transpose convert pyfm

libFM: libfm.o
	mkdir -p $(BIN_DIR)
	g++ $(FLAGS) libfm.o -o $(BIN_DIR)libFM

pyfm:
	g++ $(FLAGS) -shared -fPIC $(INCLUDES) pyfm.cpp -o $(PY_DIR)pyfm`python3-config --extension-suffix`

%.o: %.cpp
	g++ $(FLAGS) $(INCLUDES) -c $< -o $@

clean:	clean_lib
	mkdir -p $(BIN_DIR)
	rm -f $(BIN_DIR)libFM $(BIN_DIR)convert $(BIN_DIR)transpose
	rm -f $(PY_DIR)pyfm`python3-config --extension-suffix`

clean_lib:
	rm -f $(OBJECTS)

transpose: tools/transpose.o
	mkdir -p $(BIN_DIR)
	g++ $(FLAGS) tools/transpose.o -o $(BIN_DIR)transpose

convert: tools/convert.o
	mkdir -p $(BIN_DIR)
	g++ $(FLAGS) tools/convert.o -o $(BIN_DIR)convert


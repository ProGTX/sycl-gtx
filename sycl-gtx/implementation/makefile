CC=clang++

INC=-I$(INTELOCLSDKROOT)include
CFLAGS=-c -Wall -std=c++11 $(INC)

LIB=-L$(INTELOCLSDKROOT)lib\x64
LDFLAGS=$(LIB) -lOpenCL

SOURCES=data_ref.cpp kernel_name.cpp synchronizer.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=implementation

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

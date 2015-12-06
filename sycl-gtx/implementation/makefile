CC=clang++

INC=-I$(INTELOCLSDKROOT)include
CFLAGS=-c -Wall -std=c++11 $(INC)

LIB=-L$(INTELOCLSDKROOT)lib\x64
LDFLAGS=$(LIB) -lOpenCL

SOURCES=data_ref.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=data_ref

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

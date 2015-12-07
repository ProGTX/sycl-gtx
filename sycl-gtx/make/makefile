CC=clang++

INC=-I$(INTELOCLSDKROOT)include
CFLAGS=-c -Wall -std=c++11 $(INC)

LIB=-L$(INTELOCLSDKROOT)lib\x64
LDFLAGS=$(LIB) -lOpenCL

FOLDERS:=accessor functions ranges
FOLDERS:=$(foreach d, $(FOLDERS), specification/$(d)) specification src_handlers
FOLDERS:=$(foreach d, $(FOLDERS), implementation/$(d)) implementation

SOURCES=$(foreach d, $(FOLDERS), $(wildcard ../$(d)/*.cpp))
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=sycl-gtx

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS)
#@echo --- Linking target: $@. First dependency is: $<
#$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	@echo --- Building target: $@. First dependency is: $<
	$(CC) $(CFLAGS) $< -o $(notdir $@)

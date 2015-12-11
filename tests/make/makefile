CC=clang++
TARGET=Release

INC=-I$(INTELOCLSDKROOT)include -I../../sycl-gtx
CFLAGS=-c -Wall -std=c++11 $(INC)

ifeq ($(TARGET),Release)
# TODO: Clang 3.7.0 fails with -O2 on Windows
CFLAGS+= -O1
else
CFLAGS+= -O0
endif

SOURCES= 				\
	../main.cpp			\
    ../tests/test1.cpp	\
    ../tests/test2.cpp	\
    ../tests/test3.cpp	\
    ../tests/test4.cpp	\
    ../tests/test5.cpp	\
    ../tests/test6.cpp	\
    ../tests/test7.cpp	\
    ../tests/test8.cpp	\
    ../tests/test9.cpp	\
    ../tests/test10.cpp	\
    ../tests/test11.cpp

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=tests

ifeq ($(OS),Windows_NT)
RM=del
OBJECTS_RM=$(subst /,\, $(OBJECTS))
else
RM=rm
OBJECTS_RM=$(OBJECTS)
endif

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)

.cpp.o:
	@echo --- Building target: $@. First dependency is: $<
	$(CC) $(CFLAGS) $< -o $@

clean:
	$(RM) $(EXECUTABLE)
	$(RM) $(OBJECTS_RM)
